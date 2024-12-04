import sys
from datetime import datetime
import os
import cv2
import numpy as np
import random
import colorsys
import time

from ultralytics import YOLO
from jetson_utils import (videoSource, videoOutput, cudaToNumpy, cudaFromNumpy,
                          cudaDeviceSynchronize)

# 기본 설정
input = videoSource("v4l2:///dev/video0")  # 카메라 입력
output = videoOutput("rtsp://localhost:8554/mystream")  # 화면에 출력

# YOLOv5 모델 로드
yolo_model = YOLO("yolov5m.pt")

# 객체별로 고유한 색을 사용하기 위한 딕셔너리
color_map = {}

# 중간값을 사용하여 색상을 고르게 분배하기 위한 큐
color_values = [0.5]

def get_next_color_value():
    # 큐에서 중간값을 꺼내서 반환하고, 다음 중간값 두 개를 추가
    value = color_values.pop(0)
    color_values.append(value / 2)
    color_values.append(value + (1 - value) / 2)
    return value

def get_distinct_color():
    # 중간값을 사용하여 색상 값을 얻어옴
    hue = get_next_color_value()
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)  # HSV 값을 RGB로 변환
    return int(r * 255), int(g * 255), int(b * 255)

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    imgCuda = input.Capture()
    
    if imgCuda is None:  # 입력 이미지가 없으면 다음 루프
        continue
    
    # CUDA 이미지에서 Numpy 배열로 변환
    img_np = cudaToNumpy(imgCuda)

    # 샤프닝 필터 적용
    sharpened = cv2.filter2D(img_np, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    # Gaussian 블러 적용 후 Canny 에지 감지
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 에지 연결을 위한 확장 (Dilate)
    kernel = np.ones((3, 3), np.uint8)  # 3x3 커널을 사용한 확장
    edges = cv2.dilate(edges, kernel, iterations=1)

    # YOLOv5 객체 탐지 수행 
    results = yolo_model(img_np)
    detections = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표 가져오기
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # 클래스 ID 가져오기

    # 각 탐지된 객체의 컨투어 기반 경계를 찾아 표시 및 저장
    for i, detection in enumerate(detections):
        # 객체의 고유 ID로 변화를 추적
        obj_id = class_ids[i]

        # 탐지된 바운딩 박스 가져오기
        x1, y1, x2, y2 = map(int, detection)
        
        # 바운딩 박스 내 ROI에서 에지 감지된 부분만 추출
        roi = edges[y1:y2, x1:x2]

        # ROI에서 컨투어 찾기
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 최소 면적을 기준으로 작은 잡음 제거
        min_contour_area = 100  # 최소 면적 (픽셀 단위)

        # 유효한 컨투어만 필터링하여 객체 모양을 정확히 표시
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)

            # 컨투어를 원래 이미지 좌표로 변환
            largest_contour += np.array([x1, y1])  # ROI 좌표를 전체 이미지 좌표로 변환

            # 객체 ID에 대해 고유 색상을 할당 (이미 있으면 기존 색상 사용)
            if obj_id == 0 and obj_id not in color_map:
                # 'person' 객체는 검정색으로 설정
                color_map[obj_id] = (0, 0, 0)
            
            if obj_id not in color_map:
                # 중간값을 사용하여 색상을 고르게 할당
                color_map[obj_id] = get_distinct_color()
            
            # 객체의 색상 가져오기
            color = color_map[obj_id]

            # 객체 모양에 맞춘 다각형 그리기 (각 객체는 고유의 색상으로 그려짐)
            cv2.drawContours(img_np, [largest_contour], -1, color, 2)

            # 객체 이름 표시
            label = yolo_model.names[obj_id]
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # 프레임 카운트 증가
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0
    
    # FPS 표시 (우측 상단)
    cv2.putText(img_np, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Numpy 배열을 CUDA 메모리로 다시 변환 후 렌더링
    imgCuda = cudaFromNumpy(img_np)
    output.Render(imgCuda)

    # 스트리밍 상태 확인
    if not input.IsStreaming() or not output.IsStreaming():
        break
