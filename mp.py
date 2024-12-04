import sys
from datetime import datetime
import os
import cv2
import numpy as np
import random
import colorsys

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

def get_distinct_color(idx, total_colors):
    # HSV 색상 공간에서 색상(Hue) 값을 고르게 분할
    hue = (idx * 360 / total_colors) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

total_objects = 20  # 감지될 수 있는 객체의 최대 수를 예상하여 설정

while True:
    imgCuda = input.Capture()
    
    if imgCuda is None:  # 입력 이미지가 없으면 다음 루프
        continue
    
    # CUDA 이미지에서 Numpy 배열로 변환
    img_np = cudaToNumpy(imgCuda)

    # 샤프닝 필터 적용
    sharpened = cv2.filter2D(img_np, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    # Morphological Gradient를 사용한 에지 감지
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3x3 커널 사용
    gradient = cv2.morphologyEx(sharpened, cv2.MORPH_GRADIENT, kernel)

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
        roi = gradient[y1:y2, x1:x2]

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
            if obj_id not in color_map:
                # HSV 색상 분할 방식 사용
                color_map[obj_id] = get_distinct_color(len(color_map), len(color_map) + 1)
            
            # 객체의 색상 가져오기
            color = color_map[obj_id]

            # 객체 모양에 맞춘 다각형 그리기 (각 객체는 고유의 색상으로 그려짐)
            cv2.drawContours(img_np, [largest_contour], -1, color, 2)

            # 객체 이름 표시
            label = yolo_model.names[obj_id]
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # Numpy 배열을 CUDA 메모리로 다시 변환 후 렌더링
    imgCuda = cudaFromNumpy(img_np)
    output.Render(imgCuda)

    # 스트리밍 상태 확인
    if not input.IsStreaming() or not output.IsStreaming():
        break
