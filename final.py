import sys
from datetime import datetime
import os
import cv2
import numpy as np
import random

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, cudaToNumpy, cudaFromNumpy,
                          cudaDeviceSynchronize)

# 기본 설정
input = videoSource("v4l2:///dev/video0")  # 카메라 입력
output = videoOutput("rtsp://localhost:8554/mystream")  # 화면에 출력
output_dir = "output_edges"  # 객체별 에지 이미지를 저장할 폴더

# 출력 폴더 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

net = detectNet("ssd-mobilenet-v2", threshold=0.5)

# 객체 ID와 에지 저장 여부를 관리하는 딕셔너리
saved_edges = {}
# 각 객체별로 고유한 색을 사용하기 위한 딕셔너리
color_map = {}

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

    # 객체 탐지 수행
    detections = net.Detect(imgCuda)

    # 각 탐지된 객체의 컨투어 기반 경계를 찾아 표시 및 저장
    for detection in detections:
        # 객체의 고유 ID로 변화를 추적
        obj_id = int(detection.ClassID)        # ROI 설정을 위한 객체 위치 좌표 설정
        x1, y1, x2, y2 = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
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
            if obj_id not in color_map:
                # RGB 컬러를 랜덤으로 생성하여 color_map에 추가
                color_map[obj_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # 객체의 색상 가져오기
            color = color_map[obj_id]

            # 객체 모양에 맞춘 다각형 그리기 (각 객체는 고유의 색상으로 그려짐)
            cv2.drawContours(img_np, [largest_contour], -1, color, 2)

            # 객체 이름 표시
            label = net.GetClassDesc(obj_id)
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 이전에 저장한 적이 없는 객체라면 에지 이미지를 저장
            if obj_id not in saved_edges:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                edge_filename = os.path.join(output_dir, f"object_{obj_id}_{timestamp}.jpg")
                cv2.imwrite(edge_filename, roi)  # ROI로 잘라낸 에지 이미지를 저장

                saved_edges[obj_id] = True  # 객체 ID를 기록하여 중복 저장 방지
                print(f"Saved edge image: {edge_filename}")

    # Numpy 배열을 CUDA 메모리로 다시 변환 후 렌더링
    imgCuda = cudaFromNumpy(img_np)
    output.Render(imgCuda)

    # 스트리밍 상태 확인
    if not input.IsStreaming() or not output.IsStreaming():
        break
