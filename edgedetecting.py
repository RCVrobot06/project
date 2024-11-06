import sys
from datetime import datetime
import os
import cv2
import numpy as np

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, cudaToNumpy, cudaFromNumpy,
                          cudaDeviceSynchronize)

# 기본 설정
input = videoSource("v4l2:///dev/video0")  # 카메라 입력
output = videoOutput("rtsp://localhost:8554/mystream")  # 화면에 출력

net = detectNet("ssd-mobilenet-v2", threshold=0.5)

while True:
    imgCuda = input.Capture()
    
    if imgCuda is None:  # 입력 이미지가 없으면 다음 루프
        continue
    
    # CUDA 이미지에서 Numpy 배열로 변환
    img_np = cudaToNumpy(imgCuda)

    # 샤프닝 필터 적용
    sharpened = cv2.filter2D(img_np, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    # Canny 에지 감지 적용
    edges = cv2.Canny(sharpened, threshold1=100, threshold2=200)

    # 객체 탐지 수행
    detections = net.Detect(imgCuda)

    # 각 탐지된 객체의 컨투어 기반 경계를 찾아 표시
    for detection in detections:
        # 탐지된 바운딩 박스 가져오기
        x1, y1, x2, y2 = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        
        # 바운딩 박스 내 ROI에서 에지 감지된 부분만 추출
        roi = edges[y1:y2, x1:x2]

        # ROI에서 컨투어 찾기
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 가장 큰 컨투어를 찾아 객체 모양을 정확히 표시
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 컨투어를 원래 이미지 좌표로 변환
            largest_contour += np.array([x1, y1])  # ROI 좌표를 전체 이미지 좌표로 변환

            # 객체 모양에 맞춘 다각형 그리기
            cv2.drawContours(img_np, [largest_contour], -1, (0, 255, 0), 2)

            # 선택적으로 컨투어를 감싸는 바운딩 박스를 출력
            x, y, w, h = cv2.boundingRect(largest_contour)
            refined_box = (x1 + x, y1 + y, x1 + x + w, y1 + y + h)
            cv2.rectangle(img_np, (refined_box[0], refined_box[1]), 
                          (refined_box[2], refined_box[3]), (255, 0, 0), 2)

            print(f"Refined bounding box: {refined_box}")

    # Numpy 배열을 CUDA 메모리로 다시 변환 후 렌더링
    imgCuda = cudaFromNumpy(img_np)
    output.Render(imgCuda)

    # 스트리밍 상태 확인
    if not input.IsStreaming() or not output.IsStreaming():
        break