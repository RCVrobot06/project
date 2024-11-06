import sys
from datetime import datetime
import os
import cv2
import numpy as np

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, saveImage,
                          cudaAllocMapped, cudaToNumpy, cudaFromNumpy,
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

    # 각 탐지된 객체의 바운딩 박스를 개선
    for detection in detections:
        # 탐지된 바운딩 박스
        x1, y1, x2, y2 = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        roi = edges[y1:y2, x1:x2]  # 바운딩 박스 내의 ROI 영역에서 에지 감지

        # 컨투어 찾기
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 가장 큰 컨투어를 찾아 바운딩 박스 조정
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            refined_box = (x1 + x, y1 + y, x1 + x + w, y1 + y + h)
            print(f"Refined bounding box: {refined_box}")
            
            # 객체 탐지 표시 업데이트
            cv2.rectangle(img_np, (refined_box[0], refined_box[1]), 
                          (refined_box[2], refined_box[3]), (255, 0, 0), 2)

    # Numpy 배열을 CUDA 메모리로 다시 변환 후 렌더링
    imgCuda = cudaFromNumpy(img_np)
    output.Render(imgCuda)

    # 스트리밍 상태 확인
    if not input.IsStreaming() or not output.IsStreaming():
        break
