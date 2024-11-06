import sys
from datetime import datetime
import os
import time
import cv2
import numpy as np

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
prev_detections = []  # 이전 디텍션 결과를 저장할 리스트
last_detection_time = time.time()  # 마지막 디텍션 시간을 기록
detection_display_duration = 2.0  # 디텍션 결과를 유지할 시간 (초)

while True:
    imgCuda = input.Capture()
    
    if imgCuda is None:  # 입력 이미지가 없으면 다음 루프
        continue
    
    # CUDA 이미지에서 Numpy 배열로 변환
    img_np = cudaToNumpy(imgCuda)

    # 2초마다 디텍션 수행
    current_time = time.time()
    if current_time - last_detection_time >= 2.0:
        # 샤프닝 필터 적용
        sharpened = cv2.filter2D(img_np, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

        # Gaussian 블러 적용 후 Canny 에지 감지
        blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # 객체 탐지 수행
        detections = net.Detect(imgCuda)
        
        # 새로운 탐지 결과로 prev_detections를 업데이트
        prev_detections = []  # 리스트 초기화 대신 새로운 컨투어로 갱신
        last_detection_time = current_time  # 마지막 디텍션 시간 업데이트

        # 각 탐지된 객체의 컨투어 기반 경계를 찾아 표시 및 저장
        for detection in detections:
            # 객체의 고유 ID로 변화를 추적
            obj_id = int(detection.ClassID)

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

                # 새로운 컨투어를 prev_detections에 추가
                prev_detections.append(largest_contour)

                # 객체 에지 이미지를 저장 (처음 감지 시에만)
                if obj_id not in saved_edges:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    edge_filename = os.path.join(output_dir, f"object_{obj_id}_{timestamp}.jpg")
                    cv2.imwrite(edge_filename, roi)  # ROI로 잘라낸 에지 이미지를 저장
                    saved_edges[obj_id] = True  # 객체 ID를 기록하여 중복 저장 방지
                    print(f"Saved edge image: {edge_filename}")

    # 이전 또는 현재 프레임의 모든 컨투어를 화면에 표시
    for contour in prev_detections:
        cv2.drawContours(img_np, [contour], -1, (0, 255, 0), 2)

    # Numpy 배열을 CUDA 메모리로 다시 변환 후 렌더링
    imgCuda = cudaFromNumpy(img_np)
    output.Render(imgCuda)

    # 스트리밍 상태 확인
    if not input.IsStreaming() or not output.IsStreaming():
        break
