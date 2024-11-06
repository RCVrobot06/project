import cv2
import numpy as np
from datetime import datetime
import os
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")  # 모델 경로 또는 'yolov8n.pt'와 같은 기본 YOLOv8 모델 사용

# 출력 폴더 생성
output_dir = "output_edges"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 카메라 설정
cap = cv2.VideoCapture(0)  # 기본 카메라 사용 (Jetson에서는 "v4l2:///dev/video0" 형식 사용 가능)

saved_edges = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8을 사용하여 객체 탐지
    results = model(frame)

    # 에지 검출을 위한 샤프닝 및 Canny 에지 처리
    sharpened = cv2.filter2D(frame, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 에지 연결을 위한 확장 (Dilate)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    for box in results.boxes:
        # 바운딩 박스 좌표 가져오기
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # ROI 내에서 에지 감지
        roi = edges[y1:y2, x1:x2]

        # 컨투어 찾기
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 작은 잡음 제거 및 가장 큰 컨투어로 바운딩 박스 표시
        min_contour_area = 100
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            largest_contour += np.array([x1, y1])  # ROI를 전체 이미지 좌표로 변환

            # 객체 모양에 맞춘 다각형 그리기
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            # 객체 에지 이미지를 저장 (처음 감지 시에만)
            obj_id = int(box.cls[0])
            if obj_id not in saved_edges:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                edge_filename = os.path.join(output_dir, f"object_{obj_id}_{timestamp}.jpg")
                cv2.imwrite(edge_filename, roi)
                saved_edges[obj_id] = True
                print(f"Saved edge image: {edge_filename}")

    # 결과 표시
    cv2.imshow("YOLOv8 Detection with Edges", frame)
    
    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
