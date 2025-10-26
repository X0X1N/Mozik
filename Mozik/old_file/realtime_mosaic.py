
import cv2
import numpy as np

# OpenCV에서 제공하는 기본 얼굴 탐지 모델 로드
# 이 파일은 OpenCV 설치 시 함께 제공됩니다.
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError("얼굴 탐지 모델(haarcascade_frontalface_default.xml)을 로드할 수 없습니다.")
except Exception as e:
    print(e)
    print("OpenCV 데이터 파일 경로를 확인하거나, OpenCV가 올바르게 설치되었는지 확인하세요.")
    exit()

# 웹캠 연결 (0은 기본 웹캠을 의미)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
    exit()

print("웹캠이 성공적으로 연결되었습니다. 'q' 키를 누르면 종료됩니다.")

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break

    # 얼굴 탐지를 위해 프레임을 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지 수행
    # scaleFactor: 이미지 피라미드에서 각 단계의 축소 비율
    # minNeighbors: 얼굴로 탐지하기 위해 필요한 최소 이웃 사각형 수
    # minSize: 탐지할 얼굴의 최소 크기
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 탐지된 각 얼굴에 대해 모자이크 처리
    for (x, y, w, h) in faces:
        # 얼굴 영역(Region of Interest, ROI)을 추출
        face_roi = frame[y:y+h, x:x+w]

        # 모자이크 처리
        # 1. 얼굴 영역을 작은 크기로 축소
        # 2. 축소된 이미지를 다시 원래 크기로 확대 (이때 INTER_NEAREST 보간법을 사용해 픽셀화 효과를 줌)
        pixel_level = 10 # 모자이크의 픽셀 크기 (값이 작을수록 모자이크가 더 세밀해짐)
        
        # roi의 높이와 너비
        roi_h, roi_w, _ = face_roi.shape

        # roi를 작은 크기로 축소
        small_roi = cv2.resize(face_roi, (pixel_level, pixel_level), interpolation=cv2.INTER_LINEAR)
        
        # 축소된 roi를 다시 원래 크기로 확대하여 모자이크 효과 생성
        mosaic_roi = cv2.resize(small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        # 원본 프레임에 모자이크 처리된 얼굴 영역을 다시 붙여넣기
        frame[y:y+h, x:x+w] = mosaic_roi

    # 결과 프레임을 화면에 표시
    cv2.imshow('Real-time Face Mosaic', frame)

    # 'q' 키를 누르면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용이 끝난 후 자원 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")
