import cv2
import numpy as np

# OpenCV에서 제공하는 기본 얼굴 탐지 모델 로드
try:
    # haarcascades 파일 경로를 시스템에 따라 조정해야 할 수도 있습니다.
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        # 파일 로드 실패 시 예외 발생
        raise IOError("얼굴 탐지 모델(haarcascade_frontalface_default.xml)을 로드할 수 없습니다.")
except Exception as e:
    print(f"오류: {e}")
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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 탐지된 각 얼굴에 대해 둥근 블러(흐림) 처리
    for (x, y, w, h) in faces:
        # 1. 얼굴 영역(ROI) 추출
        face_roi = frame[y:y+h, x:x+w]
        
        # 2. 둥근 마스크 생성 (얼굴 ROI 크기의 검은색 배경)
        # 마스크는 흑백 이미지여야 하므로, 2차원(높이, 너비)으로 생성
        mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
        
        # 3. 마스크 위에 흰색 원 그리기 (흰색(255) 부분이 블러 처리될 영역)
        (cx, cy) = (w // 2, h // 2)  # 원의 중심
        radius = min(w, h) // 2      # 원의 반지름 (사각형의 짧은 변 길이 절반)
        cv2.circle(mask, (cx, cy), radius, 255, -1) # -1은 원 내부를 채운다는 의미
        
        # 4. 얼굴 영역 블러 처리 (사각형 전체를 일단 블러 처리)
        blur_level = (99, 99) # 블러 강도 설정 (홀수, 홀수)
        blurred_roi = cv2.GaussianBlur(face_roi, blur_level, 0)

        # 5. 마스크를 사용하여 이미지 조합 (배경과 전경 분리)
        
        # a) 원본 ROI에서 원형 외곽만 추출 (마스크 반전 사용)
        # 즉, 원 영역 바깥(검은색 마스크 부분)은 원본 픽셀 유지
        mask_inv = cv2.bitwise_not(mask)
        face_bg = cv2.bitwise_and(face_roi, face_roi, mask=mask_inv)
        
        # b) 블러된 ROI에서 원형 영역만 추출 (마스크 사용)
        # 즉, 원 영역 안쪽(흰색 마스크 부분)만 블러 픽셀 적용
        face_fg = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask)
        
        # c) 배경과 전경을 합치기
        final_roi = cv2.add(face_bg, face_fg)

        # 6. 원본 프레임에 최종 결과물을 다시 붙여넣기
        frame[y:y+h, x:x+w] = final_roi

    # 결과 프레임을 화면에 표시
    cv2.imshow('Real-time Circular Face Mosaic', frame)

    # 'q' 키를 누르면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용이 끝난 후 자원 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")