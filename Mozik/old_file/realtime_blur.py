import cv2
import numpy as np
import os

cascade_path = r"C:\Users\User\AppData\Local\Programs\Python\Python313\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

# ----------------------------------------------------
# 1. 환경 설정 및 모델 로드
# ----------------------------------------------------
USER_ID = 0 
RECOGNIZER_FILE = 'user_face_model.yml'
CONFIDENCE_THRESHOLD = 80 # 이 값보다 낮아야 '사용자'로 판단 (0에 가까울수록 확실)

# 얼굴 탐지 모델 로드
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError("얼굴 탐지 모델 로드 실패.")
except Exception as e:
    print(f"오류: {e}")
    exit()

# 얼굴 인식 모델 로드 (학습된 모델이 없으면 블러 기능만 작동)
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_loaded = False
if os.path.exists(RECOGNIZER_FILE):
    try:
        recognizer.read(RECOGNIZER_FILE)
        model_loaded = True
        print("✅ 얼굴 인식 모델 로드 완료. 사용자 식별이 활성화되었습니다.")
    except Exception as e:
        print(f"⚠️ 경고: 모델 파일 로드 중 오류 발생: {e}")
        
if not model_loaded:
    print("⚠️ 경고: 학습된 모델 파일이 없어 모든 얼굴에 블러가 적용됩니다. 학습을 먼저 진행하세요.")

# 웹캠 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 성공적으로 연결되었습니다. 'q' 키를 누르면 종료됩니다.")

# ----------------------------------------------------
# 2. 실시간 탐지 및 인식 루프
# ----------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        is_user = False
        
        # ----------------------------------------------------
        # 인식 로직 (모델이 로드된 경우에만 실행)
        # ----------------------------------------------------
        if model_loaded:
            # LBPH는 고정 크기 입력 선호
            face_resized = cv2.resize(face_gray, (100, 100)) 
            
            # 예측: ID와 신뢰도(confidence)를 얻습니다.
            predicted_id, confidence = recognizer.predict(face_resized)
            
            # 사용자 판별: ID 일치 및 신뢰도 기준 통과
            if predicted_id == USER_ID and confidence < CONFIDENCE_THRESHOLD:
                is_user = True
            
            # 화면에 인식 정보 표시 (디버깅용)
            label_text = f"ID: {predicted_id} (Conf: {confidence:.1f})"
            cv2.putText(frame, label_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        if is_user:
            # ✅ 사용자 얼굴: 블러 처리 건너뛰기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "ACCESS GRANTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            continue 
        
        # ----------------------------------------------------
        # ❌ 외부인 얼굴: 둥근 블러 로직 실행
        # ----------------------------------------------------
        
        cv2.putText(frame, "BLUR APPLIED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 1. 얼굴 영역(ROI) 추출
        face_roi = frame[y:y+h, x:x+w]
        
        # 2. 둥근 마스크 생성
        mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
        (cx, cy) = (w // 2, h // 2)
        radius = min(w, h) // 2
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # 3. 얼굴 영역 블러 처리
        blur_level = (99, 99)
        blurred_roi = cv2.GaussianBlur(face_roi, blur_level, 0)

        # 4. 마스크를 사용하여 이미지 조합 및 원본 프레임에 적용
        mask_inv = cv2.bitwise_not(mask)
        face_bg = cv2.bitwise_and(face_roi, face_roi, mask=mask_inv)
        face_fg = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask)
        final_roi = cv2.add(face_bg, face_fg)
        frame[y:y+h, x:x+w] = final_roi

    cv2.imshow('Real-time Circular Face Mosaic (User Exclusion)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")