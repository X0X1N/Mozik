import cv2
import numpy as np
import os
from PIL import Image

# ----------------------------------------------------
# 1. 환경 설정
# ----------------------------------------------------
USER_ID = 0  # 이 ID를 학습된 사용자로 인식합니다.
DATASET_PATH = 'dataset'
RECOGNIZER_FILE = 'user_face_model.yml'

# 기존 디렉토리가 없으면 생성
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# 얼굴 탐지 모델 로드 (데이터 수집에 사용)
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError("얼굴 탐지 모델 로드 실패.")
except Exception as e:
    print(f"오류: {e}")
    exit()

# ----------------------------------------------------
# 2. 학습 데이터셋 준비 함수
# ----------------------------------------------------
def get_images_and_labels(path):
    # 'dataset' 폴더 내 모든 파일 경로를 가져옵니다.
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        try:
            # 흑백 이미지로 로드
            pil_image = Image.open(image_path).convert('L') 
            img_numpy = np.array(pil_image, 'uint8')

            # 파일 이름에서 ID를 추출 (예: 'user.0.30.jpg'에서 '0' 추출)
            image_id = int(os.path.split(image_path)[1].split('.')[1])
            
            face_samples.append(img_numpy)
            ids.append(image_id)
        except Exception as e:
            print(f"이미지 로드 중 오류 발생: {e} ({image_path})")
            continue
            
    return face_samples, ids

# ----------------------------------------------------
# 3. 사용자 얼굴 데이터 수집 (캡쳐)
# ----------------------------------------------------
def capture_user_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
        return

    print("\n[INFO] 캡쳐를 시작합니다. 카메라 앞에서 얼굴을 다양한 각도와 표정으로 보여주세요.")
    
    count = 0
    while count < 30: # 30장의 이미지를 목표로 합니다.
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 캡쳐 간격 조절 (너무 빠르게 저장되는 것을 방지)
            if count % 2 == 0:
                img_name = f"{DATASET_PATH}/user.{USER_ID}.{count}.jpg"
                # 탐지된 얼굴 영역만 저장
                cv2.imwrite(img_name, gray[y:y+h, x:x+w])
                print(f"이미지 저장 완료: {img_name}")
                count += 1
            
            # 저장된 이미지 수를 화면에 표시
            cv2.putText(frame, f"Captured: {count}/30", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 데이터 수집을 완료했습니다.")

# ----------------------------------------------------
# 4. 모델 학습 및 저장
# ----------------------------------------------------
def train_recognizer():
    # 데이터 수집 후 데이터셋 폴더에 이미지가 있어야 합니다.
    print("\n[INFO] 학습 데이터를 분석 중입니다...")
    faces, ids = get_images_and_labels(DATASET_PATH)

    if not faces:
        print("[ERROR] 학습할 얼굴 이미지를 찾을 수 없습니다. 'dataset' 폴더를 확인해주세요.")
        return

    print(f"[INFO] {len(np.unique(ids))}명의 사용자에 대해 {len(faces)}개의 샘플로 학습을 시작합니다.")

    # LBPH Face Recognizer 생성 및 학습
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    # 학습된 모델을 파일로 저장
    recognizer.write(RECOGNIZER_FILE)
    print(f"\n[SUCCESS] 학습이 완료되었습니다. 모델이 '{RECOGNIZER_FILE}'에 저장되었습니다.")

# ----------------------------------------------------
# 5. 실행
# ----------------------------------------------------
if __name__ == '__main__':
    # 1. 데이터 수집
    capture_user_faces()
    
    # 2. 모델 학습
    train_recognizer()