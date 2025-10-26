import cv2
import os
from deepface import DeepFace
import numpy as np
import time


FACE_DATABASE_DIR = "detected_face"
MODEL_NAME = "VGG-Face"          
DETECTOR_BACKEND = "mediapipe"   
DISTANCE_METRIC = "cosine"      
MOSAIC_FACTOR = 20             


def find_authorized_face_path(directory):
    """지정된 디렉토리에서 지원하는 확장자의 첫 번째 이미지 파일을 찾습니다."""
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            return os.path.join(directory, filename)
    return None


def apply_mosaic(image, face_area):
    """지정된 영역에 모자이크를 적용하는 함수"""
    x, y, w, h = face_area
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return image

    face_roi = image[y:y+h, x:x+w]
    
    if face_roi.size == 0:
        return image

    small_roi = cv2.resize(face_roi, (w // MOSAIC_FACTOR, h // MOSAIC_FACTOR), interpolation=cv2.INTER_NEAREST)
    mosaic_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
    
    image[y:y+h, x:x+w] = mosaic_roi
    return image


def run_mosaic_app():
    AUTHORIZED_FACE_PATH = find_authorized_face_path(FACE_DATABASE_DIR)
    
    if not AUTHORIZED_FACE_PATH:
        print(f"[ERROR] '{FACE_DATABASE_DIR}' 폴더에서 등록된 얼굴 이미지 파일을 찾을 수 없습니다.")
        print("[INFO] 지원하는 확장자: .jpg, .jpeg, .png, .bmp, .gif")
        print("[INFO] 먼저 'create_authorized_face.py'를 실행하여 얼굴을 등록해주세요.")
        return
    
    print(f"[INFO] 등록된 얼굴 파일을 사용합니다: {AUTHORIZED_FACE_PATH}")

    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print(f"[ERROR] Haar Cascade 파일을 찾을 수 없습니다: {haar_path}")
        return
    face_cascade = cv2.CascadeClassifier(haar_path)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다. 다른 앱에서 카메라를 사용 중인지 확인하세요.")
        return


    try:
        print("[INFO] 얼굴 인식 모델을 로드 중입니다...")
        DeepFace.build_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        cap.release()
        return

    print("----------------------------------------------------")
    print("[INFO] 얼굴 감지를 시작합니다...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 카메라에서 프레임을 읽지 못했습니다. 루프를 종료합니다.")
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        print(f"[DEBUG] 탐지된 얼굴 수: {len(faces)}")

        for i, (x, y, w, h) in enumerate(faces):
            current_face_img = frame[y:y+h, x:x+w]

            if current_face_img.size == 0:
                print(f"[DEBUG] 얼굴 {i}: 영역이 비어있어 건너뜁니다.")
                continue

            try:
                result = DeepFace.verify(
                    img1_path=current_face_img,
                    img2_path=AUTHORIZED_FACE_PATH,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )
                
                print(f"[DEBUG] 얼굴 {i}: 등록된 얼굴과 일치 여부: {result['verified']}")

                if not result['verified']:
                    frame = apply_mosaic(frame, (x, y, w, h))
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Authorized", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"[WARN] 얼굴 {i} 비교 중 오류: {e}. 해당 얼굴을 모자이크 처리합니다.")
                frame = apply_mosaic(frame, (x, y, w, h))

        cv2.imshow("Real-time Mosaic - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 프로그램을 종료합니다.")

if __name__ == '__main__':
    run_mosaic_app()
