import cv2
import os
from deepface import DeepFace
import numpy as np
import time


# ============================================================
# 1. 환경 설정
# ============================================================
# ⚠️ 휴대폰 IP Webcam 앱에서 보여주는 실제 스트림 주소로 바꿔주세요.
# 예: "http://192.168.0.15:8080/video"
STREAM_URL = "http://192.168.120.242:8080/video"

FACE_DATABASE_DIR = "detected_face"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "mediapipe"
DISTANCE_METRIC = "cosine"
MOSAIC_FACTOR = 20


# ============================================================
# 2. 등록된 얼굴 찾기
# ============================================================
def find_authorized_face_path(directory):
    """등록된 얼굴 이미지 파일 경로 찾기"""
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            return os.path.join(directory, filename)
    return None


# ============================================================
# 3. 모자이크 함수
# ============================================================
def apply_mosaic(image, face_area):
    """얼굴 영역에 모자이크 적용"""
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


# ============================================================
# 4. 메인 함수
# ============================================================
def run_mosaic_app():
    AUTHORIZED_FACE_PATH = find_authorized_face_path(FACE_DATABASE_DIR)
    
    if not AUTHORIZED_FACE_PATH:
        print(f"[ERROR] '{FACE_DATABASE_DIR}' 폴더에서 등록된 얼굴 이미지 파일을 찾을 수 없습니다.")
        print("[INFO] 먼저 'create_authorized_face.py'를 실행하여 얼굴을 등록하세요.")
        return

    print(f"[INFO] 등록된 얼굴 파일을 사용합니다: {AUTHORIZED_FACE_PATH}")

    # HaarCascade 로드
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print(f"[ERROR] Haar Cascade 파일을 찾을 수 없습니다: {haar_path}")
        return
    face_cascade = cv2.CascadeClassifier(haar_path)

    # ========================================================
    # 🚀 휴대폰 카메라 스트림으로부터 영상 받기
    # ========================================================
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"[ERROR] 스트림을 열 수 없습니다. 휴대폰 앱 실행 및 Wi-Fi 연결을 확인하세요.")
        return

    # DeepFace 모델 로드
    try:
        print("[INFO] 얼굴 인식 모델 로드 중...")
        DeepFace.build_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        cap.release()
        return

    print("----------------------------------------------------")
    print("[INFO] 휴대폰 카메라 스트림 연결 완료. 얼굴 인식 시작...")

    # ========================================================
    # 🔄 실시간 처리 루프
    # ========================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 스트림에서 프레임을 읽지 못했습니다.")
            break

        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)

        # 얼굴 탐지 (회색조로 변환 후 탐지)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        print(f"[DEBUG] 탐지된 얼굴 수: {len(faces)}")

        # 탐지된 얼굴 각각 처리
        for i, (x, y, w, h) in enumerate(faces):
            current_face_img = frame[y:y+h, x:x+w]

            if current_face_img.size == 0:
                print(f"[DEBUG] 얼굴 {i}: 영역이 비어있음, 건너뜀.")
                continue

            try:
                # 등록 얼굴과 비교
                result = DeepFace.verify(
                    img1_path=current_face_img,
                    img2_path=AUTHORIZED_FACE_PATH,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )
                print(f"[DEBUG] 얼굴 {i}: 일치 여부: {result['verified']}")

                if not result['verified']:
                    # 일치하지 않으면 모자이크
                    frame = apply_mosaic(frame, (x, y, w, h))
                else:
                    # 일치하면 초록 박스 표시
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Authorized", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"[WARN] 얼굴 {i} 비교 중 오류: {e}")
                frame = apply_mosaic(frame, (x, y, w, h))

        # 결과 출력
        cv2.imshow("📱 Phone Stream - Press 'q' to quit", frame)

        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 프로그램을 종료합니다.")


# ============================================================
# 5. 실행
# ============================================================
if __name__ == "__main__":
    run_mosaic_app()
