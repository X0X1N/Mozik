import cv2
import os

# ============================================================
# 1. 환경 설정
# ============================================================
# ⚠️ 반드시 휴대폰의 실제 스트림 주소로 변경하세요.
# 예: "http://192.168.0.15:8080/video"
STREAM_URL = "http://192.168.120.242:8080/video"

SAVE_DIR = "detected_face"
SAVE_PATH = os.path.join(SAVE_DIR, "authorized_face.jpg")
DETECTOR_BACKEND = "haarcascade"  # 얼굴 탐지 백엔드 (OpenCV HaarCascade 사용)

# 저장 디렉터리가 없으면 자동 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ============================================================
# 2. 얼굴 캡처 및 저장 함수
# ============================================================
def capture_and_save_face_from_stream():
    print("[INFO] 휴대폰 카메라 스트림으로 얼굴 등록을 시작합니다.")
    print("[INFO] 카메라를 응시하고 's' 키를 누르면 얼굴이 저장됩니다.")
    print("[INFO] 종료하려면 'q'를 누르세요.")

    # 스트림 URL을 사용해서 영상 입력
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print(f"[ERROR] 스트림을 열 수 없습니다. URL 또는 Wi-Fi 연결을 확인하세요.")
        return

    # Haar Cascade 분류기 로드
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print(f"[ERROR] haarcascade 파일을 찾을 수 없습니다: {haar_path}")
        cap.release()
        return

    face_cascade = cv2.CascadeClassifier(haar_path)
    detected_face_coords = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임을 읽지 못했습니다. 연결을 확인하세요.")
            break

        # 거울 모드로 반전
        frame = cv2.flip(frame, 1)

        # 얼굴 탐지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # 얼굴 탐지 시 표시
        if len(faces) > 0:
            detected_face_coords = faces[0]
            (x, y, w, h) = detected_face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Press 's' to save",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            detected_face_coords = None

        # 실시간 영상 출력
        cv2.imshow("Phone Stream - Press 's' to save, 'q' to quit", frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # 얼굴이 탐지된 경우 저장
            if detected_face_coords is not None:
                (x, y, w, h) = detected_face_coords

                # 여백 25% 추가
                pad_w = int(w * 0.25)
                pad_h = int(h * 0.25)

                img_h, img_w, _ = frame.shape
                new_x = max(0, x - pad_w)
                new_y = max(0, y - pad_h)
                new_w = min(img_w - new_x, w + 2 * pad_w)
                new_h = min(img_h - new_y, h + 2 * pad_h)

                # 여백이 포함된 얼굴 잘라내기
                padded_face_img = frame[new_y : new_y + new_h, new_x : new_x + new_w]

                # 이미지 저장
                cv2.imwrite(SAVE_PATH, padded_face_img)
                print(f"[SUCCESS] 얼굴이 '{SAVE_PATH}'에 성공적으로 저장되었습니다.")
                break
            else:
                print("[WARNING] 얼굴이 탐지되지 않았습니다. 다시 시도해주세요.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 얼굴 등록 프로그램을 종료합니다.")


# ============================================================
# 3. 실행
# ============================================================
if __name__ == "__main__":
    capture_and_save_face_from_stream()
