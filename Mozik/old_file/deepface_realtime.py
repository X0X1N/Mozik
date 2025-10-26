import cv2
try:
    from deepface import DeepFace
except Exception as e:
    # 친절한 안내 출력 후 종료합니다.
    print("[ERROR] DeepFace 라이브러리 로드 중 오류가 발생했습니다.")
    print(f"[ERROR] 원인: {e}")
    print("[HINT] 보통 'tf-keras' 패키지가 필요하거나 TensorFlow/retinaface 버전 불일치 때문에 발생합니다.")
    print("[HINT] Windows PowerShell에서 다음을 실행해 보세요:")
    print("    python -m pip install --upgrade pip")
    print("    python -m pip install tf-keras")
    print("또는 TensorFlow 버전을 낮추려면: python -m pip install 'tensorflow<2.20'")
    # 추가 안내
    print("[INFO] 문제 해결 후 스크립트를 다시 실행하세요.")
    raise
import os
import time

# ----------------------------------------------------
# 1. 환경 설정
# ----------------------------------------------------
DB_PATH = "dataset"  # 얼굴 이미지 데이터베이스 경로
MODEL_NAME = "VGG-Face"  # 사용할 얼굴 인식 모델
DETECTOR_BACKEND = "mediapipe"  # 얼굴 탐지 백엔드
DISTANCE_METRIC = "cosine"  # 거리 측정 방식

# 데이터베이스 경로가 없으면 생성
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
    print(f"[INFO] '{DB_PATH}' 폴더가 없어 새로 생성했습니다.")
    print(f"[INFO] 이 폴더에 'user.0.xx.jpg'와 같은 형식으로 인식할 얼굴 이미지를 저장해주세요.")

# ----------------------------------------------------
# 2. 실시간 얼굴 인식 함수
# ----------------------------------------------------
def realtime_face_recognition():
    print("[INFO] 실시간 얼굴 인식을 시작합니다. 종료하려면 'q'를 누르세요.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    # 인식 성능을 위해 프레임 처리 간격 설정
    prev_time = 0
    frame_rate_delay = 0.5  # 0.5초에 한 번씩 얼굴 분석

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if (current_time - prev_time) > frame_rate_delay:
            prev_time = current_time
            
            try:
                # DeepFace.find를 사용하여 얼굴 인식
                # enforce_detection=False로 설정하여 얼굴이 없어도 오류를 발생시키지 않음
                dfs = DeepFace.find(
                    img_path=frame,
                    db_path=DB_PATH,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False,
                    silent=True  # 상세 로그 비활성화
                )

                # dfs 결과가 비어있지 않고, 데이터프레임에 결과가 있을 경우
                if dfs and not dfs[0].empty:
                    df = dfs[0]
                    # 가장 유사도가 높은 얼굴 정보 가져오기
                    best_match = df.iloc[0]
                    
                    # 얼굴 위치 좌표
                    x, y, w, h = best_match['source_x'], best_match['source_y'], best_match['source_w'], best_match['source_h']
                    
                    # 사각형 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 신원 확인 (파일 경로에서 이름 추출)
                    identity = best_match['identity']
                    # 예: 'dataset/user.0.1.jpg' -> 'user.0'
                    label = os.path.basename(identity).split('.')[0] + "." + os.path.basename(identity).split('.')[1]

                    # 텍스트 표시
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                # find 함수에서 얼굴을 찾지 못하거나 라이브러리 내부에서 예외가 발생할 수 있음
                # 개발 중에는 예외를 출력해 원인을 파악할 수 있도록 최소한의 로그를 남깁니다.
                print(f"[DEBUG] DeepFace.find 예외: {e}")
                # traceback 정보가 필요하면 아래 주석을 해제하세요.
                # import traceback
                # traceback.print_exc()
                pass

        # 화면에 영상 출력
        cv2.imshow("Realtime Face Recognition (DeepFace)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 프로그램을 종료합니다.")

# ----------------------------------------------------
# 3. 실행
# ----------------------------------------------------
if __name__ == '__main__':
    # DeepFace가 처음 실행될 때 모델 파일을 다운로드하므로 시간이 걸릴 수 있습니다.
    # 미리 모델을 로드하여 초기 딜레이를 줄일 수 있습니다.
    try:
        print("[INFO] 얼굴 인식 모델을 로드 중입니다. 잠시만 기다려주세요...")
        DeepFace.build_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")
    except Exception as e:
        print(f"[ERROR] 모델 로드 중 오류 발생: {e}")
        print("[INFO] 필요한 경우 'pip install --upgrade deepface'를 실행하여 라이브러리를 업데이트해보세요.")
        exit()

    realtime_face_recognition()
