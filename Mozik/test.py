import cv2

# 🚨 휴대폰 앱에서 확인한 실제 스트림 URL로 변경해야 합니다.
# (예: "http://192.168.0.15:8080/video")
STREAM_URL = "http://192.168.120.242:8080/video" 

def run_stream_client():
    # 웹캠 인덱스 대신 스트림 URL을 사용하여 영상을 캡처합니다.
    cap = cv2.VideoCapture(STREAM_URL)
    
    if not cap.isOpened():
        print(f"[ERROR] 스트림을 열 수 없습니다. 휴대폰 앱 실행 및 Wi-Fi 연결을 확인하세요.")
        return

    print("[INFO] 휴대폰 카메라 스트림 켜짐. 'q'를 눌러 종료.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[WARNING] 프레임을 읽을 수 없습니다.")
            break
        
        # 여기서 얼굴 탐지 로직이 작동됩니다.
        cv2.imshow("Phone Camera Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_stream_client()