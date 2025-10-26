import cv2

url = "http://192.168.45.177:8080/video"  # 필요 시 /mjpeg로 변경
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # FFmpeg 사용

if not cap.isOpened():
    print("카메라 연결 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 가져오기 실패")
        break

    cv2.imshow("Smartphone Cam", frame)

    if cv2.waitKey(1) & 0xFF ==27:
        break

cap.release()
cv2.destroyAllWindows()