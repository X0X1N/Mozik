import cv2
import os
import pkgutil



SAVE_DIR = "detected_face"  
SAVE_PATH = os.path.join(SAVE_DIR, "authorized_face.jpg")
DETECTOR_BACKEND = "haarcascade" 


if not os.path.exists(SAVE_DIR): 
    os.makedirs(SAVE_DIR) 


def capture_and_save_face(): 
    print("[INFO] 얼굴 등록을 시작합니다. 카메라를 응시하고 's' 키를 누르면 얼굴이 저장됩니다.")
    print("[INFO] 종료하려면 'q'를 누르세요.")


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

   
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
            break
        
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        
        if len(faces) > 0:
            detected_face_coords = faces[0]
            (x, y, w, h) = detected_face_coords
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            detected_face_coords = None

        cv2.imshow("Authorize Face - Press 's' to save, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            
            if detected_face_coords is not None:
                (x, y, w, h) = detected_face_coords
                
               
                pad_w = int(w * 0.25)
                pad_h = int(h * 0.25)
                
              
                img_h, img_w, _ = frame.shape
                new_x = max(0, x - pad_w)
                new_y = max(0, y - pad_h)
                new_w = min(img_w - new_x, w + 2 * pad_w)
                new_h = min(img_h - new_y, h + 2 * pad_h)
                
                padded_face_img = frame[new_y:new_y+new_h, new_x:new_x+new_w]
                
                cv2.imwrite(SAVE_PATH, padded_face_img)
                print(f"[SUCCESS] 얼굴이 '{SAVE_PATH}'에 성공적으로 저장되었습니다.")
                break
            else:
                print("[WARNING] 저장할 얼굴이 탐지되지 않았습니다. 다시 시도해주세요.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 얼굴 등록 프로그램을 종료합니다.")

if __name__ == '__main__':
    capture_and_save_face()
