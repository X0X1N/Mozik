import numpy as np #numpy를 사용하기 위해 import함.
import cv2 #opencv를 사용하기 위해 import함. 
cap = cv2.VideoCapture(0) #비디오 캡쳐 객체 생성. 숫자는 어떤 카메라를 사용하는지 

cap.set(3,640)
cap.set(4,480)

while(True):
    ret,frame=cap.read()
    if not ret:
        print("카메라에서 프레임을 가져오지 못했습니다.")
        break
    cv2.imshow('frame', frame)

    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()