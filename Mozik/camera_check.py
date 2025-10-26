import cv2
import time

backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
backend_names = ['CAP_DSHOW', 'CAP_MSMF', 'CAP_ANY']

for b, name in zip(backends, backend_names):
    print('--- Testing', name, '---')
    try:
        cap = cv2.VideoCapture(0, b)
        if not cap.isOpened():
            print('open failed')
            continue
        success_count = 0
        frame_shape = None
        start = time.time()
        # try to read up to 30 frames or 5 seconds
        while time.time() - start < 5 and success_count < 5:
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
                frame_shape = frame.shape
                print('read ok, shape=', frame_shape)
            else:
                print('read failed')
            time.sleep(0.1)
        cap.release()
        print('summary:', 'success_count=', success_count, 'frame_shape=', frame_shape)
    except Exception as e:
        print('exception:', e)

print('camera check done')
