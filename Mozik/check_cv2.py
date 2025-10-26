try:
    import cv2
    print("cv2 imported, version", cv2.__version__)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("repr(e):", repr(e))
