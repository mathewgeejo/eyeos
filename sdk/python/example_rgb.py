"""Optional OpenCV capture adapter: python example_rgb.py.
For existing software, supply its RGB frames through the same process_rgb call.
"""
import cv2
from eye_tracker import Tracker

with Tracker() as tracker:
    camera = cv2.VideoCapture(0)
    try:
        while camera.isOpened():
            timestamp = tracker.timestamp_ms()
            ok, bgr = camera.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            event = tracker.process_rgb(rgb.tobytes(), rgb.shape[1], rgb.shape[0], timestamp)
            print(event["estimate"])
            # A graphical host draws calibration_progress()['target'] before enabling control.
            cv2.imshow("Local camera input (no OS mouse input)", bgr)
            if cv2.waitKey(1) == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
