import cv2
import time
from threading import Thread, Lock

class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

cap = WebcamVideoStream().start()
time.sleep(1.0)

for i in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"Frame {i}: shape={frame.shape}")
    else:
        print(f"Frame {i}: FAIL (ret={ret}, frame={frame})")
    time.sleep(0.5)

cap.stopped = True