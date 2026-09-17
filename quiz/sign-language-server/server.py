"""
Sign Language Quiz — FastAPI Server (Dual Mode: Local + Remote)

Modes:
  LOCAL  (CAMERA_MODE=local):
    - Server mở webcam, chạy inference, stream MJPEG (giống bản gốc)
    - Dùng cho development/demo trên máy local

  REMOTE (CAMERA_MODE=remote):
    - Client gửi JPEG frames qua WebSocket binary
    - Server nhận frame → MediaPipe + STA-GCN → trả kết quả JSON
    - Dùng cho deployment trên VPS/cloud có GPU

Kiến trúc Remote:
  Browser: getUserMedia → canvas → JPEG → WS binary → Server
  Server:  decode JPEG → Holistic + STA-GCN → JSON result → WS → Browser
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import time
import json
import asyncio
from threading import Thread, Lock
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from models.sta_gcn import Model


# ==========================================
# PATH SETUP
# ==========================================
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.dirname(SERVER_DIR)

print(f"📁 Server dir: {SERVER_DIR}")
print(f"📁 Static dir: {STATIC_DIR}")

for fname in ["main.html", "quiz.css", "quiz.js"]:
    fpath = os.path.join(STATIC_DIR, fname)
    print(f"   {'✅' if os.path.isfile(fpath) else '❌'} {fpath}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# CẤU HÌNH
# ==========================================
# --- Mode ---
CAMERA_MODE = os.environ.get("CAMERA_MODE", "local").lower()  # "local" hoặc "remote"
print(f"🔧 CAMERA_MODE = {CAMERA_MODE}")

NUM_JOINTS = 27
SEQUENCE_LENGTH = 60
NUM_CLASSES = 50
WEIGHT_PATH = os.path.join(SERVER_DIR, 'stagcn_tiny_supcon_50cls_best.pt')
PREDICTION_THRESHOLD = 0.40

# Camera capture — chỉ dùng cho LOCAL mode
CAMERA_SRC = int(os.environ.get("CAMERA_SRC", "1"))
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Holistic inference
HOLISTIC_COMPLEXITY = 0
HOLISTIC_INPUT_WIDTH = 640

# Coordinate space cho STA-GCN (phải match training data)
MODEL_INPUT_WIDTH = 1280
MODEL_INPUT_HEIGHT = 720

# Display stream — chỉ dùng cho LOCAL mode
STREAM_WIDTH = None
JPEG_QUALITY = 70
JPEG_CHROMA_QUALITY = 70
MJPEG_TARGET_FPS = 30

MIRROR_FRAME = True

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

MP_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]
MP_HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

MODEL_DICT = {
    0: "an", 1: "ban", 2: "banchan", 3: "begai", 4: "beo",
    5: "betrai", 6: "bo", 7: "cao", 8: "chaidau", 9: "chay",
    10: "co", 11: "cuoi", 12: "danhrang", 13: "dau", 14: "di",
    15: "divesinh", 16: "doidep", 17: "dung", 18: "gangtay", 19: "gay",
    20: "nothing",
    21: "baonhieu", 22: "bittat", 23: "captoc", 24: "chao", 25: "emtrai",
    26: "goidau", 27: "khanmat", 28: "khoc", 29: "khoemanh", 30: "kinh",
    31: "luoc", 32: "ma", 33: "macquanao", 34: "mat", 35: "metmoi",
    36: "mieng", 37: "mui", 38: "muluoitrai", 39: "nam", 40: "ngoi",
    41: "ngu", 42: "nhay", 43: "niemvui", 44: "non", 45: "ruachan",
    46: "ruamat", 47: "ruatay", 48: "sach", 49: "suckhoe"
}

ACCENT_MAP = {
    "ĂN": "an", "BẠN": "ban", "BÀN CHÂN": "banchan", "BÉ GÁI": "begai", "BÉO": "beo",
    "BÉ TRAI": "betrai", "BỐ": "bo", "CAO": "cao", "CHẢI ĐẦU": "chaidau", "CHẠY": "chay",
    "CỔ": "co", "CƯỜI": "cuoi", "ĐÁNH RĂNG": "danhrang", "ĐẦU": "dau", "ĐI": "di",
    "ĐI VỆ SINH": "divesinh", "ĐÔI DÉP": "doidep", "ĐỨNG": "dung", "GĂNG TAY": "gangtay",
    "GẦY": "gay", "NOTHING": "nothing",
    "BAO NHIÊU": "baonhieu", "BÍT TẤT": "bittat", "CẶP TÓC": "captoc", "CHÀO": "chao",
    "EM TRAI": "emtrai", "GỘI ĐẦU": "goidau", "KHĂN MẶT": "khanmat", "KHÓC": "khoc",
    "KHỎE MẠNH": "khoemanh", "KÍNH": "kinh", "LƯỢC": "luoc", "MÁ": "ma",
    "MẶC QUẦN ÁO": "macquanao", "MẮT": "mat", "MỆT MỎI": "metmoi", "MIỆNG": "mieng",
    "MŨI": "mui", "MŨ LƯỠI TRAI": "muluoitrai", "NẰM": "nam", "NGỒI": "ngoi",
    "NGỦ": "ngu", "NHẢY": "nhay", "NIỀM VUI": "niemvui", "NÓN": "non",
    "RỬA CHÂN": "ruachan", "RỬA MẶT": "ruamat", "RỬA TAY": "ruatay",
    "SÁCH": "sach", "SỨC KHỎE": "suckhoe"
}
NO_ACCENT_TO_ACCENT = {v: k for k, v in ACCENT_MAP.items()}


# ==========================================
# ONE-EURO FILTER — smoothing landmarks để giảm jitter
# ==========================================
# Reference: Casiez, G., Roussel, N., & Vogel, D. (2012).
#   "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input
#    in Interactive Systems." CHI 2012.
#
# Ý tưởng: low-pass filter với cutoff frequency ADAPTIVE theo tốc độ.
#   - Tay đứng yên (speed thấp) → cutoff thấp → smooth mạnh, hết jitter
#   - Tay di chuyển nhanh (speed cao) → cutoff cao → gần như no-op,
#     không tạo lag perceivable.
#
# Tuning constants:
#   - min_cutoff: cutoff baseline khi tay đứng yên. Thấp = smooth hơn nhưng lag.
#                 Range thường: 0.5–2.0 Hz.
#   - beta: tốc độ tăng cutoff theo speed. Cao = phản ứng nhanh hơn.
#                 Range thường: 0.0001–0.01.
#   - d_cutoff: cutoff cho velocity estimator. Thường giữ = 1.0.

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def _exp_smooth(a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t):
        """x: np.ndarray shape (N, D); t: timestamp (sec)"""
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = max(t - self.t_prev, 1e-6)

        # Velocity (đạo hàm theo thời gian)
        dx = (x - self.x_prev) / t_e
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._exp_smooth(a_d, dx, self.dx_prev)

        # Adaptive cutoff theo |dx_hat|
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exp_smooth(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


# ==========================================
# HELPERS
# ==========================================
def _smooth_landmark_list(lm_list, filt, t):
    """
    Smooth toàn bộ landmark proto (33 pose hoặc 21 hand) bằng One-Euro filter.
    In-place modify .x, .y của từng landmark.

    lm_list: NormalizedLandmarkList từ MediaPipe (hoặc None)
    filt:    OneEuroFilter instance
    t:       timestamp hiện tại

    Returns: modified lm_list (hoặc None nếu input None)
    """
    if lm_list is None:
        filt.reset()
        return None

    n = len(lm_list.landmark)
    raw = np.empty((n, 2), dtype=np.float32)
    for i, lm in enumerate(lm_list.landmark):
        raw[i, 0] = lm.x
        raw[i, 1] = lm.y

    smoothed = filt.filter(raw, t)

    for i, lm in enumerate(lm_list.landmark):
        lm.x = float(smoothed[i, 0])
        lm.y = float(smoothed[i, 1])

    return lm_list


def extract_frame_keypoints(results, w, h):
    pts = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    if results.pose_landmarks:
        for i, idx in enumerate(MP_POSE_INDICES):
            lm = results.pose_landmarks.landmark[idx]
            pts[i] = [lm.x * w, lm.y * h, lm.visibility]
    if results.left_hand_landmarks:
        for i, idx in enumerate(MP_HAND_INDICES):
            lm = results.left_hand_landmarks.landmark[idx]
            pts[7 + i] = [lm.x * w, lm.y * h, 1.0]
    if results.right_hand_landmarks:
        for i, idx in enumerate(MP_HAND_INDICES):
            lm = results.right_hand_landmarks.landmark[idx]
            pts[17 + i] = [lm.x * w, lm.y * h, 1.0]
    return pts


def preprocess_for_model(frames_list):
    data = np.array(frames_list)
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=-1)
    data[0] -= data[0, :, 0, 0].mean(axis=0)
    data[1] -= data[1, :, 0, 0].mean(axis=0)
    return torch.from_numpy(data).unsqueeze(0).float()


# ==========================================
# PRACTICE STATE — Per-session cho remote mode
# ==========================================
class PracticeState:
    def __init__(self):
        self.active = False
        self.state = "IDLE"
        self.target_no_accent = ""
        self.target_display = ""
        self.phase_start = 0.0
        self.frames_queue = deque(maxlen=SEQUENCE_LENGTH)
        self.last_top3 = []
        self.last_success = False
        self.event_queue = []
        self.lock = Lock()

        # Per-session filters (remote mode — mỗi user có filter riêng)
        self.pose_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        self.lhand_filter = OneEuroFilter(min_cutoff=1.5, beta=0.01)
        self.rhand_filter = OneEuroFilter(min_cutoff=1.5, beta=0.01)

        # Per-session holistic (remote mode)
        self.holistic = None

    def init_holistic(self):
        """Lazy-init Holistic instance cho session này."""
        if self.holistic is None:
            self.holistic = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=HOLISTIC_COMPLEXITY
            )
        return self.holistic

    def close_holistic(self):
        """Giải phóng Holistic khi session kết thúc."""
        if self.holistic is not None:
            self.holistic.close()
            self.holistic = None

    def push_event(self, evt):
        with self.lock:
            self.event_queue.append(evt)

    def pop_events(self):
        with self.lock:
            evts = self.event_queue
            self.event_queue = []
            return evts


def step_state_machine(ps, kpts):
    """Chạy state machine cho một PracticeState instance."""
    if not ps.active:
        return

    state = ps.state

    if state == "WAIT":
        elapsed = time.time() - ps.phase_start
        remaining = max(0.0, 2.0 - elapsed)
        ps.push_event({
            "type": "status",
            "state": "WAIT",
            "countdown": round(remaining, 2),
            "message": f"Chuẩn bị... {int(remaining) + 1}" if remaining > 0.05 else "BẮT ĐẦU!"
        })

        if elapsed >= 2.0:
            ps.state = "COLLECT"
            ps.frames_queue.clear()
            ps.push_event({"type": "status", "state": "COLLECT",
                          "progress": 0.0,
                          "message": "Hãy thực hiện ký hiệu!"})

    elif state == "COLLECT":
        ps.frames_queue.append(kpts)
        progress = len(ps.frames_queue) / SEQUENCE_LENGTH

        if len(ps.frames_queue) % 5 == 0 or len(ps.frames_queue) == SEQUENCE_LENGTH:
            ps.push_event({"type": "status", "state": "COLLECT",
                          "progress": round(progress, 3),
                          "message": f"Thu thập: {len(ps.frames_queue)}/{SEQUENCE_LENGTH}"})

        if len(ps.frames_queue) == SEQUENCE_LENGTH:
            ps.state = "PREDICT"
            _run_predict(ps)

    elif state == "SHOW":
        elapsed = time.time() - ps.phase_start
        if elapsed >= 3.0:
            ps.state = "WAIT"
            ps.phase_start = time.time()
            ps.frames_queue.clear()
            ps.push_event({"type": "status", "state": "WAIT", "message": "Lượt mới..."})


def _run_predict(ps):
    try:
        inp = preprocess_for_model(ps.frames_queue).to(device)
        with torch.no_grad():
            out, _ = model(inp)
            prob = F.softmax(out, dim=1).cpu().numpy()[0]
        idx3 = np.argsort(prob)[-3:][::-1]
        top3 = [{
            "label": MODEL_DICT[i].upper(),
            "labelVn": NO_ACCENT_TO_ACCENT.get(MODEL_DICT[i].upper(), MODEL_DICT[i].upper()),
            "score": round(float(prob[i]) * 100, 1)
        } for i in idx3]

        ps.last_top3 = top3
        is_correct = (top3[0]["label"] == ps.target_no_accent
                      and top3[0]["score"] >= PREDICTION_THRESHOLD * 100)
        ps.last_success = is_correct

        ps.push_event({
            "type": "result",
            "success": is_correct,
            "top3": top3,
            "message": ("CHÍNH XÁC!" if is_correct else f"AI nhận: {top3[0]['labelVn']}")
        })

        if is_correct:
            ps.state = "IDLE"
            ps.active = False
        else:
            ps.state = "SHOW"
            ps.phase_start = time.time()

    except Exception as e:
        print(f"[Predict error] {e}")
        ps.state = "SHOW"
        ps.phase_start = time.time()


def process_frame_remote(ps, jpeg_bytes):
    """
    Remote mode: decode JPEG → Holistic → extract keypoints → state machine.
    Trả về True nếu xử lý thành công.
    """
    # Decode JPEG
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return False

    h, w = frame.shape[:2]

    # Resize cho Holistic (nhẹ hơn)
    scale = HOLISTIC_INPUT_WIDTH / w
    small = cv2.resize(frame, (HOLISTIC_INPUT_WIDTH, int(h * scale)))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb_small.flags.writeable = False

    # Chạy Holistic
    holistic_inst = ps.init_holistic()
    results = holistic_inst.process(rgb_small)

    # One-Euro smoothing
    t_now = time.time()
    _smooth_landmark_list(results.pose_landmarks, ps.pose_filter, t_now)
    _smooth_landmark_list(results.left_hand_landmarks, ps.lhand_filter, t_now)
    _smooth_landmark_list(results.right_hand_landmarks, ps.rhand_filter, t_now)

    # Extract keypoints
    kpts = extract_frame_keypoints(results, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

    # State machine
    step_state_machine(ps, kpts)

    return True


# ==========================================
# LOAD MODEL
# ==========================================
print("⚙️  Loading STA-GCN model...")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"   → CUDA available: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"   → Apple Silicon MPS available")
else:
    device = torch.device('cpu')
    print(f"   → Falling back to CPU")

model = Model(
    num_class=NUM_CLASSES, num_point=NUM_JOINTS, num_person=1,
    graph='graph_stagcn.sign_27.Graph', graph_args={}
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device).eval()

print("   → Warming up model...")
with torch.no_grad():
    dummy = torch.randn(1, 3, SEQUENCE_LENGTH, NUM_JOINTS, 1).to(device)
    for _ in range(3):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
print("✅ Model loaded + warmed up.")


# ==========================================
# LOCAL MODE: Camera + InferenceWorker + MJPEG
# ==========================================
# Global filters cho LOCAL mode
pose_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
lhand_filter = OneEuroFilter(min_cutoff=1.5, beta=0.01)
rhand_filter = OneEuroFilter(min_cutoff=1.5, beta=0.01)

# Global practice state cho LOCAL mode (chỉ 1 user)
practice_state = PracticeState()

camera = None
holistic = None
inference_worker = None

if CAMERA_MODE == "local":
    class WebcamVideoStream:
        def __init__(self, src=CAMERA_SRC, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, fps)
            self.grabbed, self.frame = self.stream.read()
            self.stopped = False
            self.lock = Lock()

        def start(self):
            Thread(target=self._update, daemon=True).start()
            return self

        def _update(self):
            while not self.stopped:
                grabbed, frame = self.stream.read()
                if grabbed and frame is not None:
                    with self.lock:
                        self.grabbed = grabbed
                        self.frame = frame

        def read(self):
            with self.lock:
                if self.frame is None:
                    return False, None
                return self.grabbed, self.frame.copy()

        def stop(self):
            self.stopped = True
            time.sleep(0.1)
            self.stream.release()


    class InferenceWorker:
        """LOCAL mode only — Thread architecture với double-buffering."""
        def __init__(self):
            self.overlay_frame = None
            self.overlay_lock = Lock()
            self._frame_id = 0
            self._jpeg_buf = None

            self.latest_landmarks = None
            self.latest_landmarks_kpts = None
            self.landmarks_lock = Lock()

            self.stopped = False
            self.inference_fps = 0.0
            self.render_fps = 0.0

        def start(self):
            Thread(target=self._inference_loop, daemon=True, name="InferenceB1").start()
            Thread(target=self._render_loop, daemon=True, name="RenderB2").start()
            return self

        def _inference_loop(self):
            """B1: chỉ chạy Holistic, không vẽ gì cả."""
            prev_t = time.time()
            while not self.stopped:
                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                h, w = frame.shape[:2]
                scale = HOLISTIC_INPUT_WIDTH / w
                small = cv2.resize(frame, (HOLISTIC_INPUT_WIDTH, int(h * scale)))
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb_small.flags.writeable = False
                results = holistic.process(rgb_small)

                t_now = time.time()
                _smooth_landmark_list(results.pose_landmarks, pose_filter, t_now)
                _smooth_landmark_list(results.left_hand_landmarks, lhand_filter, t_now)
                _smooth_landmark_list(results.right_hand_landmarks, rhand_filter, t_now)

                kpts = extract_frame_keypoints(results, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

                with self.landmarks_lock:
                    self.latest_landmarks = results
                    self.latest_landmarks_kpts = kpts

                now = time.time()
                dt = now - prev_t
                if dt > 0:
                    self.inference_fps = 0.9 * self.inference_fps + 0.1 * (1.0 / dt)
                prev_t = now

        def _render_loop(self):
            """B2: đọc landmarks cached, vẽ overlay, flip."""
            prev_t = time.time()
            target_interval = 1.0 / 30.0

            while not self.stopped:
                loop_start = time.time()

                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                h, w = frame.shape[:2]

                with self.landmarks_lock:
                    results = self.latest_landmarks
                    kpts = self.latest_landmarks_kpts

                if results is not None:
                    line_thick = max(2, int(w / 640))
                    circle_r = max(3, int(w / 480))

                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 110, 200), thickness=line_thick, circle_radius=circle_r),
                        mp_drawing.DrawingSpec(color=(80, 80, 180), thickness=line_thick)
                    )
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=line_thick, circle_radius=circle_r),
                        mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=line_thick)
                    )
                    mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=line_thick, circle_radius=circle_r),
                        mp_drawing.DrawingSpec(color=(0, 120, 200), thickness=line_thick)
                    )

                if MIRROR_FRAME:
                    frame = cv2.flip(frame, 1)

                if kpts is not None:
                    step_state_machine(practice_state, kpts)

                now = time.time()
                dt = now - prev_t
                if dt > 0:
                    self.render_fps = 0.9 * self.render_fps + 0.1 * (1.0 / dt)
                prev_t = now

                # HUD
                ps = practice_state
                font = cv2.FONT_HERSHEY_SIMPLEX
                state = ps.state if ps.active else "IDLE"

                cv2.putText(frame, f"FPS: {int(self.render_fps)}", (w - 150, 30),
                            font, 0.7, (0, 255, 0), 2)

                if state == "WAIT":
                    elapsed = time.time() - ps.phase_start
                    remaining = max(0, 2.0 - elapsed)
                    count_num = int(remaining) + 1 if remaining > 0.05 else 0
                    if count_num > 0:
                        cv2.putText(frame, f"CHUAN BI... {count_num}",
                                    (50, 100), font, 1.5, (0, 255, 255), 3)

                elif state == "COLLECT":
                    n = len(ps.frames_queue)
                    progress = int((n / SEQUENCE_LENGTH) * 400)
                    cv2.rectangle(frame, (50, 50), (450, 80), (255, 255, 255), 2)
                    cv2.rectangle(frame, (50, 50), (50 + progress, 80), (255, 0, 0), -1)
                    cv2.putText(frame, f"THU THAP: {n}/{SEQUENCE_LENGTH}",
                                (50, 40), font, 0.8, (255, 255, 255), 2)

                elif state == "PREDICT":
                    cv2.putText(frame, "DANG PHAN TICH...",
                                (50, 100), font, 1.2, (200, 100, 255), 3)

                elif state == "SHOW":
                    if ps.last_top3 and len(ps.last_top3) >= 1:
                        t1 = ps.last_top3[0]
                        score1 = t1.get("score", 0)
                        label1 = t1.get("labelVn", t1.get("label", "?"))
                        main_color = (0, 255, 0) if ps.last_success else (0, 0, 255)
                        cv2.putText(frame,
                                    f"TOP 1: {label1} ({score1:.1f}%)",
                                    (50, 120), font, 1.2, main_color, 3)
                        if len(ps.last_top3) >= 2:
                            t2 = ps.last_top3[1]
                            cv2.putText(frame,
                                        f"Top 2: {t2.get('labelVn', t2.get('label', '?'))} ({t2.get('score', 0):.1f}%)",
                                        (50, 160), font, 0.8, (0, 255, 255), 2)
                        if len(ps.last_top3) >= 3:
                            t3 = ps.last_top3[2]
                            cv2.putText(frame,
                                        f"Top 3: {t3.get('labelVn', t3.get('label', '?'))} ({t3.get('score', 0):.1f}%)",
                                        (50, 190), font, 0.8, (0, 255, 255), 2)

                if STREAM_WIDTH is not None and w > STREAM_WIDTH:
                    stream_h = int(h * STREAM_WIDTH / w)
                    frame = cv2.resize(frame, (STREAM_WIDTH, stream_h),
                                       interpolation=cv2.INTER_AREA)

                ok, jpeg = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                ])

                with self.overlay_lock:
                    self.overlay_frame = frame
                    self._frame_id += 1
                    if ok:
                        self._jpeg_buf = jpeg.tobytes()

                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        def get_overlay(self):
            with self.overlay_lock:
                if self.overlay_frame is None:
                    return None, 0
                return self.overlay_frame, self._frame_id

        def get_jpeg(self):
            with self.overlay_lock:
                if self._jpeg_buf is None:
                    return None, 0
                return self._jpeg_buf, self._frame_id

        def stop(self):
            self.stopped = True

    # Khởi tạo camera + holistic + inference worker
    print("📷 Starting camera + Holistic (LOCAL mode)...")
    camera = WebcamVideoStream().start()
    time.sleep(1.0)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=HOLISTIC_COMPLEXITY
    )
    print("✅ Camera + Holistic ready.")

    print("🧠 Starting inference worker...")
    inference_worker = InferenceWorker().start()
    time.sleep(0.5)
    print("✅ Inference worker ready.")

else:
    print("🌐 REMOTE mode — no camera needed. Waiting for WebSocket frames.")


# ==========================================
# STATIC FILES
# ==========================================
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "main.html"))

@app.get("/main.html")
async def serve_main():
    return FileResponse(os.path.join(STATIC_DIR, "main.html"))

@app.get("/quiz.css")
async def serve_css():
    return FileResponse(os.path.join(STATIC_DIR, "quiz.css"), media_type="text/css")

@app.get("/quiz.js")
async def serve_js():
    return FileResponse(os.path.join(STATIC_DIR, "quiz.js"), media_type="application/javascript")


imgs_dir = os.path.join(STATIC_DIR, "imgs")
if os.path.isdir(imgs_dir):
    app.mount("/imgs", StaticFiles(directory=imgs_dir), name="imgs")
    print(f"✅ Mounted /imgs → {imgs_dir}")


# ==========================================
# REST
# ==========================================
@app.get("/health")
def health_check():
    fps = inference_worker.inference_fps if inference_worker else 0
    return {
        "status": "ok",
        "model_loaded": True,
        "camera_mode": CAMERA_MODE,
        "inference_fps": round(fps, 1)
    }

@app.get("/vocab_list")
def get_vocab_list():
    vocabs = [k for k in ACCENT_MAP.keys() if k != "NOTHING"]
    return {"vocabs": sorted(vocabs)}

@app.get("/mode")
def get_mode():
    """Client dùng endpoint này để biết server đang chạy mode nào."""
    return {"camera_mode": CAMERA_MODE}


# ==========================================
# MJPEG STREAM — chỉ cho LOCAL mode
# ==========================================
if CAMERA_MODE == "local":
    def _mjpeg_generator():
        print("[MJPEG] Client connected.")
        last_frame_id = 0

        while True:
            jpeg_bytes, frame_id = inference_worker.get_jpeg()

            if jpeg_bytes is None or frame_id == last_frame_id:
                time.sleep(0.003)
                continue

            last_frame_id = frame_id
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n\r\n'
                   + jpeg_bytes + b'\r\n')


    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            _mjpeg_generator(),
            media_type='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate, private',
                'Pragma': 'no-cache',
                'Expires': '0',
                'X-Accel-Buffering': 'no',
            }
        )


# ==========================================
# WEBSOCKET — Dual mode
# ==========================================
@app.websocket("/ws/practice")
async def websocket_practice(ws: WebSocket):
    await ws.accept()

    if CAMERA_MODE == "local":
        # ── LOCAL MODE: giống bản gốc, dùng global practice_state ──
        ps = practice_state
        pusher_task = None

        try:
            async def event_pusher():
                while True:
                    events = ps.pop_events()
                    for evt in events:
                        try:
                            await ws.send_json(evt)
                        except Exception:
                            return
                    await asyncio.sleep(0.05)

            pusher_task = asyncio.create_task(event_pusher())

            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                mtype = msg.get("type", "")

                if mtype == "start":
                    word = msg.get("target_word", "").strip().upper()
                    if word not in ACCENT_MAP:
                        await ws.send_json({"type": "error", "message": f"Từ '{word}' không hợp lệ."})
                        continue
                    ps.target_no_accent = ACCENT_MAP[word].upper()
                    ps.target_display = word
                    ps.frames_queue.clear()
                    ps.last_top3 = []
                    ps.state = "WAIT"
                    ps.phase_start = time.time()
                    ps.active = True
                    await ws.send_json({"type": "started", "target_word": word})

                elif mtype == "stop":
                    ps.active = False
                    ps.state = "IDLE"
                    ps.frames_queue.clear()
                    await ws.send_json({"type": "stopped"})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if pusher_task:
                pusher_task.cancel()
            ps.active = False

    else:
        # ── REMOTE MODE: per-session state, nhận frames từ client ──
        ps = PracticeState()
        pusher_task = None

        try:
            async def event_pusher():
                while True:
                    events = ps.pop_events()
                    for evt in events:
                        try:
                            await ws.send_json(evt)
                        except Exception:
                            return
                    await asyncio.sleep(0.05)

            pusher_task = asyncio.create_task(event_pusher())

            # Gửi thông tin mode cho client
            await ws.send_json({"type": "mode", "camera_mode": "remote"})

            while True:
                msg = await ws.receive()

                if msg["type"] == "websocket.receive":
                    # Binary message = JPEG frame từ client
                    if "bytes" in msg and msg["bytes"]:
                        jpeg_data = msg["bytes"]
                        # Xử lý frame trong thread pool để không block event loop
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, process_frame_remote, ps, jpeg_data)
                        # Gửi ack để client biết có thể gửi frame tiếp (Send-Then-Wait)
                        try:
                            await ws.send_json({"type": "frame_ack"})
                        except Exception:
                            break

                    # Text message = JSON control (start/stop)
                    elif "text" in msg and msg["text"]:
                        data = json.loads(msg["text"])
                        mtype = data.get("type", "")

                        if mtype == "start":
                            word = data.get("target_word", "").strip().upper()
                            if word not in ACCENT_MAP:
                                await ws.send_json({"type": "error", "message": f"Từ '{word}' không hợp lệ."})
                                continue
                            ps.target_no_accent = ACCENT_MAP[word].upper()
                            ps.target_display = word
                            ps.frames_queue.clear()
                            ps.last_top3 = []
                            ps.state = "WAIT"
                            ps.phase_start = time.time()
                            ps.active = True
                            # Reset filters cho lượt mới
                            ps.pose_filter.reset()
                            ps.lhand_filter.reset()
                            ps.rhand_filter.reset()
                            await ws.send_json({"type": "started", "target_word": word})

                        elif mtype == "stop":
                            ps.active = False
                            ps.state = "IDLE"
                            ps.frames_queue.clear()
                            await ws.send_json({"type": "stopped"})

                elif msg["type"] == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WS Remote Error] {e}")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if pusher_task:
                pusher_task.cancel()
            ps.active = False
            ps.close_holistic()


@app.on_event("shutdown")
def shutdown():
    print("🛑 Shutting down...")
    if inference_worker:
        inference_worker.stop()
    if camera:
        camera.stop()
    if holistic:
        holistic.close()