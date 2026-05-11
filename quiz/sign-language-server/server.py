"""
Sign Language Quiz — FastAPI Server (WebSocket Architecture)
============================================================
Camera chạy 100% trong browser (getUserMedia).
Browser gửi frame qua WebSocket → Server xử lý MediaPipe + STA-GCN → trả JSON realtime.
Không dùng cv2.imshow, không blocking, không native window.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import time
import base64
import json
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import model từ project
from models.sta_gcn import Model

app = FastAPI(title="Sign Language Quiz API")

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
NUM_JOINTS = 27
SEQUENCE_LENGTH = 60
NUM_CLASSES = 50
WEIGHT_PATH = './stagcn_tiny_supcon_50cls_best.pt'
PREDICTION_THRESHOLD = 0.40

mp_holistic = mp.solutions.holistic

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
# KEYPOINT / SKELETON EXTRACTION
# ==========================================
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


def extract_skeleton_for_browser(results, w, h):
    """Trích skeleton landmarks để browser vẽ overlay trên canvas."""
    skel = {"pose": [], "leftHand": [], "rightHand": []}
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            skel["pose"].append([round(lm.x * w, 1), round(lm.y * h, 1), round(lm.visibility, 2)])
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            skel["leftHand"].append([round(lm.x * w, 1), round(lm.y * h, 1)])
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            skel["rightHand"].append([round(lm.x * w, 1), round(lm.y * h, 1)])
    return skel


def preprocess_for_model(frames_list):
    data = np.array(frames_list)            # (T, V, C)
    data = np.transpose(data, (2, 0, 1))    # (C, T, V)
    data = np.expand_dims(data, axis=-1)    # (C, T, V, 1)
    data[0] -= data[0, :, 0, 0].mean(axis=0)
    data[1] -= data[1, :, 0, 0].mean(axis=0)
    return torch.from_numpy(data).unsqueeze(0).float()


async def step_state_machine(kpts, state, frames_queue, phase_start,
                              last_top3, target_no_accent, target_display):
    """
    Pure state-machine step. Both `keypoints` and `frame` handlers funnel here
    after they've produced a (27, 3) np.array of keypoints. Returns the response
    dict augmented with `_next_state`, `_next_phase_start`, `_last_top3` metadata
    for the caller to update its local state.
    """
    resp = {
        "type": "status",
        "state": state,
        "top3": last_top3,
        "progress": 0,
        "countdown": 0,
        "message": "",
        "_next_state": state,
        "_next_phase_start": phase_start,
        "_last_top3": last_top3,
    }

    if state == "WAIT":
        elapsed = time.time() - phase_start
        remaining = max(0, 2.5 - elapsed)
        resp["countdown"] = round(remaining, 1)
        resp["message"] = f"Chuẩn bị... {int(remaining + 0.5)}"
        if elapsed >= 2.0:
            resp["_next_state"] = "COLLECT"
            resp["state"] = "COLLECT"
            frames_queue.clear()

    elif state == "COLLECT":
        frames_queue.append(kpts)
        progress = len(frames_queue) / SEQUENCE_LENGTH
        resp["progress"] = round(progress, 3)
        resp["message"] = f"Thu thập: {len(frames_queue)}/{SEQUENCE_LENGTH}"
        if len(frames_queue) == SEQUENCE_LENGTH:
            resp["_next_state"] = "PREDICT"
            # Fall through to PREDICT this very tick to minimize perceived lag
            state = "PREDICT"

    if state == "PREDICT":
        inp = preprocess_for_model(frames_queue).to(device)
        with torch.no_grad():
            out, _ = model(inp)
            prob = F.softmax(out, dim=1).cpu().numpy()[0]
            idx3 = np.argsort(prob)[-3:][::-1]
            top3 = []
            for i in idx3:
                label = MODEL_DICT[i].upper()
                top3.append({
                    "label": label,
                    "labelVn": NO_ACCENT_TO_ACCENT.get(label, label),
                    "score": round(float(prob[i]) * 100, 1)
                })

        resp["top3"] = top3
        resp["_last_top3"] = top3

        # Check correctness
        if (top3[0]["label"] == target_no_accent
                and top3[0]["score"] >= PREDICTION_THRESHOLD * 100):
            resp["type"] = "result"
            resp["success"] = True
            resp["message"] = f"CHÍNH XÁC! '{target_display}'"
            resp["_next_state"] = "IDLE"
        else:
            resp["_next_state"] = "SHOW"
            resp["state"] = "SHOW"
            resp["_next_phase_start"] = time.time()
            predicted = top3[0]["labelVn"] if top3[0]["score"] >= PREDICTION_THRESHOLD * 100 else "KHÔNG RÕ"
            resp["message"] = f"AI nhận: {predicted}"

    elif state == "SHOW":
        resp["top3"] = last_top3
        elapsed = time.time() - phase_start
        resp["message"] = "Xem kết quả..."
        if elapsed >= 3.0:
            resp["_next_state"] = "WAIT"
            resp["state"] = "WAIT"
            resp["_next_phase_start"] = time.time()
            frames_queue.clear()
            resp["message"] = "Lượt mới..."

    return resp


# ==========================================
# LOAD MODEL
# ==========================================
print("⚙️  Loading STA-GCN model...")
device = torch.device('cpu')
model = Model(
    num_class=NUM_CLASSES, num_point=NUM_JOINTS, num_person=1,
    graph='graph_stagcn.sign_27.Graph', graph_args={}
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device).eval()
print("✅ Model loaded. Server ready.")


# ==========================================
# STATIC FILE SERVING
# ==========================================
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/quiz.css")
async def serve_css():
    return FileResponse("quiz.css", media_type="text/css")

@app.get("/quiz.js")
async def serve_js():
    return FileResponse("quiz.js", media_type="application/javascript")


# ==========================================
# REST ENDPOINTS
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.get("/vocab_list")
def get_vocab_list():
    vocabs = [k for k in ACCENT_MAP.keys() if k != "NOTHING"]
    return {"vocabs": sorted(vocabs)}


# ==========================================
# WEBSOCKET: PRACTICE SESSION (CORE)
# ==========================================
@app.websocket("/ws/practice")
async def websocket_practice(ws: WebSocket):
    """
    Protocol:
    → { type: "start", target_word: "CHÀO" }
    → { type: "keypoints", joints: [[x,y,v], ...27 joints] }  (PREFERRED — browser MediaPipe)
    → { type: "frame", image: "data:image/jpeg;base64,..." }  (LEGACY — server MediaPipe)
    ← { type: "status", state, top3, progress, ... }
    ← { type: "result", success: true/false, ... }
    → { type: "stop" }

    Hybrid architecture: browser runs MediaPipe (PoseLandmarker + HandLandmarker via
    Tasks-Vision API) and extracts 27 joints in the SAME indexing as the training feeder.
    This eliminates ~150ms round-trip latency from skeleton overlay and reduces bandwidth
    from ~50KB/frame (JPEG) to ~324B/frame (27 × 3 × 4 bytes).
    """
    await ws.accept()

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    state = "IDLE"
    target_no_accent = ""
    target_display = ""
    phase_start = 0.0
    last_top3 = []
    holistic = None  # Only instantiated when a legacy `frame` message arrives

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type", "")

            # ── START ──
            if mtype == "start":
                word = msg.get("target_word", "").strip().upper()
                if word not in ACCENT_MAP:
                    await ws.send_json({"type": "error", "message": f"Từ '{word}' không hợp lệ."})
                    continue

                target_no_accent = ACCENT_MAP[word].upper()
                target_display = word
                frames_queue.clear()
                last_top3 = []
                state = "WAIT"
                phase_start = time.time()

                # Keep Holistic alive across retries — closing+reopening adds
                # ~200ms of model init on every retry. It'll get lazily created
                # on first `frame` message if not yet present.
                await ws.send_json({"type": "started", "target_word": target_display})

            # ── KEYPOINTS (PREFERRED, low-latency hybrid) ──
            elif mtype == "keypoints" and state != "IDLE":
                joints_raw = msg.get("joints", None)
                if joints_raw is None:
                    continue

                # Validate shape: must be exactly (27, 3)
                try:
                    kpts = np.asarray(joints_raw, dtype=np.float32)
                    if kpts.shape != (NUM_JOINTS, 3):
                        await ws.send_json({
                            "type": "error",
                            "message": f"Sai shape keypoints: {kpts.shape}, cần ({NUM_JOINTS}, 3)"
                        })
                        continue
                except Exception:
                    continue

                resp = await step_state_machine(
                    kpts, state, frames_queue, phase_start,
                    last_top3, target_no_accent, target_display
                )
                # Unpack updated state from resp metadata
                state = resp.pop("_next_state", state)
                phase_start = resp.pop("_next_phase_start", phase_start)
                last_top3 = resp.pop("_last_top3", last_top3)
                await ws.send_json(resp)
                # Detect terminal success (state moved to IDLE inside helper)
                if resp.get("type") == "result" and resp.get("success"):
                    pass  # State already set to IDLE

            # ── FRAME (LEGACY, server-side MediaPipe) ──
            elif mtype == "frame" and state != "IDLE":
                img_b64 = msg.get("image", "")
                if not img_b64:
                    continue

                # Lazy-create holistic (only paid when legacy path is used)
                if holistic is None:
                    holistic = mp_holistic.Holistic(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        model_complexity=0
                    )

                # Decode JPEG
                try:
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    buf = base64.b64decode(img_b64)
                    arr = np.frombuffer(buf, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                kpts = extract_frame_keypoints(results, w, h)

                resp = await step_state_machine(
                    kpts, state, frames_queue, phase_start,
                    last_top3, target_no_accent, target_display
                )
                # Attach skeleton ONLY for legacy clients that still want it
                resp["skeleton"] = extract_skeleton_for_browser(results, w, h)
                resp["frameW"] = w
                resp["frameH"] = h
                state = resp.pop("_next_state", state)
                phase_start = resp.pop("_next_phase_start", phase_start)
                last_top3 = resp.pop("_last_top3", last_top3)
                await ws.send_json(resp)

            # ── STOP ──
            elif mtype == "stop":
                state = "IDLE"
                frames_queue.clear()
                if holistic:
                    holistic.close()
                    holistic = None
                await ws.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if holistic:
            holistic.close()
