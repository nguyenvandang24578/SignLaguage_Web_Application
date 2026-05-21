"""
Sign Language Quiz — HYBRID Server (browser-side MediaPipe + server STA-GCN)
==============================================================================

Kiến trúc:
  Browser:
    - getUserMedia (camera local, nét tối đa)
    - MediaPipe Tasks JS: PoseLandmarker + HandLandmarker
    - Extract 27 keypoints theo convention training
    - One-Euro Filter ở client để smooth
    - Vẽ skeleton lên canvas overlay (0ms lag — local render)
    - Gửi keypoints qua WebSocket (~324 bytes/frame thay vì ~100KB JPEG)

  Server (file này):
    - KHÔNG có OpenCV camera, KHÔNG có Holistic, KHÔNG có MJPEG stream
    - Chỉ: WebSocket nhận keypoints → state machine → STA-GCN predict → trả JSON
    - Lightweight, có thể chạy trên Raspberry Pi nếu muốn

Protocol (WebSocket /ws/practice):
  Client → Server:
    { type: "start",  target_word: "CHÀO" }
    { type: "keypoints", joints: [[x,y,vis], ... ×27] }
    { type: "stop" }

  Server → Client:
    { type: "started", target_word }
    { type: "status",  state: WAIT|COLLECT|PREDICT|SHOW, ... }
    { type: "result",  success: bool, top3: [...] }
    { type: "stopped" }
    { type: "error",   message }

Để dùng port khác với server.py cũ (8000):
  uvicorn server_hybrid:app --port 8001
"""

import os
import time
import json
import asyncio
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from threading import Lock
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.sta_gcn import Model


# ==========================================
# PATHS
# ==========================================
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.dirname(SERVER_DIR)
print(f"📁 Server dir: {SERVER_DIR}")
print(f"📁 Static dir: {STATIC_DIR}")


app = FastAPI(title="Sign Language Quiz — Hybrid Server")
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
WEIGHT_PATH = os.path.join(SERVER_DIR, 'stagcn_tiny_supcon_50cls_best.pt')
PREDICTION_THRESHOLD = 0.40

# Coordinate space khi STA-GCN training extract keypoints
# (browser sẽ tự nhân landmarks normalized × MODEL_INPUT_* trước khi gửi,
#  hoặc gửi normalized rồi server nhân — chọn cách 2 để client nhẹ hơn)
MODEL_INPUT_WIDTH = 1280
MODEL_INPUT_HEIGHT = 720

# ==========================================
# VOCAB
# ==========================================
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
# LOAD MODEL
# ==========================================
print("⚙️  Loading STA-GCN model...")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"   → CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"   → Apple Silicon MPS")
else:
    device = torch.device('cpu')
    print(f"   → CPU")

model = Model(
    num_class=NUM_CLASSES, num_point=NUM_JOINTS, num_person=1,
    graph='graph_stagcn.sign_27.Graph', graph_args={}
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device).eval()

# Warmup
print("   → Warming up...")
with torch.no_grad():
    dummy = torch.randn(1, 3, SEQUENCE_LENGTH, NUM_JOINTS, 1).to(device)
    for _ in range(3):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
print("✅ Model loaded + warmed up.")


# ==========================================
# PREPROCESS — match exactly server.py original
# ==========================================
def preprocess_for_model(frames_list):
    """
    frames_list: list of (27, 3) np arrays
    output: torch tensor (1, C=3, T=60, V=27, M=1)
    """
    data = np.array(frames_list)            # (T, V, C)
    data = np.transpose(data, (2, 0, 1))    # (C, T, V)
    data = np.expand_dims(data, axis=-1)    # (C, T, V, 1)
    # Center hóa X và Y theo frame đầu tiên, joint đầu tiên
    data[0] -= data[0, :, 0, 0].mean(axis=0)
    data[1] -= data[1, :, 0, 0].mean(axis=0)
    return torch.from_numpy(data).unsqueeze(0).float()


# ==========================================
# STATIC FILES (serve quiz UI từ parent dir)
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

# Mount imgs nếu có
imgs_dir = os.path.join(STATIC_DIR, "imgs")
if os.path.isdir(imgs_dir):
    app.mount("/imgs", StaticFiles(directory=imgs_dir), name="imgs")


# ==========================================
# REST
# ==========================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "mode": "hybrid",
        "model_loaded": True,
        "device": str(device),
        "sequence_length": SEQUENCE_LENGTH,
        "num_joints": NUM_JOINTS,
    }

@app.get("/vocab_list")
def get_vocab_list():
    vocabs = [k for k in ACCENT_MAP.keys() if k != "NOTHING"]
    return {"vocabs": sorted(vocabs)}

@app.get("/model_info")
def get_model_info():
    """Endpoint cho browser biết training convention để extract đúng."""
    return {
        "num_joints": NUM_JOINTS,
        "sequence_length": SEQUENCE_LENGTH,
        "input_resolution": {"width": MODEL_INPUT_WIDTH, "height": MODEL_INPUT_HEIGHT},
        "pose_slot_to_mp_index": {
            "0": {"mp": 0,  "name": "NOSE"},
            "1": {"mp": 11, "name": "LEFT_SHOULDER"},
            "2": {"mp": 12, "name": "RIGHT_SHOULDER"},
            "3": {"mp": 13, "name": "LEFT_ELBOW"},
            "4": {"mp": 14, "name": "RIGHT_ELBOW"},
            "5": {"mp": 15, "name": "LEFT_WRIST"},
            "6": {"mp": 16, "name": "RIGHT_WRIST"},
        },
        "hand_slot_to_mp_index": [
            {"slot_offset": 0, "mp": 0,  "name": "WRIST"},
            {"slot_offset": 1, "mp": 4,  "name": "THUMB_TIP"},
            {"slot_offset": 2, "mp": 5,  "name": "INDEX_MCP"},
            {"slot_offset": 3, "mp": 8,  "name": "INDEX_TIP"},
            {"slot_offset": 4, "mp": 9,  "name": "MIDDLE_MCP"},
            {"slot_offset": 5, "mp": 12, "name": "MIDDLE_TIP"},
            {"slot_offset": 6, "mp": 13, "name": "RING_MCP"},
            {"slot_offset": 7, "mp": 16, "name": "RING_TIP"},
            {"slot_offset": 8, "mp": 17, "name": "PINKY_MCP"},
            {"slot_offset": 9, "mp": 20, "name": "PINKY_TIP"},
        ],
        # Convention nội bộ giữ nguyên từ server.py cũ để đảm bảo không bị
        # mis-align với weights đã train. Browser MUST follow exactly:
        #   pts[7..16]  = LEFT  hand (categoryName="Left" trong MediaPipe JS)
        #   pts[17..26] = RIGHT hand (categoryName="Right")
        "hand_slot_mapping": {
            "left_hand_starts_at_slot": 7,
            "right_hand_starts_at_slot": 17,
            "note": "Following original server.py convention. Do not change."
        }
    }


# ==========================================
# WEBSOCKET — main hybrid endpoint
# ==========================================
@app.websocket("/ws/practice")
async def websocket_practice(ws: WebSocket):
    await ws.accept()

    # Per-connection state (không global, mỗi client một state riêng)
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    state = "IDLE"
    target_no_accent = ""
    target_display = ""
    phase_start = 0.0
    last_top3 = []

    # Stats cho ablation paper sau này
    frames_received = 0
    predictions_made = 0
    session_start = time.time()

    print("[WS-Hybrid] Client connected")

    try:
        # Background task: push WAIT countdown ticks (server tự đếm thay vì client)
        async def countdown_pusher():
            nonlocal state, phase_start, frames_queue
            while True:
                await asyncio.sleep(0.1)  # 10 Hz cho countdown
                if state == "WAIT":
                    elapsed = time.time() - phase_start
                    remaining = max(0.0, 2.0 - elapsed)
                    try:
                        await ws.send_json({
                            "type": "status",
                            "state": "WAIT",
                            "countdown": round(remaining, 2),
                            "message": f"Chuẩn bị... {int(remaining) + 1}"
                                       if remaining > 0.05 else "BẮT ĐẦU!"
                        })
                    except Exception:
                        return
                    if elapsed >= 2.0:
                        state = "COLLECT"
                        frames_queue.clear()
                        try:
                            await ws.send_json({
                                "type": "status", "state": "COLLECT",
                                "progress": 0.0,
                                "message": "Hãy thực hiện ký hiệu!"
                            })
                        except Exception:
                            return
                elif state == "SHOW":
                    elapsed = time.time() - phase_start
                    if elapsed >= 3.0:
                        state = "WAIT"
                        phase_start = time.time()
                        frames_queue.clear()

        countdown_task = asyncio.create_task(countdown_pusher())

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type", "")

            # ─── START ───
            if mtype == "start":
                word = msg.get("target_word", "").strip().upper()
                if word not in ACCENT_MAP:
                    await ws.send_json({"type": "error",
                                       "message": f"Từ '{word}' không hợp lệ."})
                    continue
                target_no_accent = ACCENT_MAP[word].upper()
                target_display = word
                frames_queue.clear()
                last_top3 = []
                state = "WAIT"
                phase_start = time.time()
                await ws.send_json({"type": "started", "target_word": word})

            # ─── KEYPOINTS ───
            elif mtype == "keypoints" and state == "COLLECT":
                joints_raw = msg.get("joints")
                if joints_raw is None:
                    continue

                try:
                    kpts = np.asarray(joints_raw, dtype=np.float32)
                except Exception:
                    continue

                if kpts.shape != (NUM_JOINTS, 3):
                    await ws.send_json({
                        "type": "error",
                        "message": f"Sai shape keypoints {kpts.shape}, cần ({NUM_JOINTS}, 3)"
                    })
                    continue

                frames_queue.append(kpts)
                frames_received += 1
                progress = len(frames_queue) / SEQUENCE_LENGTH

                # Báo progress mỗi 5 frame để không spam
                if len(frames_queue) % 5 == 0 or len(frames_queue) == SEQUENCE_LENGTH:
                    await ws.send_json({
                        "type": "status",
                        "state": "COLLECT",
                        "progress": round(progress, 3),
                        "message": f"Thu thập: {len(frames_queue)}/{SEQUENCE_LENGTH}"
                    })

                # Đủ 60 frames → predict
                if len(frames_queue) == SEQUENCE_LENGTH:
                    state = "PREDICT"
                    await ws.send_json({
                        "type": "status", "state": "PREDICT",
                        "message": "Đang phân tích..."
                    })

                    # Run inference (nhanh, không block lâu)
                    t0 = time.perf_counter()
                    inp = preprocess_for_model(list(frames_queue)).to(device)
                    with torch.no_grad():
                        out, _ = model(inp)
                        prob = F.softmax(out, dim=1).cpu().numpy()[0]
                    inference_ms = (time.perf_counter() - t0) * 1000
                    predictions_made += 1

                    idx3 = np.argsort(prob)[-3:][::-1]
                    top3 = [{
                        "label": MODEL_DICT[i].upper(),
                        "labelVn": NO_ACCENT_TO_ACCENT.get(
                            MODEL_DICT[i].upper(), MODEL_DICT[i].upper()),
                        "score": round(float(prob[i]) * 100, 1)
                    } for i in idx3]
                    last_top3 = top3

                    is_correct = (top3[0]["label"] == target_no_accent
                                  and top3[0]["score"] >= PREDICTION_THRESHOLD * 100)

                    await ws.send_json({
                        "type": "result",
                        "success": is_correct,
                        "top3": top3,
                        "inference_ms": round(inference_ms, 2),
                        "message": ("CHÍNH XÁC!" if is_correct
                                    else f"AI nhận: {top3[0]['labelVn']}")
                    })

                    if is_correct:
                        state = "IDLE"
                    else:
                        state = "SHOW"
                        phase_start = time.time()

            # ─── STOP ───
            elif mtype == "stop":
                state = "IDLE"
                frames_queue.clear()
                await ws.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS-Hybrid] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Cleanup
        try:
            countdown_task.cancel()
        except Exception:
            pass

        elapsed = time.time() - session_start
        print(f"[WS-Hybrid] Disconnected. "
              f"Session: {elapsed:.1f}s, "
              f"frames: {frames_received}, "
              f"predictions: {predictions_made}")


if __name__ == "__main__":
    import uvicorn
    # Default port 8001 để không xung đột với server.py cũ (8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)
