# SignVN — Quiz Module

Real-time Vietnamese Sign Language recognition quiz. Webcam-based practice with **STA-GCN** model running on a Python WebSocket server; thin browser client.

## Prerequisites

- Python **3.10** (3.11 works; avoid 3.12+ — MediaPipe wheels lag behind)
- A webcam
- Model weights file: `stagcn_tiny_supcon_50cls_best.pt` (place in the same folder as `server.py`)
- Model package: `models/sta_gcn.py` + `graph_stagcn/sign_27.py` (already in the project)

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install opencv-python mediapipe
```

> **Note:** the current `requirements.txt` is missing `opencv-python` and `mediapipe` — both are required by `server.py`. Install them manually as shown above, or add these two lines to `requirements.txt`:
> ```
> opencv-python==4.8.1.78
> mediapipe==0.10.14
> ```

## Run the server (mandatory — must start before opening the page)

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

You should see:

```
⚙️  Loading STA-GCN model...
✅ Model loaded. Server ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Keep this terminal open. The server handles:
- MediaPipe Holistic for keypoint extraction (server-side)
- STA-GCN inference (50-class Vietnamese sign recognition)
- WebSocket session state machine (`/ws/practice`)

## Open the frontend

Open `index.html` in a modern browser (**Chrome / Edge recommended**).

The page connects to `ws://127.0.0.1:8000/ws/practice` automatically. If you're serving the frontend from elsewhere, update `WS_URL` in `quiz.js`.

> **Camera permission:** the browser will prompt for webcam access. `getUserMedia` only works on `localhost`, `127.0.0.1`, or HTTPS — not raw `file://` in some browsers. If you hit issues, serve the page with a simple static server:
> ```bash
> python -m http.server 5500
> # then open http://localhost:5500/index.html
> ```

## How it works (short)

1. Browser captures webcam at 480×360, encodes JPEG (q=0.5), sends to server via WebSocket
2. Server runs MediaPipe Holistic → extracts 27 keypoints (matching the training feeder)
3. After 60 frames collected, STA-GCN predicts top-3 classes
4. If top-1 matches target word with confidence ≥ 40%, success is signaled back to browser

Send-then-wait pattern keeps at most one frame in network at a time — no backlog, no lag.

## Quiz modes

- **Giải Mã Ký Hiệu** — multiple-choice quiz from sign videos
- **Luyện Tập Chỉ Định** — practice a specific word until AI recognizes it
- **Một Từ Ngẫu Nhiên** — practice a random word freely
- **Thử Thách Có Chấm Điểm** — N-word timed challenge with scoring

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| `FileNotFoundError: stagcn_tiny_...pt` | Place the weight file next to `server.py` |
| WebSocket fails | Check the server is running on port 8000; check firewall |
| Camera permission denied | Use `http://localhost`, not `file://`; reset site permissions |
| Lag / high CPU | Lower `CAMERA_WIDTH`/`CAMERA_HEIGHT` in `quiz.js` (e.g. 320×240) |
