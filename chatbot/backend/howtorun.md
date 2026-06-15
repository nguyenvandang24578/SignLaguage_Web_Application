# Hướng dẫn chạy Chatbot Backend (Docker)

## Yêu cầu

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) hoặc Docker Engine (Linux)
- Kết nối internet (để pull image và gọi API)

---

## 1. Chuẩn bị model Qwen 3

Tạo folder `models/` và tải model GGUF vào:

```bash
cd chatbot/backend
mkdir models
```

Tải model từ HuggingFace:

```bash
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf --local-dir ./models
mv models/qwen3-8b-q4_k_m.gguf models/qwen3.gguf
```

> **Lưu ý**: Chọn bản quantize phù hợp RAM:
>
> - Q4_K_M (~5GB) — khuyên dùng, cân bằng chất lượng/tốc độ
> - Q8_0 (~9GB) — chất lượng cao hơn, cần nhiều RAM hơn

---

## 2. Cấu hình .env

Đảm bảo file `.env` có đủ các biến:

```env
OPENAI_API_KEY=sk-proj-...
QDRANT_URL=https://...
QDRANT_API_KEY=...
COLLECTION_NAME=vsl_knowledge_base
LLAMA_SERVER_URL=http://llama-server:8080/v1/chat/completions
```

---

## 3. Chạy

```bash
cd chatbot/backend
docker compose up --build
```

Lần đầu sẽ mất vài phút để build image và pull dependencies.

---

## 4. Kiểm tra

- Backend API: http://localhost:8000/health
- Llama server: http://localhost:8080

---

## 5. Dừng

```bash
docker compose down
```

---

## 6. Nếu có GPU NVIDIA

Trong `docker-compose.yml`, uncomment phần deploy của `llama-server`:

```yaml
llama-server:
  image: ghcr.io/ggerganov/llama.cpp:server-cuda # đổi image
  command: >
    --host 0.0.0.0
    --port 8080
    -m /models/qwen3.gguf
    -c 4096
    -ngl 99                                          # offload tất cả layers lên GPU
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

---

## 7. Truy cập shell/debug

```bash
# Mở bash trong container backend
docker exec -it chatbot-backend bash

# Chạy Python
docker exec -it chatbot-backend python
```

---

## Cấu trúc file

```
chatbot/backend/
├── models/
│   └── qwen3.gguf          ← model LLM
├── data/
│   └── chat.db             ← database (auto-generated)
├── .env                    ← API keys & config
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
├── server.py
├── System.py
├── Tools.py
├── ChatHistory.py
└── requirements.txt
```
