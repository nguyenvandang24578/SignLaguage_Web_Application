# VNSignMate — Học Ngôn Ngữ Ký Hiệu Việt Nam

Nền tảng học ngôn ngữ ký hiệu Việt Nam (VNSL) với bài học theo chủ đề, mô hình 3D tương tác, nhận diện tay thời gian thực và trợ lý AI.

## 🚀 Chạy bằng Docker (khuyên dùng)

### Yêu cầu

- [Docker](https://docs.docker.com/get-docker/) (>= 24.0)
- [Docker Compose](https://docs.docker.com/compose/install/) (>= 2.20)

### Cách chạy

```bash
# 1. Clone repo
git clone https://github.com/nguyenvandang24578/SignLaguage_Web_Application.git
cd SignLaguage_Web_Application

# 2. (Quan trọng) Cấu hình API key
#    Sửa file chatbot/backend/.env, điền OPENAI_API_KEY của bạn

# 3. Build và chạy
docker compose up -d

# 4. Mở trình duyệt
#    → http://localhost:8080
```

### Dừng ứng dụng

```bash
docker compose down
```

### Xem logs

```bash
docker compose logs -f
# hoặc xem riêng từng service:
docker compose logs -f backend
docker compose logs -f app
```

### Build lại sau khi thay đổi code

```bash
docker compose up -d --build
```

## 🐳 Cấu trúc Docker

```
docker-compose.yml          # Orchestrator: định nghĩa 2 service
├── backend                 # FastAPI + ChromaDB (Python 3.11)
│   ├── chatbot/backend/Dockerfile
│   ├── chatbot/backend/.env           ← API keys (cần cấu hình)
│   └── chatbot/backend/src/           ← Mã nguồn Python
├── app                     # Nginx: serve static files + reverse proxy
│   ├── nginx.conf                    ← Config nginx
│   ├── index.html                    ← Trang chủ
│   ├── learn/                        ← Module học
│   ├── quiz/                         ← Module kiểm tra
│   └── chatbot/frontend/             ← Chatbot UI
└── volumes (persistent)
    ├── vsl_chat_history              ← Lịch sử chat (SQLite)
    └── vsl_chroma_db                 ← Vector database (ChromaDB)
```

### Chi tiết các service

| Service  | Cổng ngoài | Mô tả                              |
| -------- | ---------- | ----------------------------------- |
| `app`    | `:8080`    | Nginx: static files + proxy `/api`  |
| `backend`| `:8000`    | FastAPI backend + ChromaDB + OpenAI |

### Luồng request

```
Trình duyệt → localhost:8080 → Nginx
  ├── / (static) → serve file trực tiếp
  └── /api/*    → proxy sang backend:8000
```

## 🖥️ Chạy không Docker (development)

### Backend

```bash
cd chatbot/backend
pip install -r requirements.txt
python src/server.py
# → http://localhost:8000
```

### Frontend

Mở trực tiếp file `index.html` trong trình duyệt, hoặc dùng Live Server (VSCode) ở port 5500.

## 📁 Cấu trúc project

```
SignLaguage_Web_Application/
├── index.html              # Trang chủ (hub)
├── learn/                  # Module học ký hiệu
├── quiz/                   # Module kiểm tra + server nhận diện
├── chatbot/                # Chatbot AI
│   ├── backend/            # FastAPI + ChromaDB
│   │   ├── src/            # Mã nguồn Python
│   │   ├── Dockerfile
│   │   └── .env            # API keys
│   └── frontend/           # Giao diện chatbot
├── docker-compose.yml      # Docker Compose (chạy toàn bộ)
├── nginx.conf              # Nginx config
└── .gitignore
```
