## Cấu trúc thư mục

```
chatbot/
├── backend/                   # Backend Python (FastAPI)
│   ├── src/                   # Mã nguồn Python
│   │   ├── server.py          # FastAPI server
│   │   ├── System.py          # Hệ thống AI
│   │   ├── Tools.py           # Tools & ChromaDB
│   │   ├── ChatHistory.py     # Quản lý lịch sử chat
│   │   └── Create_vectorDB.py # Tạo vector database
│   ├── data/                  # Dữ liệu SQLite chat
│   ├── data_vsl/              # Dữ liệu ngôn ngữ ký hiệu
│   ├── chroma_db/             # Vector database
│   ├── .env                   # Biến môi trường
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                  # Frontend web
│   ├── index.html             # Trang chính
│   ├── css/style.css          # Stylesheet
│   ├── js/app.js              # JavaScript
│   └── assets/                # Hình ảnh
│       ├── account.png
│       └── robot.png
├── run_servers.sh             # Script chạy cả llama-server + backend
└── README.md
```

## Run Server backend

```bash
cd chatbot/backend
python src/server.py
```

## Run both llama-server + backend

```bash
cd chatbot
bash run_servers.sh
```

---

**_Note: Create `.env` before running server.py**

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp chatbot/backend/.env.example chatbot/backend/.env
```

Required environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `SERPAPI_KEY` - (Optional) For web search
