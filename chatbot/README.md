## Run Server backend

```bash
cd chatbot/backend
python server.py
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
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `SERPAPI_KEY` - (Optional) For web search
