import os
import sys
import uvicorn
import html
import json
import asyncio
import signal
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import System
import ChatHistory
import Tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo và dọn dẹp tài nguyên theo lifecycle."""
    # ── STARTUP ──
    logger.info("Khởi động server...")
    System.preload_embedding_model()
    logger.info("Hệ thống đã sẵn sàng! (ChromaDB local + OpenAI)")
    yield
    # ── SHUTDOWN ──
    logger.info("=== Server shutting down ===")
    try:
        # 1. Cleanup toàn bộ resource
        await System.cleanup()

        # 2. Đóng SQLite connection
        await ChatHistory.close()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

    # 3. Giải phóng port
    Tools.release_port(port=8000)

    logger.info("=== Server shutdown complete ===")


app = FastAPI(title="VieSign AI API", lifespan=lifespan)

ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str | None = None

    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return html.escape(v.strip())


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)
    links: list[dict] = Field(default_factory=list)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if req.session_id:
        session = await ChatHistory.load_session(req.session_id)
        if not session:
            session = await ChatHistory.create_session()
    else:
        session = await ChatHistory.create_session()

    session_id = session["session_id"]

    conversation_history = ChatHistory.get_history_list(
        session["messages"],
        max_turns=System.config.MAX_CONTEXT_TURNS
    )

    logger.info(f"Processing query for session {session_id}: {req.message[:100]}...")
    try:
        result = await System.run_query(
            query=req.message,
            conversation_history=conversation_history,
        )
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    response_text = result.get("answer", "")
    tools_used = result.get("tools_used", [])
    links = result.get("links", [])

    if not response_text or not response_text.strip():
        response_text = "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."

    await ChatHistory.append_messages(
        session, req.message, response_text,
        tools_used=tools_used, links=links,
    )

    logger.info(f"Completed query for session {session_id} ({len(tools_used)} tools)")

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        tools_used=tools_used,
        links=links,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if req.session_id:
        session = await ChatHistory.load_session(req.session_id)
        if not session:
            session = await ChatHistory.create_session()
    else:
        session = await ChatHistory.create_session()

    session_id = session["session_id"]
    conversation_history = ChatHistory.get_history_list(
        session["messages"],
        max_turns=System.config.MAX_CONTEXT_TURNS
    )

    async def generate():
        try:
            full_response = ""
            async for event in System.run_query_streaming(
                query=req.message,
                conversation_history=conversation_history,
            ):
                if event.get("type") == "info":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event.get("type") == "token":
                    full_response += event["content"]
                    yield f"data: {json.dumps(event)}\n\n"
                elif event.get("type") == "_done":
                    tools_used = event.get('tools_used', [])
                    links = event.get('links', [])
                    if full_response.strip():
                        await ChatHistory.append_messages(
                            session, req.message, full_response,
                            tools_used=tools_used, links=links,
                        )
                    done_event = {
                        'type': 'done',
                        'session_id': session_id,
                        'tools_used': tools_used,
                        'links': links,
                    }
                    yield f"data: {json.dumps(done_event)}\n\n"

            if not full_response.strip():
                fallback = "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."
                yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'tools_used': []})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Lỗi xử lý: vui lòng thử lại'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history")
async def get_history_list():
    sessions = await ChatHistory.list_sessions()
    return {"sessions": sessions}


@app.get("/api/history/{session_id}")
async def get_history_detail(session_id: str):
    session = await ChatHistory.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return {"session": session}


@app.delete("/api/history/{session_id}")
async def delete_history(session_id: str):
    success = await ChatHistory.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return {"message": "Đã xoá phiên thành công.", "session_id": session_id}


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0-chroma"}


# ── Serve frontend static files ──────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"Frontend mounted from: {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_DIR}")


def _handle_signal(sig, frame):
    """Handler cho SIGINT/SIGTERM — gọi cleanup đồng bộ rồi thoát."""
    sig_name = signal.Signals(sig).name
    logger.warning(f"Received {sig_name}, initiating shutdown...")
    logger.info("Giải phóng tài nguyên (cleanup sync)...")
    Tools.cleanup_sync()
    logger.info(f"Shutdown complete after {sig_name}.")
    sys.exit(0)


if __name__ == "__main__":
    # ── Đăng ký signal handler cho trường hợp uvicorn không bắt kịp ──
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    shutdown_timeout = System.config.SHUTDOWN_TIMEOUT

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=debug_mode,
    )