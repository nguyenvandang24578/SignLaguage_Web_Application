import os
import uvicorn
import html
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import System
import ChatHistory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VieSign AI API")

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


@app.on_event("startup")
async def startup_event():
    logger.info("Khởi động server...")
    System.preload_embedding_model()
    logger.info("Hệ thống đã sẵn sàng! (ChromaDB local + OpenAI)")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutdown...")
    await System.cleanup()
    logger.info("Đã giải phóng tài nguyên.")


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

    await ChatHistory.append_messages(session, req.message, response_text)

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
                    if full_response.strip():
                        await ChatHistory.append_messages(session, req.message, full_response)
                    done_event = {
                        'type': 'done',
                        'session_id': session_id,
                        'tools_used': event.get('tools_used', []),
                        'links': event.get('links', []),
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


if __name__ == "__main__":
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=debug_mode,
    )
