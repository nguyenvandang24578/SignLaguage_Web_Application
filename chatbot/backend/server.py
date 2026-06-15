import uvicorn
import html
import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import System
import ChatHistory
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Cấu hình log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Request size limit middleware (1MB)
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 1_000_000):
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return Response(
                content='{"detail": "Request body too large. Max size: 1MB"}',
                status_code=413,
                media_type="application/json"
            )
        return await call_next(request)


# =============================================================================
# TASK TRACKING SYSTEM - Background task management
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ChatTask:
    task_id: str
    session_id: str
    user_message: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""

# In-memory task store (use Redis in production for multi-worker)
_task_store: Dict[str, ChatTask] = {}
_task_store_lock = asyncio.Lock()

async def create_task(session_id: str, user_message: str) -> ChatTask:
    """Create a new chat task."""
    task = ChatTask(
        task_id=str(uuid.uuid4()),
        session_id=session_id,
        user_message=user_message
    )
    async with _task_store_lock:
        _task_store[task.task_id] = task
    return task

async def get_task(task_id: str) -> Optional[ChatTask]:
    """Get task by ID."""
    async with _task_store_lock:
        return _task_store.get(task_id)

async def update_task(task_id: str, **kwargs) -> Optional[ChatTask]:
    """Update task fields."""
    async with _task_store_lock:
        task = _task_store.get(task_id)
        if task:
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
    return task

async def cancel_task(task_id: str) -> Optional[ChatTask]:
    """Cancel a running task."""
    # Set cancellation flag for the System module
    await System.set_cancel_flag(task_id, True)
    
    async with _task_store_lock:
        task = _task_store.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            task.current_step = "Cancelled by user"
            return task
    return None

async def cleanup_old_tasks(max_age_hours: int = 24):
    """Remove completed/failed/cancelled tasks older than max_age_hours."""
    now = datetime.now()
    async with _task_store_lock:
        to_delete = [
            tid for tid, task in _task_store.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            and (now - task.created_at).total_seconds() > max_age_hours * 3600
        ]
        for tid in to_delete:
            del _task_store[tid]


# =============================================================================
# BACKGROUND TASK PROCESSING
# =============================================================================

async def process_chat_task(task_id: str, graph, session: dict, conversation_history: list):
    """Background task to process chat query and save result."""
    task = await get_task(task_id)
    if not task:
        return
    
    await update_task(task_id, status=TaskStatus.RUNNING, started_at=datetime.now(), current_step="Initializing", progress=0.1)
    
    try:
        # Step 1: Run query through agent
        await update_task(task_id, current_step="Processing query", progress=0.3)
        
        # Check for cancellation before running query
        task = await get_task(task_id)
        if task and task.status == TaskStatus.CANCELLED:
            logger.info(f"Task {task_id} was cancelled before query execution")
            return
        
        result = await System.run_query(
            query=task.user_message,
            graph=graph,
            conversation_history=conversation_history,
            task_id=task_id
        )
        
        # Check for cancellation after query completes
        task = await get_task(task_id)
        if task and task.status == TaskStatus.CANCELLED:
            # Save partial result if available
            response_text = result.get("answer", "")
            tools_used = result.get("tools_used", [])
            if response_text and response_text.strip():
                await ChatHistory.append_messages(session, task.user_message, response_text)
                await update_task(
                    task_id,
                    status=TaskStatus.CANCELLED,
                    completed_at=datetime.now(),
                    progress=1.0,
                    current_step="Cancelled - partial response saved",
                    result={
                        "response": response_text,
                        "tools_used": tools_used,
                        "session_id": session["session_id"],
                        "partial": True
                    }
                )
            else:
                await update_task(
                    task_id,
                    status=TaskStatus.CANCELLED,
                    completed_at=datetime.now(),
                    current_step="Cancelled - no response generated"
                )
            logger.info(f"Task {task_id} cancelled after query execution")
            await System.clear_cancel_flag(task_id)
            return
        
        await update_task(task_id, current_step="Saving response", progress=0.8)
        response_text = result.get("answer", "")
        tools_used = result.get("tools_used", [])
        
        # Handle empty response
        QUOTA_MSG = "⚠️ Xin lỗi, hệ thống AI tạm thời không khả dụng (hết quota). Vui lòng thử lại sau."
        if not response_text or not response_text.strip():
            response_text = QUOTA_MSG
        
        # Step 2: Save to session history
        await ChatHistory.append_messages(session, task.user_message, response_text)
        
        # Step 3: Mark completed
        await update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(),
            progress=1.0,
            current_step="Done",
            result={
                "response": response_text,
                "tools_used": tools_used,
                "session_id": session["session_id"]
            }
        )
        logger.info(f"Task {task_id} completed for session {session['session_id']}")
        await System.clear_cancel_flag(task_id)
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        await update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now(),
            error=str(e),
            current_step="Error"
        )
        await System.clear_cancel_flag(task_id)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="VieSign AI API")

# Add request size limit middleware
app.add_middleware(RequestSizeLimitMiddleware, max_size=1_000_000)

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
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

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global state
app.state.graph = None

@app.on_event("startup")
async def startup_event():
    logger.info("Đang khởi tạo hệ thống RAG và Embedding Model...")
    System.preload_embedding_model()
    app.state.graph = System.build_graph()
    logger.info("Hệ thống đã sẵn sàng!")
    
    # Start periodic cleanup
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodic cleanup of old tasks."""
    while True:
        await asyncio.sleep(3600)  # Every hour
        await cleanup_old_tasks(24)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return html.escape(v.strip())

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)

class TaskCreateResponse(BaseModel):
    task_id: str
    session_id: str
    status: TaskStatus
    message: str = "Task created. Poll /api/chat/status/{task_id} for result."

class TaskStatusResponse(BaseModel):
    task_id: str
    session_id: str
    status: TaskStatus
    progress: float
    current_step: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/api/chat", response_model=TaskCreateResponse)
@limiter.limit("30/minute")
async def chat_endpoint(request: Request, req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Start a chat task in the background. Returns immediately with task_id.
    Poll /api/chat/status/{task_id} for progress and result.
    """
    if not app.state.graph:
        raise HTTPException(status_code=500, detail="Hệ thống chưa sẵn sàng.")
    
    # 1. Session management
    if req.session_id:
        session = await ChatHistory.load_session(req.session_id)
        if not session:
            session = await ChatHistory.create_session()
    else:
        session = await ChatHistory.create_session()
    
    session_id = session["session_id"]
    conversation_history = ChatHistory.get_history_list(session["messages"])
    
    # 2. Create task
    task = await create_task(session_id, req.message)
    
    # 3. Start background processing
    background_tasks.add_task(
        process_chat_task,
        task.task_id,
        app.state.graph,
        session,
        conversation_history
    )
    
    logger.info(f"Started task {task.task_id} for session {session_id}")
    
    return TaskCreateResponse(
        task_id=task.task_id,
        session_id=session_id,
        status=task.status
    )


@app.get("/api/chat/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status and result (for polling)."""
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(
        task_id=task.task_id,
        session_id=task.session_id,
        status=task.status,
        progress=task.progress,
        current_step=task.current_step,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at
    )


@app.get("/api/chat/stream/{task_id}")
async def stream_task_status(task_id: str):
    """Server-Sent Events stream for real-time task updates."""
    async def event_generator():
        last_status = None
        while True:
            task = await get_task(task_id)
            if not task:
                yield f"data: {{\"error\": \"Task not found\"}}\n\n"
                break
            
            # Only send update if status changed
            if task.status != last_status:
                import json
                data = {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "progress": task.progress,
                    "current_step": task.current_step,
                    "result": task.result,
                    "error": task.error
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_status = task.status
            
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                break
            
            await asyncio.sleep(0.5)  # Poll every 500ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/chat/cancel/{task_id}")
async def cancel_chat_task(task_id: str):
    """Cancel a running chat task."""
    task = await cancel_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "message": "Task cancelled successfully"
    }


# =============================================================================
# HISTORY ENDPOINTS (unchanged)
# =============================================================================

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


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "active_tasks": len([t for t in _task_store.values() if t.status == TaskStatus.RUNNING]),
        "pending_tasks": len([t for t in _task_store.values() if t.status == TaskStatus.PENDING]),
        "cancelled_tasks": len([t for t in _task_store.values() if t.status == TaskStatus.CANCELLED])
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)