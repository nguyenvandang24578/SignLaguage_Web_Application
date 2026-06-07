import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import System
import ChatHistory
import logging
from typing import Optional

# Cấu hình log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(title="VieSign AI API")

# Cấu hình CORS để web HTML (chạy ở port khác) có thể gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên sửa thành domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến toàn cục lưu trữ trạng thái hệ thống
app.state.graph = None

@app.on_event("startup")
async def startup_event():
    logger.info("Đang khởi tạo hệ thống RAG và Embedding Model...")
    System.preload_embedding_model()
    app.state.graph = System.build_graph()
    logger.info("Hệ thống đã sẵn sàng!")

# Định nghĩa các Request/Response Model
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Truyền lên nếu muốn chat tiếp phiên cũ, bỏ trống để tạo mới

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not app.state.graph:
        raise HTTPException(status_code=500, detail="Hệ thống chưa sẵn sàng.")
    
    # 1. Quản lý Session
    if req.session_id:
        session = ChatHistory.load_session(req.session_id)
        if not session:
            # Nếu truyền ID sai/cũ, tự tạo mới
            session = ChatHistory.create_session()
    else:
        session = ChatHistory.create_session()
        
    session_id = session["session_id"]
    
    # 2. Lấy lịch sử hội thoại thuần
    conversation_history = ChatHistory.get_history_list(session["messages"])
    
    try:
        # 3. Chạy qua Agent (LangGraph)
        logger.info(f"Đang xử lý câu hỏi: {req.message} (Session: {session_id})")
        result = System.run_query(
            query=req.message,
            graph=app.state.graph,
            conversation_history=conversation_history
        )
        response_text = result.get("answer", "")
        tools_used = result.get("tools_used", [])

        # Nếu model trả rỗng (hết quota), dùng thông báo chuẩn
        QUOTA_MSG = "⚠️ Xin lỗi, hệ thống AI tạm thời không khả dụng (hết quota). Vui lòng thử lại sau."
        if not response_text or not response_text.strip():
            response_text = QUOTA_MSG

        # 4. Lưu lại lịch sử (lưu đúng nội dung trả về, kể cả thông báo quota)
        ChatHistory.append_messages(session, req.message, response_text)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            tools_used=tools_used,
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history_list():
    """API lấy danh sách lịch sử để hiển thị trên Sidebar của Web"""
    sessions = ChatHistory.list_sessions()
    return {"sessions": sessions}

@app.get("/api/history/{session_id}")
async def get_history_detail(session_id: str):
    """API lấy chi tiết tin nhắn của một phiên cụ thể"""
    session = ChatHistory.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return {"session": session}

@app.delete("/api/history/{session_id}")
async def delete_history(session_id: str):
    """API xoá một phiên chat"""
    success = ChatHistory.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return {"message": "Đã xoá phiên thành công.", "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)