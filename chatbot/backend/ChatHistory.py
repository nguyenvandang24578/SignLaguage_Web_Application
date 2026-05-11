import json
import os
import uuid
from datetime import datetime
from typing import Optional

HISTORY_DIR = "chat_histories"



def _ensure_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _short_title(text: str, max_len: int = 50) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


# ─── Core API ────────────────────────────────────────────────────────────────

def create_session() -> dict:
    """Tạo một phiên mới, trả về session dict rỗng."""
    _ensure_dir()
    session_id = _new_session_id()
    now = _now_iso()
    session = {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "title": "(chưa có tin nhắn)",
        "turn_count": 0,
        "messages": [],
    }
    _save_session(session)
    return session


def load_session(session_id: str) -> Optional[dict]:
    """Load phiên từ file JSON. Trả về None nếu không tìm thấy."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_session(session: dict):
    """Ghi session xuống file JSON."""
    _ensure_dir()
    path = _session_path(session["session_id"])
    session["updated_at"] = _now_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def append_messages(session: dict, user_query: str, bot_response: str) -> dict:
    """
    Thêm một cặp (user, assistant) vào session và lưu xuống file.
    Trả về session đã cập nhật.
    """
    now = _now_iso()

    session["messages"].append({"role": "user",      "content": user_query,    "timestamp": now})
    session["messages"].append({"role": "assistant",  "content": bot_response,  "timestamp": now})
    session["turn_count"] = len(session["messages"]) // 2

    # Dùng câu hỏi đầu tiên làm tiêu đề phiên
    if session["turn_count"] == 1:
        session["title"] = _short_title(user_query)

    _save_session(session)
    return session


def get_history_list(messages: list) -> list:
    """
    Chuyển messages (có timestamp) → list thuần
    {"role", "content"} để truyền vào AgentState.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ─── Session listing & selection UI ─────────────────────────────────────────

def list_sessions() -> list[dict]:
    """
    Trả về danh sách tất cả phiên, sắp xếp mới nhất lên đầu.
    Mỗi phần tử: {"session_id", "title", "turn_count", "updated_at"}
    """
    _ensure_dir()
    sessions = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data["session_id"],
                "title":      data.get("title", "(không có tiêu đề)"),
                "turn_count": data.get("turn_count", 0),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception:
            continue  # Bỏ qua file lỗi

    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def print_session_list(sessions: list[dict]):
    """In danh sách phiên ra terminal."""
    if not sessions:
        print("  (Chưa có phiên nào được lưu)")
        return
    print(f"\n  {'#':<4} {'Thời gian cập nhật':<22} {'Lượt':<6} Tiêu đề")
    print("  " + "-" * 70)
    for i, s in enumerate(sessions, 1):
        dt = s["updated_at"].replace("T", " ")
        print(f"  [{i}]  {dt:<22} {s['turn_count']:<6} {s['title']}")
    print()


def select_or_create_session() -> dict:
    """
    Hiển thị menu chọn phiên ngay khi khởi động.
    Trả về session dict (mới hoặc đã load).
    """
    sessions = list_sessions()

    print("\n" + "=" * 60)
    print("  LỊCH SỬ HỘI THOẠI")
    print("=" * 60)

    if not sessions:
        print("  Chưa có phiên nào. Tạo phiên mới...")
        session = create_session()
        print(f"  ✓ Phiên mới: {session['session_id']}\n")
        return session

    print_session_list(sessions)
    print("  [0] Tạo phiên mới")
    print("  [1–{}] Load phiên cũ".format(len(sessions)))

    while True:
        choice = input("  Chọn: ").strip()

        if choice == "0" or choice == "":
            session = create_session()
            print(f"  ✓ Phiên mới: {session['session_id']}\n")
            return session

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                sid = sessions[idx - 1]["session_id"]
                session = load_session(sid)
                if session:
                    turn = session["turn_count"]
                    print(f"  ✓ Đã load phiên '{session['title']}' ({turn} lượt trước)\n")
                    return session

        print("  Lựa chọn không hợp lệ, nhập lại.")


def delete_session(session_id: str) -> bool:
    """Xoá file JSON của phiên. Trả về True nếu thành công."""
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False