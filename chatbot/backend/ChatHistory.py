import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional

DB_PATH = "chat.db"


# ─── Internal ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                title       TEXT    DEFAULT '(chưa có tin nhắn)',
                turn_count  INTEGER DEFAULT 0,
                messages    TEXT    DEFAULT '[]',
                created_at  TEXT,
                updated_at  TEXT
            )
        """)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _short_title(text: str, max_len: int = 50) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


# ─── Core API ────────────────────────────────────────────────────────────────

def create_session() -> dict:
    """Tạo phiên mới, lưu vào SQLite, trả về session dict."""
    _ensure_table()
    now = _now()
    session = {
        "session_id": str(uuid.uuid4()),
        "title":      "(chưa có tin nhắn)",
        "turn_count": 0,
        "messages":   [],
        "created_at": now,
        "updated_at": now,
    }
    with _conn() as c:
        c.execute("""
            INSERT INTO sessions (session_id, title, turn_count, messages, created_at, updated_at)
            VALUES (:session_id, :title, :turn_count, :messages, :created_at, :updated_at)
        """, {**session, "messages": json.dumps([], ensure_ascii=False)})
    return session


def load_session(session_id: str) -> Optional[dict]:
    """Load phiên từ DB. Trả về None nếu không tìm thấy."""
    _ensure_table()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def _save_session(session: dict):
    """Cập nhật session trong DB."""
    session["updated_at"] = _now()
    with _conn() as c:
        c.execute("""
            UPDATE sessions
            SET title = :title, turn_count = :turn_count,
                messages = :messages, updated_at = :updated_at
            WHERE session_id = :session_id
        """, {
            **session,
            "messages": json.dumps(session["messages"], ensure_ascii=False)
        })


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["messages"] = json.loads(d["messages"])
    return d


def append_messages(session: dict, user_query: str, bot_response: str) -> dict:
    """Thêm cặp (user, assistant) vào session và lưu DB."""
    now = _now()
    session["messages"].append({"role": "user",      "content": user_query,   "timestamp": now})
    session["messages"].append({"role": "assistant",  "content": bot_response, "timestamp": now})
    session["turn_count"] = len(session["messages"]) // 2

    # Dùng câu hỏi đầu tiên làm tiêu đề
    if session["turn_count"] == 1:
        session["title"] = _short_title(user_query)

    _save_session(session)
    return session


def get_history_list(messages: list, max_turns: int = 3) -> list:
    """
    Trả về list {"role", "content"} của N lượt gần nhất để truyền vào model.
    max_turns=3 → tối đa 6 messages (3 user + 3 assistant).
    """
    recent = messages[-(max_turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


# ─── Session listing ──────────────────────────────────────────────────────────

def list_sessions() -> list[dict]:
    """Danh sách tất cả phiên, mới nhất lên đầu."""
    _ensure_table()
    with _conn() as c:
        rows = c.execute("""
            SELECT session_id, title, turn_count, updated_at
            FROM sessions
            ORDER BY updated_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    """Xoá phiên khỏi DB. Trả về True nếu thành công."""
    _ensure_table()
    with _conn() as c:
        affected = c.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        ).rowcount
    return affected > 0