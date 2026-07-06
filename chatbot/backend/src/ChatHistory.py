import aiosqlite
import json
import uuid
import os
import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.getenv("DB_PATH", os.path.join(BACKEND_ROOT, "chat_history", "chat.db"))

_pool_lock = asyncio.Lock()
_pool = None


async def _get_pool():
    global _pool
    async with _pool_lock:
        if _pool is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            _pool = await aiosqlite.connect(DB_PATH)
            _pool.row_factory = aiosqlite.Row
            await _pool.execute("PRAGMA journal_mode=WAL")
            await _pool.execute("PRAGMA synchronous=NORMAL")
            await _pool.execute("PRAGMA cache_size=-32768")
            await _pool.execute("PRAGMA temp_store=MEMORY")
            await _pool.execute("PRAGMA mmap_size=268435456")
            await _pool.commit()
        return _pool


async def _ensure_table():
    pool = await _get_pool()
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT    DEFAULT '(chưa có tin nhắn)',
            turn_count  INTEGER DEFAULT 0,
            messages    TEXT    DEFAULT '[]',
            created_at  TEXT,
            updated_at  TEXT
        )
    """)
    await pool.commit()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _short_title(text: str, max_len: int = 50) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def _row_to_dict(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["messages"] = json.loads(d["messages"])
    return d


async def create_session() -> dict:
    await _ensure_table()
    now = _now()
    session = {
        "session_id": str(uuid.uuid4()),
        "title":      "(chưa có tin nhắn)",
        "turn_count": 0,
        "messages":   [],
        "created_at": now,
        "updated_at": now,
    }
    pool = await _get_pool()
    await pool.execute("""
        INSERT INTO sessions (session_id, title, turn_count, messages, created_at, updated_at)
        VALUES (:session_id, :title, :turn_count, :messages, :created_at, :updated_at)
    """, {**session, "messages": json.dumps([], ensure_ascii=False)})
    await pool.commit()
    return session


async def load_session(session_id: str) -> Optional[dict]:
    await _ensure_table()
    pool = await _get_pool()
    async with pool.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ) as cursor:
        row = await cursor.fetchone()
    if not row:
        return None
    return _row_to_dict(row)


async def _save_session(session: dict):
    session["updated_at"] = _now()
    pool = await _get_pool()
    await pool.execute("""
        UPDATE sessions
        SET title = :title, turn_count = :turn_count,
            messages = :messages, updated_at = :updated_at
        WHERE session_id = :session_id
    """, {
        **session,
        "messages": json.dumps(session["messages"], ensure_ascii=False)
    })
    await pool.commit()


async def append_messages(session: dict, user_query: str, bot_response: str,
                          tools_used: list | None = None, links: list | None = None) -> dict:
    now = _now()
    session["messages"].append({"role": "user",      "content": user_query,   "timestamp": now})
    session["messages"].append({
        "role": "assistant",
        "content": bot_response,
        "timestamp": now,
        "tools_used": tools_used or [],
        "links": links or [],
    })
    session["turn_count"] = len(session["messages"]) // 2

    if session["turn_count"] == 1:
        session["title"] = _short_title(user_query)

    await _save_session(session)
    return session


def get_history_list(messages: list, max_turns: int = 3) -> list:
    recent = messages[-(max_turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


async def list_sessions() -> list[dict]:
    await _ensure_table()
    pool = await _get_pool()
    async with pool.execute("""
        SELECT session_id, title, turn_count, updated_at
        FROM sessions
        ORDER BY updated_at DESC
    """) as cursor:
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def delete_session(session_id: str) -> bool:
    await _ensure_table()
    pool = await _get_pool()
    cursor = await pool.execute(
        "DELETE FROM sessions WHERE session_id = ?", (session_id,)
    )
    await pool.commit()
    return cursor.rowcount > 0


async def close():
    """Đóng kết nối SQLite — gọi khi server shutdown."""
    global _pool
    async with _pool_lock:
        if _pool is not None:
            try:
                await _pool.close()
                logger.info("SQLite connection closed.")
            except Exception as e:
                logger.warning(f"SQLite close error: {e}")
            _pool = None

