from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import aiosqlite  # type: ignore[import-not-found]
from pydantic import BaseModel

from jules.logging import trace

_db: Optional[aiosqlite.Connection] = None


class ChatMessage(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    ts: float


async def _get_db() -> aiosqlite.Connection:
    global _db
    if _db is None:
        path = Path(os.environ.get("JULES_CHAT_DB", "data/chat.sqlite")).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(path)
        await _db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                role TEXT,
                content TEXT,
                ts REAL
            )
            """
        )
        await _db.commit()
    return _db


@trace
async def insert(msg: ChatMessage) -> None:
    db = await _get_db()
    await db.execute(
        "INSERT INTO messages (id, thread_id, role, content, ts) VALUES (?, ?, ?, ?, ?)",
        (msg.id, msg.thread_id, msg.role, msg.content, msg.ts),
    )
    await db.commit()


@trace
async def count() -> int:
    db = await _get_db()
    async with db.execute("SELECT COUNT(*) FROM messages") as cur:
        row = await cur.fetchone()
    return int(row[0])
