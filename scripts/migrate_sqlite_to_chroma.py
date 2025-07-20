"""Migrate legacy messages table into Chroma."""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

from db.chroma import save_message


async def _run(db: Path) -> None:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT thread_id, role, content FROM messages ORDER BY id")
    rows = cur.fetchall()
    for thread_id, role, content in rows:
        await asyncio.to_thread(save_message, thread_id, role, content)
        print(".", end="", flush=True)
    print(f"\nMigrated {len(rows)} rows.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: migrate_sqlite_to_chroma.py <db_path>")
        raise SystemExit(1)
    db_path = Path(sys.argv[1])
    asyncio.run(_run(db_path))


if __name__ == "__main__":  # pragma: no cover
    main()
