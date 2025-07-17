"""Thread-safe synchronous SQLite checkpoint saver for LangGraph."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

# Location: <repo>/data/checkpoints.sqlite (auto-created)
# We want this to resolve to "<project root>/data" in the Docker image where
# the application code is located at /app/app/.  Using ``parents[2]`` would
# jump to the filesystem root (/) – therefore we only go *one* level up.
#
# /app/app/memory.py  ->  parent      == /app/app
#                         parent.parent == /app
# So the database file becomes /app/data/checkpoints.sqlite which is the path
# mounted by docker-compose.
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Shared-cache mode makes multiple connections cooperate while still allowing
# each thread its own handle.
_URI = f"file:{CHECKPOINT_PATH}?cache=shared"

# Open connection with check_same_thread=False so we can share it across the
# FastAPI worker threads if needed.  LangGraph usage is blocking (we run it in
# a separate thread via run_in_executor), so this is acceptable.
conn = sqlite3.connect(_URI, uri=True, check_same_thread=False)

checkpointer = SqliteSaver(conn)
checkpointer.setup()

__all__ = ["checkpointer"]
