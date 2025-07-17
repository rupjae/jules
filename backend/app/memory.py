"""SQLite-based graph checkpointing helpers."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "data" / "checkpoints.sqlite"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()
