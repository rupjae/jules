"""SQLite-based graph checkpointing helpers."""

from __future__ import annotations

from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver


saver_path = Path("/app/data/checkpoints.sqlite")
saver_path.parent.mkdir(parents=True, exist_ok=True)
saver_cm = SqliteSaver.from_conn_string(str(saver_path))
checkpointer = saver_cm.__enter__()
checkpointer.setup()
