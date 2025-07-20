"""Singleton LangGraph checkpointer with persistent SQLite backend."""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path


from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

# ---------------------------------------------------------------------------

# Local import – works both when the *app* package is top-level (Docker image)
# and when we run from the repo root where ``backend`` is the package root.
from .config import get_settings  # noqa: E402 – after stubs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer():  # noqa: D401
    """Return a singleton checkpointer."""

    settings = get_settings()
    db_path = Path(settings.checkpoint_db).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)  # type: ignore[arg-type]
    logger.info("Using persistent SqliteSaver at %s", db_path)
    return saver
