"""Singleton LangGraph checkpointer with optional persistent SQLite backend."""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – absent in minimal envs
    SqliteSaver = None  # type: ignore[assignment]

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback minimal stub

    class MemorySaver:  # type: ignore[override]
        """Tiny in-memory stand-in exposing the methods used by the backend."""

        def __init__(self) -> None:  # noqa: D401
            self._store: dict[str, dict[str, Any]] = {}
            self._writes: list[dict[str, Any]] = []

        def get_latest_checkpoint(self, thread_id: str):  # noqa: D401
            return self._store.get(thread_id)

        def put_checkpoint(self, thread_id: str, state: dict[str, Any]):  # noqa: D401
            self._store[thread_id] = state
            self._writes.append({"thread_id": thread_id, "state": state.copy()})

        def put_writes(self, *, task_path: str | None = None):  # noqa: D401, ANN001
            return self._writes

        def get_next_version(self, thread_id: str):  # noqa: D401
            versions = self._store.setdefault(thread_id, {}).setdefault("_v", [])
            ver = len(versions) + 1
            versions.append(ver)
            return ver

        def get_tuple(self, *args, **kwargs):  # noqa: D401, ANN001
            class _T:  # noqa: D401 – helper
                def __init__(self, checkpoint):
                    self.checkpoint = checkpoint

            thread_id = None
            if args:
                cfg = args[0]
                thread_id = getattr(cfg, "thread_id", None) or getattr(cfg, "path", None)
            state = self.get_latest_checkpoint(thread_id) if thread_id else None
            return _T(state)  # type: ignore[return-value]

# ---------------------------------------------------------------------------

# Local import – works both when the *app* package is top-level (Docker image)
# and when we run from the repo root where ``backend`` is the package root.
from .config import get_settings  # noqa: E402 – after stubs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer():  # noqa: D401
    """Return a singleton checkpointer (SQLite when available)."""

    settings = get_settings()
    db_path = Path(settings.checkpoint_db).expanduser()

    if SqliteSaver is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)  # type: ignore[arg-type]
        logger.info("Using persistent SqliteSaver at %s", db_path)
    else:
        saver = MemorySaver()  # type: ignore[call-arg]
        logger.warning("langgraph-checkpoint-sqlite not installed – in-memory saver active")

    return saver
