"""Singleton LangGraph checkpointer with persistent SQLite backend."""

from __future__ import annotations

import logging
import sqlite3
import sys
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

    # ---------------------------------------------------------------------
    # Testing safeguard – when the default *production* path would be used
    # inside a pytest session, replace it with an **in-memory** database so
    # that repeated test runs do not keep appending to
    # ``data/checkpoints.sqlite`` which quickly balloons in size and slows
    # down future runs (see issue #??).
    #
    # Integration tests that *need* durable storage (e.g. to verify that two
    # independent graph instances share state) already override the
    # ``JULES_CHECKPOINT_DB`` environment variable to point at a temporary
    # file.  We therefore only switch to the in-memory backend when
    # 1) pytest is active and
    # 2) the default path would have been used.
    # ---------------------------------------------------------------------

    default_path = "data/jules_memory.sqlite3"
    running_under_pytest = "pytest" in sys.modules  # type: ignore[arg-type]

    if running_under_pytest and settings.checkpoint_db == default_path:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        saver = SqliteSaver(conn)  # type: ignore[arg-type]
        logger.info("Using *in-memory* SqliteSaver for tests instead of %s", default_path)
        return saver

    # Persistent on-disk database (default production path)
    db_path = Path(settings.checkpoint_db).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)  # type: ignore[arg-type]
    logger.info("Using persistent SqliteSaver at %s", db_path)
    return saver
