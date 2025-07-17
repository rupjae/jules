from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    POSTGRES_URI: Optional[str] = None
    USE_IN_MEMORY: bool = False


settings = Settings()


def get_checkpointer():
    """Return a configured LangGraph checkpointer."""
    if settings.USE_IN_MEMORY:
        logger.debug("Using MemorySaver for checkpoints (test mode).")
        return MemorySaver()

    if not settings.POSTGRES_URI:
        raise RuntimeError(
            "POSTGRES_URI env var is required when USE_IN_MEMORY is false."
        )

    logger.debug("Using PostgresSaver for checkpoints: %s", settings.POSTGRES_URI)
    return PostgresSaver.from_conn_string(settings.POSTGRES_URI)
