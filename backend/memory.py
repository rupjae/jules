from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer() -> MemorySaver:
    """Return a singleton MemorySaver instance."""
    saver = MemorySaver()
    logger.debug("MemorySaver initialised (singleton).")
    return saver
