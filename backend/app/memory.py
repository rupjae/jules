"""Postgres-based graph checkpointing helpers."""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver

from .config import get_settings


@lru_cache
def get_checkpointer() -> PostgresSaver:
    """Return a configured PostgresSaver and ensure tables exist."""
    saver_cm = PostgresSaver.from_conn_string(get_settings().postgres_uri)
    saver = saver_cm.__enter__()
    saver.setup()
    return saver
