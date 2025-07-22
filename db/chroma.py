"""Chroma persistence utilities."""

from __future__ import annotations

import logging
import os
import time
from uuid import uuid4

import anyio

from chromadb import HttpClient
from chromadb.config import Settings
import httpx
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction, QueryResult
from typing import Any
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from jules.logging import trace

logger = logging.getLogger(__name__)

_client: ClientAPI | None = None
_collection: Collection | None = None
_embedding: EmbeddingFunction[Any] | None = None


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        host = os.environ.get("CHROMA_HOST", "chroma")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        try:
            timeout_ms = int(os.environ.get("CHROMA_TIMEOUT_MS", "100"))
        except ValueError:
            logger.warning("Invalid CHROMA_TIMEOUT_MS; using default 100 ms")
            timeout_ms = 100
        _client = HttpClient(
            host=host, port=port, settings=Settings(anonymized_telemetry=False)
        )
        try:
            if hasattr(_client, "_server") and hasattr(_client._server, "_session"):
                _client._server._session.timeout = httpx.Timeout(timeout_ms / 1000)
        except Exception:
            logger.warning("Failed to configure Chroma timeout", exc_info=True)
    return _client


def _get_embedding() -> EmbeddingFunction[Any]:
    global _embedding
    if _embedding is None:
        emb_cls = getattr(embedding_functions, "OpenAIEmbeddingFunction")
        _embedding = emb_cls(
            model_name="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    return _embedding


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        emb = _get_embedding()
        client = _get_client()
        _collection = client.get_or_create_collection(
            "threads_memory",
            embedding_function=emb,
        )
    return _collection


class StoredMsg(BaseModel):
    """Chat message persisted to Chroma."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    thread_id: str
    role: str
    content: str
    ts: float = Field(default_factory=time.time)


class SearchHit(BaseModel):
    """Vector search result."""

    text: str
    distance: float = Field(..., description="Cosine distance; lower is more similar")
    similarity: float | None = Field(
        None, ge=0.0, le=1.0, description="Monotonic similarity derived from distance"
    )
    ts: float | None = None
    role: str | None = None


@trace
def save_message(msg: StoredMsg) -> None:
    """Persist a single message."""

    try:
        col = _get_collection()
        col.add(
            ids=[msg.id],
            documents=[msg.content],
            metadatas=[
                {
                    "thread_id": msg.thread_id,
                    "role": msg.role,
                    "ts": msg.ts,
                }
            ],
        )
    except Exception:
        logger.warning("Failed to save message", exc_info=True)


@trace
async def search(
    where: dict[str, str] | None, query: str, k: int = 8
) -> list[SearchHit]:
    """Return the closest messages to *query* filtered by *where*.

    Results include ``similarity`` in addition to raw ``distance``.
    """

    try:
        col = _get_collection()
        try:
            timeout_ms = int(os.environ.get("CHROMA_TIMEOUT_MS", "100"))
        except ValueError:
            logger.warning("Invalid CHROMA_TIMEOUT_MS; using default 100 ms")
            timeout_ms = 100
        with anyio.fail_after(timeout_ms / 1000):

            def _run_query() -> QueryResult:
                kwargs = {"query_texts": [query], "n_results": k}
                if where:
                    kwargs["where"] = where
                return col.query(**kwargs)

            res: QueryResult = await anyio.to_thread.run_sync(_run_query)
    except Exception:
        logger.warning("Chroma search failed", exc_info=True)
        return []

    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    results: list[SearchHit] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i]
        dist = max(dist, 1e-9)
        similarity = 1 / (1 + dist)
        results.append(
            SearchHit(
                text=doc,
                distance=dist,
                similarity=similarity,
                ts=meta.get("ts"),
                role=meta.get("role"),
            )
        )
    return results


__all__ = [
    "StoredMsg",
    "save_message",
    "search",
    "SearchHit",
]
