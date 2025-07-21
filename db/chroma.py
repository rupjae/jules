"""Chroma persistence utilities."""

from __future__ import annotations

import logging
import os
import time
from uuid import uuid4

from chromadb import HttpClient
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction, QueryResult
from typing import Any
from chromadb.utils import embedding_functions
from typing import TypedDict, cast
from pydantic import BaseModel, Field

from jules.logging import trace

logger = logging.getLogger(__name__)

_client: ClientAPI | None = None
_collection: Collection | None = None
_embedding: EmbeddingFunction[Any] | None = None


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        _client = HttpClient(host=host, port=port)
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


class SearchEntry(TypedDict, total=False):
    """Result item returned from :func:`search`."""

    id: str
    content: str
    score: float
    thread_id: str
    role: str
    ts: float


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
def search(thread_id: str, query: str, k: int = 8) -> list[SearchEntry]:
    """Return the closest messages for *thread_id* to *query*."""

    try:
        col = _get_collection()
        res: QueryResult = col.query(
            query_texts=[query],
            n_results=k,
            where={"thread_id": thread_id},
        )
    except Exception:
        logger.warning("Chroma search failed", exc_info=True)
        return []

    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    results: list[SearchEntry] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        entry: SearchEntry = {
            "id": ids[i],
            "content": doc,
            "score": dists[i],
        }
        entry.update(cast(SearchEntry, meta))
        results.append(entry)
    return results


__all__ = [
    "StoredMsg",
    "save_message",
    "search",
]
