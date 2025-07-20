"""Chroma client wrapper for storing and searching chat messages."""
from __future__ import annotations

import datetime as _dt
import logging
import os
import uuid
from typing import Any, Dict, List
import asyncio

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)

_CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
_CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

try:
    _client = chromadb.HttpClient(host=_CHROMA_HOST, port=_CHROMA_PORT)
    _embedding_func = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",
    )
    _collection = _client.get_or_create_collection(
        "threads_memory", embedding_function=_embedding_func
    )
except Exception:  # noqa: BLE001
    logger.exception("Chroma connection failed")
    _client = None
    _collection = None


async def save_message(thread_id: str, role: str, content: str) -> None:
    """Persist a single message to Chroma."""
    if _collection is None:
        logger.warning("Chroma not available; skipping save")
        return
    try:
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        await asyncio.to_thread(
            _collection.add,
            documents=[content],
            ids=[str(uuid.uuid4())],
            metadatas=[{"thread_id": thread_id, "role": role, "ts": now}],
        )
    except Exception:  # noqa: BLE001
        logger.exception("Chroma save failed")


def search(thread_id: str, query: str, k: int = 8) -> List[Dict[str, Any]]:
    """Return top-K semantic matches for a thread."""
    if _collection is None:
        raise RuntimeError("chroma unavailable")
    try:
        res = _collection.query(
            query_texts=[query], n_results=k, where={"thread_id": thread_id}
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("chroma unavailable") from exc
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out: List[Dict[str, Any]] = []
    for text, meta, score in zip(docs, metas, dists):
        out.append(
            {
                "text": text,
                "role": meta.get("role"),
                "ts": meta.get("ts"),
                "score": score,
            }
        )
    return out


def count() -> int:
    """Return number of documents in collection."""
    if _collection is None:
        return 0
    return _collection.count()
