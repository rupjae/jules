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
from typing import Any, Set
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

# External vector-store helpers
# ---------------------------------------------------------------------------
# We try to import LangChain lazily.  In production it should be installed via
# poetry, but tests don’t depend on it.  When missing we silently fall back to
# the legacy dense-search path.

try:
    from langchain.vectorstores import Chroma as LCChroma  # type: ignore
except ImportError:  # pragma: no cover – langchain optional in minimal installs
    LCChroma = None  # type: ignore

# Local runtime configuration -------------------------------------------------

from app.config import get_settings
from functools import lru_cache
from jules.logging import trace

# Cache the settings instance once at import time – these values are immutable
# for the lifetime of the process and reading from the cached copy avoids the
# relatively expensive environment parsing on every search call.

settings = get_settings()

logger = logging.getLogger(__name__)

_client: ClientAPI | None = None
_collection: Collection | None = None
_embedding: EmbeddingFunction[Any] | None = None


# ---------------------------------------------------------------------------
# Max-Marginal Relevance helper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _mmr_collection_wrapper():
    """Return a **LangChain** Chroma wrapper for the shared collection.

    We construct the wrapper lazily to avoid the import cost when LangChain is
    unavailable (e.g. minimal CI jobs).  The wrapper itself is a very thin
    proxy, so rebuilding it for each search call is cheap (<1 ms).
    """

    if LCChroma is None:  # LangChain missing – caller must fall back.
        raise RuntimeError("langchain unavailable")

    # `langchain.vectorstores.Chroma` signature changed; the constructor now
    # expects *collection_name*|*client* instead of a ready `Collection`.
    # Re-create a wrapper that points at the already-initialised collection to
    # avoid maintaining separate storage.

    col = _get_collection()

    try:
        return LCChroma(collection=col, embedding_function=_get_embedding())  # type: ignore[arg-type]
    except TypeError:
        # Fall back to newer signature – supply client + existing name.
        return LCChroma(
            client=_get_client(),
            collection_name=col.name,
            embedding_function=_get_embedding(),
        )


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        host = os.environ.get("CHROMA_HOST", "chroma")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        # The initial embedding call can easily take >1 s when the model (or
        # an external dependency such as a SentenceTransformer ONNX weight)
        # needs to be downloaded.  A very low default led to frequent
        # *ReadTimeout* errors during the first `save_message()` call after
        # startup.  We therefore adopt a more forgiving **5 s** default while
        # still allowing runtime override via *CHROMA_TIMEOUT_MS*.

        try:
            timeout_ms = int(os.environ.get("CHROMA_TIMEOUT_MS", "5000"))
        except ValueError:
            logger.warning("Invalid CHROMA_TIMEOUT_MS; using default 5000 ms")
            timeout_ms = 5000
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
        # Prefer **OpenAI** when an API key is available; otherwise fall back
        # to the lightweight local MiniLM model to avoid runtime failures and
        # large vector dimensionality mismatches (OpenAI = 3072 dims vs MiniLM
        # = 384).  The conditional keeps the embedding dimensionality stable
        # throughout the process lifetime which is crucial because Chroma
        # permanently binds the dimension at collection-creation time.

        # Default strategy --------------------------------------------------
        # 1. If the operator explicitly requests local embeddings via
        #    FORCE_LOCAL_EMBEDDINGS=true → MiniLM.
        # 2. Else, try the OpenAI path.  This is cheap in unit tests where the
        #    class is monkey-patched to a dummy embedder and avoids pulling
        #    the 79 MB MiniLM model during CI.
        # 3. If OpenAI instantiation fails (e.g. missing key), silently fall
        #    back to MiniLM.

        if os.getenv("FORCE_LOCAL_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}:
            emb_cls = getattr(
                embedding_functions, "SentenceTransformerEmbeddingFunction"
            )
            _embedding = emb_cls(model_name="all-MiniLM-L6-v2")
        else:
            try:
                emb_cls = getattr(embedding_functions, "OpenAIEmbeddingFunction")
                _embedding = emb_cls(
                    model_name="text-embedding-3-large",
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                )
            except Exception:
                emb_cls = getattr(
                    embedding_functions, "SentenceTransformerEmbeddingFunction"
                )
                _embedding = emb_cls(model_name="all-MiniLM-L6-v2")
    return _embedding


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        # Lazily resolve the shared collection.  We try **get** first to avoid
        # a round-trip that *may* trigger a 409 Conflict on some Chroma
        # versions when the collection already exists – see
        # https://github.com/chroma-core/chroma/issues/890.

        emb = _get_embedding()
        client = _get_client()

        try:
            _collection = client.get_collection("threads_memory")

            # Ensure the embedding function is attached – the server does not
            # persist embedding config and the client wrapper needs it for
            # all similarity operations.
            if getattr(_collection, "_embedding_function", None) is None:
                _collection._embedding_function = emb  # type: ignore[attr-defined]

        except Exception:
            # Collection missing – create it now.  Should a race lead to a
            # *UniqueConstraintError* we fall back to fetching the existing
            # one which is guaranteed to succeed on the second attempt.
            try:
                _collection = client.create_collection(
                    "threads_memory", embedding_function=emb
                )
            except Exception:
                _collection = client.get_collection("threads_memory")
                if getattr(_collection, "_embedding_function", None) is None:
                    _collection._embedding_function = emb  # type: ignore[attr-defined]

    return _collection


class StoredMsg(BaseModel):
    """Chat message persisted to Chroma."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    thread_id: str
    role: str
    content: str
    ts: float = Field(default_factory=time.time)


class SearchHit(BaseModel):
    """Vector search result without raw distance."""

    text: str
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Monotonic similarity derived from distance"
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
    except Exception as exc:
        # Several recoverable failure modes require a one-time recreation of
        # the collection – dimension mismatch (when it was previously created
        # with a different embedding size) or an outright *InvalidCollection*
        # after an unexpected server restart.

        err_msg = str(exc)
        if (
            "does not match collection dimensionality" in err_msg
            or "does not exist" in err_msg
        ):
            try:
                client = _get_client()
                client.delete_collection("threads_memory")
            except Exception:
                logger.warning("Failed to delete mismatching collection", exc_info=True)

            # Reset cached handle and try once more.
            global _collection
            _collection = None
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
                return
            except Exception:  # pragma: no cover – second failure is fatal
                logger.warning("Second attempt to save message failed", exc_info=True)

        logger.warning("Failed to save message", exc_info=True)


@trace
async def search(
    where: dict[str, str] | None, query: str, k: int | None = None
) -> list[SearchHit]:
    """Return *top-k* semantically similar messages to *query* using a simple
    Max-Marginal-Relevance style **de-duplication** strategy.

    The function intentionally oversamples the initial candidate set and then
    filters it down to *k* unique texts to approximate the effect of true MMR
    without adding a heavyweight dependency on LangChain in the hot path.
    """

    top_k = k or settings.SEARCH_TOP_K
    oversample = settings.SEARCH_MMR_OVERSAMPLE

    if top_k <= 0:
        return []

    # ------------------------------------------------------------------
    # Preferred path – use LangChain’s built-in MMR which balances
    # relevance vs novelty and already de-duplicates.
    # ------------------------------------------------------------------

    if LCChroma is not None:

        def _run_mmr() -> list[Any]:  # type: ignore[valid-type]
            store = _mmr_collection_wrapper()
            fetch_k = top_k * oversample
            # Chroma’s LC wrapper exposes .max_marginal_relevance_search()
            return store.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=fetch_k,
                filter=where,
                lambda_mult=settings.SEARCH_MMR_LAMBDA,
            )

        try:
            docs = await anyio.to_thread.run_sync(_run_mmr)
        except Exception:
            logger.warning("Chroma MMR search failed – falling back", exc_info=True)
            docs = []

        if docs:
            results: list[SearchHit] = []
            for d in docs:
                meta = d.metadata or {}
                sim = (
                    meta.get("relevance_score")
                    or meta.get("similarity")
                    or meta.get("score")
                    or 1.0
                )
                results.append(
                    SearchHit(
                        text=d.page_content,
                        similarity=float(sim),
                        ts=meta.get("ts"),
                        role=meta.get("role"),
                    )
                )
            return results

    # ------------------------------------------------------------------
    # Fallback – legacy dense search with manual uniqueness filter.
    # ------------------------------------------------------------------

    try:
        col = _get_collection()
        timeout_ms = int(os.environ.get("CHROMA_TIMEOUT_MS", "100"))
    except ValueError:
        logger.warning("Invalid CHROMA_TIMEOUT_MS; using default 100 ms")
        timeout_ms = 100

    try:
        with anyio.fail_after(timeout_ms / 1000):

            def _run_query() -> QueryResult:
                n_results = top_k * oversample
                kwargs: dict[str, Any] = {
                    "query_texts": [query],
                    "n_results": n_results,
                }
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

    seen: Set[str] = set()
    results: list[SearchHit] = []
    for i, doc in enumerate(docs):
        if doc in seen and len(results) + (len(docs) - i - 1) >= top_k:
            continue
        seen.add(doc)

        raw_meta = metas[i] if i < len(metas) else None
        meta = raw_meta or {}
        dist = dists[i]
        dist = max(dist, 1e-9)
        similarity = 1 / (1 + dist)

        results.append(
            SearchHit(
                text=doc,
                similarity=similarity,
                ts=meta.get("ts"),
                role=meta.get("role"),
            )
        )
        if len(results) >= top_k:
            break

    return results


__all__ = [
    "StoredMsg",
    "save_message",
    "search",
    "SearchHit",
]
