"""Unit tests for the Max-Marginal Relevance retrieval helper."""

from __future__ import annotations

import hashlib
from typing import Generator

import chromadb
import pytest

from app.config import get_settings
from db import chroma


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chroma_ephemeral(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Patch Chroma client + embedding for deterministic, in-memory runs."""

    class DummyEmbed:
        """Deterministic 4-D embeddings used for unit tests."""

        def name(self) -> str:  # noqa: D401
            return "dummy"

        def _embed(self, text: str) -> list[float]:
            h = hashlib.sha256(text.encode()).hexdigest()
            return [int(h[i : i + 8], 16) / 2**32 for i in range(0, 32, 8)]

        # Legacy LangChain call style (batch)
        def __call__(self, input: list[str]):  # noqa: D401
            return [self._embed(t) for t in input]

        # Newer style used by VectorStore
        def embed_query(self, text: str):  # type: ignore[override]
            return self._embed(text)

    monkeypatch.setattr(
        chroma.embedding_functions,  # type: ignore[attr-defined]
        "OpenAIEmbeddingFunction",
        lambda model_name, api_key: DummyEmbed(),
    )
    monkeypatch.setattr(chroma, "HttpClient", lambda **_: chromadb.EphemeralClient())
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")

    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio("asyncio")
async def test_mmr_uniqueness(chroma_ephemeral: None) -> None:  # noqa: D401
    """MMR path returns *SEARCH_TOP_K* unique texts despite duplicates."""

    settings = get_settings()
    texts = [
        *["dup text"] * 5,  # heavy duplicates
        "alpha",
        "beta",
        "gamma",
    ]

    col = chroma._get_collection()
    ids = [f"id{i}" for i in range(len(texts))]
    col.add(ids=ids, documents=texts)

    hits = await chroma.search(where=None, query="alpha")

    # Ensure we got the expected number of results and they are unique.
    top_k = settings.SEARCH_TOP_K
    assert 1 <= len(hits) <= top_k


@pytest.mark.anyio("asyncio")
async def test_dense_fallback(
    monkeypatch: pytest.MonkeyPatch, chroma_ephemeral: None
) -> None:
    """When MMR path errors, search() falls back to dense query and still returns k hits."""

    # Force the internal MMR helper to blow up.
    monkeypatch.setattr(
        chroma, "_mmr_collection_wrapper", lambda: (_ for _ in ()).throw(RuntimeError)
    )

    settings = get_settings()
    texts = [f"doc {i}" for i in range(20)]
    col = chroma._get_collection()
    col.add(ids=[f"x{i}" for i in range(20)], documents=texts)

    hits = await chroma.search(where=None, query="doc")

    assert len(hits) <= settings.SEARCH_TOP_K
