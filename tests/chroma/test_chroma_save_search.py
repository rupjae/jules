from __future__ import annotations

import hashlib
import time
from collections.abc import Generator

import chromadb
import pytest

from db import chroma


@pytest.fixture()
def chroma_fake_embed(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Patch embedding to deterministic vectors and HttpClient."""

    class DummyEmbed:
        def __call__(self, input: list[str]) -> list[list[float]]:
            vecs: list[list[float]] = []
            for t in input:
                h = hashlib.sha256(t.encode()).hexdigest()
                vec = [int(h[i : i + 8], 16) / 2**32 for i in range(0, 32, 8)]
                vecs.append(vec)
            return vecs

    monkeypatch.setattr(
        chroma.embedding_functions,
        "OpenAIEmbeddingFunction",
        lambda model_name, api_key: DummyEmbed(),
    )
    monkeypatch.setattr(chroma, "HttpClient", lambda **_: chromadb.EphemeralClient())
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    yield


def test_chroma_save_search(chroma_fake_embed: None) -> None:
    msg = chroma.StoredMsg(thread_id="t1", role="user", content="hello", ts=time.time())
    chroma.save_message(msg)
    res = chroma.search("t1", "hello", k=1)
    assert res
    assert res[0]["content"] == "hello"
    assert res[0]["distance"] <= 0.25
