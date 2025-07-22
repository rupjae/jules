from __future__ import annotations

import chromadb
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import chat as chat_router
from db import chroma


@pytest.fixture()
def chroma_ephemeral(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class DummyEmbed:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in input]

    monkeypatch.setattr(
        chroma.embedding_functions,  # type: ignore[attr-defined]
        "OpenAIEmbeddingFunction",
        lambda model_name, api_key: DummyEmbed(),
    )
    monkeypatch.setattr(chroma, "HttpClient", lambda **_: chromadb.EphemeralClient())
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setenv("JULES_CHAT_DB", str(tmp_path / "chat.sqlite"))


def test_global_search_returns_hits(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        tid1 = "550e8400-e29b-41d4-a716-446655440011"
        tid2 = "550e8400-e29b-41d4-a716-446655440012"
        client.post(
            "/api/chat/message",
            json={"thread_id": tid1, "role": "user", "content": "hello one"},
        )
        client.post(
            "/api/chat/message",
            json={"thread_id": tid2, "role": "user", "content": "hello two"},
        )

        r = client.get("/api/chat/search", params={"query": "hello"})
        assert r.status_code == 200
        assert len(r.json()) >= 2


def test_thread_search_still_isolated(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        tid1 = "550e8400-e29b-41d4-a716-446655440013"
        tid2 = "550e8400-e29b-41d4-a716-446655440014"
        client.post(
            "/api/chat/message",
            json={"thread_id": tid1, "role": "user", "content": "alpha"},
        )
        client.post(
            "/api/chat/message",
            json={"thread_id": tid2, "role": "user", "content": "beta"},
        )

        r = client.get("/api/chat/search", params={"thread_id": tid1, "query": "alpha"})
        assert r.status_code == 200
        hits = r.json()
        assert len(hits) == 1
        assert hits[0]["text"] == "alpha"


def test_similarity_field_present(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        tid = "550e8400-e29b-41d4-a716-446655440015"
        client.post(
            "/api/chat/message",
            json={"thread_id": tid, "role": "user", "content": "hello"},
        )

        r = client.get("/api/chat/search", params={"thread_id": tid, "query": "hello"})
        assert r.status_code == 200
        hits = r.json()
        assert hits
        assert 0.0 <= hits[0]["similarity"] <= 1.0


def test_min_similarity_filters(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        tid = "550e8400-e29b-41d4-a716-446655440016"
        client.post(
            "/api/chat/message",
            json={"thread_id": tid, "role": "user", "content": "hello"},
        )
        client.post(
            "/api/chat/message",
            json={"thread_id": tid, "role": "user", "content": "unrelated"},
        )

        r = client.get(
            "/api/chat/search",
            params={"thread_id": tid, "query": "hello", "min_similarity": 0.9},
        )
        assert r.status_code == 200
        hits = r.json()
        assert len(hits) == 1
        assert hits[0]["text"] == "hello"
