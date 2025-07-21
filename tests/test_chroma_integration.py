from __future__ import annotations

import anyio
import chromadb
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import chat as chat_router
from db import chroma, sqlite


class DummyEmbed:
    def __call__(self, input: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in input:
            if "apple" in t or "fruit" in t:
                vecs.append([1.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 1.0, 0.0])
        return vecs


@pytest.fixture()
def chroma_ephemeral(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        chroma.embedding_functions,  # type: ignore[attr-defined]
        "OpenAIEmbeddingFunction",
        lambda model_name, api_key: DummyEmbed(),
    )
    monkeypatch.setattr(chroma, "HttpClient", lambda **_: chromadb.EphemeralClient())
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setenv("JULES_CHAT_DB", str(tmp_path / "chat.sqlite"))


async def _sqlite_count() -> int:
    return await sqlite.count()


def test_dual_write(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        before_sql = anyio.run(_sqlite_count)
        before_chroma = chroma._get_collection().count()
        tid = "550e8400-e29b-41d4-a716-446655440000"
        r = client.post(
            "/api/chat/message",
            json={"thread_id": tid, "role": "user", "content": "hi"},
        )
        assert r.status_code == 200
        after_sql = anyio.run(_sqlite_count)
        after_chroma = chroma._get_collection().count()
        assert after_sql == before_sql + 1
        assert after_chroma == before_chroma + 1


def test_search_semantic(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        tid = "550e8400-e29b-41d4-a716-446655440001"
        client.post(
            "/api/chat/message",
            json={"thread_id": tid, "role": "user", "content": "apple"},
        )
        client.post(
            "/api/chat/message",
            json={"thread_id": tid, "role": "user", "content": "fruit"},
        )
        r = client.get("/api/chat/search", params={"thread_id": tid, "query": "fruit"})
        assert r.status_code == 200
        hits = r.json()
        assert any(h["text"] == "apple" for h in hits)


def test_search_chroma_down(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    async def boom(*_: object, **__: object) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(chroma, "search", boom)

    with TestClient(app) as client:
        r = client.get(
            "/api/chat/search",
            params={"thread_id": "550e8400-e29b-41d4-a716-446655440002", "query": "x"},
        )
        assert r.status_code == 503
        assert r.json() == {"detail": "vector search unavailable"}


def test_message_legacy_route(chroma_ephemeral: None) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    with TestClient(app) as client:
        r = client.post(
            "/api/chat/message/legacy",
            params={
                "thread_id": "550e8400-e29b-41d4-a716-446655440003",
                "role": "user",
                "content": "hi",
            },
        )
        assert r.status_code == 200
