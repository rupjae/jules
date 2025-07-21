from __future__ import annotations

import anyio
import chromadb
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import chat as chat_router
from app.graphs import main_graph
from db import chroma, sqlite


def _decode(body: str) -> str:
    return "".join(
        line.split("data: ", 1)[1]
        for line in body.splitlines()
        if line.startswith("data:")
    )


class EchoLLM:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def invoke(self, messages):
        return main_graph.AIMessage(content="pong")


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


def test_stream_persists(
    chroma_ephemeral: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = FastAPI()
    app.include_router(chat_router.router)

    monkeypatch.setattr(main_graph, "ChatOpenAI", EchoLLM)
    app.state.graph = main_graph.build_graph()

    with TestClient(app) as client:
        before_sql = anyio.run(sqlite.count)
        before_chroma = chroma._get_collection().count()

        tid = "550e8400-e29b-41d4-a716-446655440010"
        r = client.get(f"/api/chat?thread_id={tid}&message=hi")
        assert r.status_code == 200
        _decode(r.text)

        after_sql = anyio.run(sqlite.count)
        after_chroma = chroma._get_collection().count()

        assert after_sql == before_sql + 2
        assert after_chroma == before_chroma + 2
