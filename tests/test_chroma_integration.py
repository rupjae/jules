from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.schema import AIMessage

from app.routers import chat as chat_router
from app.graphs import main_graph
from db import chroma

app = FastAPI()
app.include_router(chat_router.router)


class EchoLLM:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def invoke(self, messages):
        return AIMessage(content="ok")


def _decode(body: str) -> str:
    return "".join(
        line.split("data: ", 1)[1]
        for line in body.splitlines()
        if line.startswith("data:")
    )


def _setup(monkeypatch, tmp_path):
    monkeypatch.setenv("JULES_CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setattr(main_graph, "ChatOpenAI", EchoLLM)
    import importlib
    import backend.app.checkpointer as cp

    importlib.reload(cp)
    importlib.reload(main_graph)
    app.state.graph = main_graph.build_graph()
    app.state.checkpointer = cp.get_checkpointer()

    class DummyEmbed:
        def __call__(self, texts):
            return [
                [1.0, 0.0] if "apple" in t or "fruit" in t else [0.0, 1.0]
                for t in texts
            ]

    chroma._collection = chroma._client.get_or_create_collection(
        "threads_memory", embedding_function=DummyEmbed()
    )


def test_dual_write(monkeypatch, tmp_path, chroma_service):
    _setup(monkeypatch, tmp_path)
    before_sql = app.state.checkpointer.conn.execute(
        "SELECT COUNT(*) FROM checkpoints"
    ).fetchone()[0]
    before_chroma = chroma.count()
    with TestClient(app) as client:
        r = client.get("/api/chat?message=hi")
        assert r.status_code == 200
        _decode(r.text)
    after_sql = app.state.checkpointer.conn.execute(
        "SELECT COUNT(*) FROM checkpoints"
    ).fetchone()[0]
    after_chroma = chroma.count()
    assert after_sql > before_sql
    assert after_chroma == before_chroma + 2


def test_search_semantic(monkeypatch, tmp_path, chroma_service):
    _setup(monkeypatch, tmp_path)
    tid = "123e4567-e89b-12d3-a456-426614174000"
    with TestClient(app) as client:
        r1 = client.get(f"/api/chat?thread_id={tid}&message=apple")
        _decode(r1.text)
        r2 = client.get(f"/api/chat?thread_id={tid}&message=banana")
        _decode(r2.text)
        sr = client.get(f"/api/chat/search?thread_id={tid}&q=fruit")
        assert sr.status_code == 200
        texts = [res["text"] for res in sr.json()]
        assert any("apple" in t for t in texts)
