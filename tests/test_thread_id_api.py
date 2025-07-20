from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.schema import AIMessage
from app.routers import chat as chat_router
from app.graphs import main_graph


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
        return AIMessage(content=" ".join(m.content for m in messages))


app = FastAPI()
app.include_router(chat_router.router)


def _setup(monkeypatch):
    monkeypatch.setattr(main_graph, "ChatOpenAI", EchoLLM)
    app.state.graph = main_graph.build_graph()


def test_query_param_persists_thread(monkeypatch) -> None:
    _setup(monkeypatch)
    tid = "123e4567-e89b-12d3-a456-426614174000"
    with TestClient(app) as client:
        r1 = client.get(f"/api/chat?thread_id={tid}&message=hello")
        assert r1.status_code == 200
        _decode(r1.text)
        r2 = client.get(f"/api/chat?thread_id={tid}&message=again")
        body = _decode(r2.text)
        assert "hello" in body


def test_invalid_thread_id(monkeypatch) -> None:
    _setup(monkeypatch)
    with TestClient(app) as client:
        r = client.get("/api/chat?thread_id=bogus&message=hi")
        assert r.status_code == 400
