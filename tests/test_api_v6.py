"""Integration test – ensure /api/chat/v6 endpoint wires **Graph v6** end-to-end."""

from __future__ import annotations


import pytest


# httpx ships with async test client helpers
import httpx


from backend.app.main import app  # FastAPI application under test


@pytest.mark.anyio
async def test_chat_v6_endpoint_contains_background(monkeypatch):
    """POST /api/chat/v6 should succeed and surface retrieval context."""

    # Patch RetrievalAgent to *force* a cheat-sheet so we can deterministically
    # assert the “### Background” marker without hitting Chroma / OpenAI.
    import jules.graph_v6 as graph_mod

    async def fake_decide(self, user_message, *, thread_id=None):  # noqa: D401
        return {"need_search": True, "query": "alpha", "k": 3}

    async def fake_search(self, query: str, k: int = 3, thread_id=None):  # noqa: D401
        class Dummy:
            def __init__(self, text: str):
                self.text = text
                self.similarity = 1.0
                self.role = "assistant"
                self.ts = 0.0

        return [Dummy("doc1"), Dummy("doc2")]  # type: ignore[return-value]

    async def fake_summary(self, results):  # noqa: D401
        return "- bullet a\n- bullet b"

    monkeypatch.setattr(
        graph_mod._retrieval_agent.__class__, "decide", fake_decide, raising=False
    )
    monkeypatch.setattr(graph_mod._search_tool.__class__, "__call__", fake_search, raising=False)
    monkeypatch.setattr(
        graph_mod._retrieval_agent.__class__,
        "cheat_sheet_from_results",
        fake_summary,
        raising=False,
    )

    # Prevent outbound OpenAI call from JulesAgent
    class _StubLLM:
        async def ainvoke(self, _msgs, **_):
            from langchain.schema import AIMessage  # type: ignore

            return AIMessage("stub-reply")  # type: ignore[arg-type]

    graph_mod._jules_agent.llm = _StubLLM()

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        r = await client.post("/api/chat/v6", json={"message": "who is alpha?"})

        assert r.status_code == 200
        data = r.json()

        # Basic shape
        assert "reply" in data

        # Background marker should be included for retrieval-heavy queries
        assert "### Background" in data["reply"]


# ---------------------------------------------------------------------------
# Parametrised test – unknown / missing versions should gracefully fallback to
# v5 (legacy) instead of returning 422.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("version", [None, "v5", "v7"])
async def test_chat_post_version_fallback(monkeypatch, version):
    """POST /api/chat with unknown version should succeed (fallback to v5)."""

    # Stub ChatOpenAI so _llm_node does not hit the network.
    import backend.app.graphs.main_graph as graph_mod

    class _StubLLM:
        def __init__(self, *_, **__):
            pass

        def invoke(self, _msgs):
            from langchain.schema import AIMessage  # type: ignore

            return AIMessage("stub-v5-reply")  # type: ignore[arg-type]

    monkeypatch.setattr(graph_mod, "ChatOpenAI", _StubLLM, raising=True)

    # Patch SqliteSaver async methods used by LangGraph checkpointer to no-ops
    # Replace the default checkpointer with *None* and rebuild the v5 graph so
    # async SqliteSaver calls are entirely skipped.
    # Instead of wrestling with LangGraph's async checkpointer, return a very
    # small stub graph for any *non-v6* request.  This keeps the test focused
    # on router logic rather than graph internals.

    class _StubGraph:
        async def ainvoke(self, _state, _config=None):  # noqa: D401
            return {"reply": "stub-v5-fallback"}

    import backend.app.routers.chat as chat_router

    def _fake_get_graph(version: str | None = None):  # noqa: D401
        if version == "v6":
            return graph_mod.get_graph("v6")
        return _StubGraph()

    monkeypatch.setattr(chat_router, "_graphs", graph_mod, raising=False)
    monkeypatch.setattr(chat_router._graphs, "get_graph", _fake_get_graph, raising=True)

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        payload = {"message": "hello"}
        if version is not None:
            payload["version"] = version

        r = await client.post("/api/chat", json=payload)

        assert r.status_code == 200
        data = r.json()

        # ensure reply present and cheat_sheet only for v5 fallback (none expected)
        assert data["reply"]
        if version == "v7":
            assert data.get("cheat_sheet") is None
