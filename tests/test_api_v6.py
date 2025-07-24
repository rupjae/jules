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
