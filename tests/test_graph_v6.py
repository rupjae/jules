"""Unit tests for the Retrieval-aware Graph v6 pipeline."""

from __future__ import annotations

import types

import pytest


import jules.graph_v6 as graph_mod


class DummyResult:  # mimics tools.SearchResult without importing heavy deps
    def __init__(self, text: str) -> None:  # noqa: D401 – simple helper
        self.text = text
        self.similarity = 1.0
        self.role = "assistant"
        self.ts = 0.0


def test_graph_v6_no_search(monkeypatch):
    """When RetrievalAgent says *no search*, graph should skip Chroma."""

    async def fake_decide(self, user_message, *, thread_id=None):  # noqa: D401
        return {"need_search": False}

    monkeypatch.setattr(
        graph_mod._retrieval_agent.__class__, "decide", fake_decide, raising=False
    )

    # Stub out Jules LLM to avoid real OpenAI call
    class _StubLLM:
        async def ainvoke(self, _msgs, **_):
            from langchain.schema import AIMessage  # type: ignore

            return AIMessage("stub-reply")  # type: ignore[arg-type]

    graph_mod._jules_agent.llm = _StubLLM()

    sg = graph_mod.graph
    import asyncio
    out = asyncio.run(sg.ainvoke({"user_message": "hi"}))
    # Graph should contain direct reply from Jules node without cheat-sheet
    assert "reply" in out
    assert out["reply"] == "stub-reply"


def test_graph_v6_search(monkeypatch):
    """Happy path with search and cheat-sheet injection."""

    async def fake_decide(self, user_message, *, thread_id=None):
        return {
            "need_search": True,
            "query": "foo",
            "k": 2,
        }

    async def fake_search(self, query: str, k: int = 5, thread_id=None):  # noqa: D401
        return [DummyResult("doc1"), DummyResult("doc2")]

    async def fake_summary(self, results):
        return "- bullet 1\n- bullet 2"

    monkeypatch.setattr(
        graph_mod._retrieval_agent.__class__, "decide", fake_decide, raising=False
    )
    class _StubSearch:
        async def __call__(self, query: str, k: int = 5, thread_id=None):
            return await fake_search(query, k, thread_id)

    monkeypatch.setattr(graph_mod, "_search_tool", _StubSearch())
    monkeypatch.setattr(
        graph_mod._retrieval_agent.__class__,
        "cheat_sheet_from_results",
        fake_summary,
        raising=False,
    )

    class _StubLLM:
        async def ainvoke(self, _msgs, **_):
            from langchain.schema import AIMessage  # type: ignore

            return AIMessage("stub-reply")  # type: ignore[arg-type]

    graph_mod._jules_agent.llm = _StubLLM()

    sg = graph_mod.graph
    import asyncio
    out = asyncio.run(sg.ainvoke({"user_message": "hi"}))

    assert out["cheat_sheet"] == "- bullet 1\n- bullet 2"
    assert out["reply"] == "stub-reply"
