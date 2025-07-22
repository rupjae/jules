from __future__ import annotations

import httpx
import pytest
import respx
import sys
import types

from langgraph.prebuilt.tool_node import ToolNode


class DummyLLM:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = outputs

    async def __call__(self, _state: dict, *a: object, **k: object) -> dict:
        return self.outputs.pop(0)


class FakeLLMNode(DummyLLM):
    def __init__(self, *a, **k):
        super().__init__([])

    async def __call__(self, _state: dict, *_: object, **__: object) -> dict:
        return self.outputs.pop(0)


@pytest.mark.anyio("asyncio")
async def test_no_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_nodes = types.SimpleNamespace(LLMNode=FakeLLMNode, ToolNode=ToolNode)
    monkeypatch.setitem(sys.modules, "langgraph.nodes", dummy_nodes)
    from src.jules.graph import main_v6

    retrieval = DummyLLM([{"content": "NO_SEARCH"}])
    jules = DummyLLM([{"assistant_reply": "hi"}])
    monkeypatch.setattr(main_v6, "retrieval_llm", retrieval)
    monkeypatch.setattr(main_v6, "jules_llm", jules)
    graph = main_v6.build_graph()
    steps = []
    async for step in graph.astream({"user_msg": "Hi"}):
        steps.append(step)
    assert steps[0]["retrieval_agent"]["content"] == "NO_SEARCH"
    assert steps[-1]["jules"]["assistant_reply"] == "hi"


DUMMY_DOCS = [
    {"text": "d1", "similarity": 0.9, "role": "assistant", "ts": 1},
    {"text": "d2", "similarity": 0.8, "role": "user", "ts": 2},
    {"text": "d3", "similarity": 0.7, "role": "user", "ts": 3},
]


@pytest.mark.anyio("asyncio")
@respx.mock  # type: ignore[misc]
async def test_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_nodes = types.SimpleNamespace(LLMNode=FakeLLMNode, ToolNode=ToolNode)
    monkeypatch.setitem(sys.modules, "langgraph.nodes", dummy_nodes)
    from src.jules.graph import main_v6

    route = respx.get("http://localhost:8000/api/chat/search").mock(
        return_value=httpx.Response(200, json=DUMMY_DOCS)
    )
    retrieval = DummyLLM(
        [
            {
                "type": "function_call",
                "name": "chroma_search",
                "args": {"query": "What is SIEM?", "k": 5},
            },
            {"info_packet": "INFO_PACKET: summary"},
        ]
    )
    jules = DummyLLM([{"assistant_reply": "The answer"}])
    monkeypatch.setattr(main_v6, "retrieval_llm", retrieval)
    monkeypatch.setattr(main_v6, "jules_llm", jules)
    graph = main_v6.build_graph()
    steps = []
    async for step in graph.astream({"user_msg": "What is SIEM?"}):
        steps.append(step)
    assert steps[0]["retrieval_agent"]["type"] == "function_call"
    assert route.called
    assert "INFO_PACKET" in steps[2]["retrieval_agent"]["info_packet"]
    assert steps[-1]["jules"]["assistant_reply"] == "The answer"
