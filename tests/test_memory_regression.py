from __future__ import annotations

from typing import List

from langchain.schema import AIMessage, HumanMessage
from app.graphs import main_graph


def _collect_content(prompt: str, thread_id: str) -> str:
    latest = ""
    for step in main_graph.graph.stream(
        {"messages": [HumanMessage(content=prompt)]},
        {"configurable": {"thread_id": thread_id}},
    ):
        msgs: List | None = step.get("llm", {}).get("messages")
        if msgs:
            latest = msgs[-1].content
    return latest


def test_memory_regression(monkeypatch) -> None:
    """Assistant should remember earlier turns within the same thread."""

    class FakeLLM:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def invoke(self, messages):
            joined = " ".join(m.content for m in messages)
            return AIMessage(content=joined)

    monkeypatch.setattr(main_graph, "ChatOpenAI", FakeLLM)
    main_graph.graph = main_graph.build_graph()

    tid = "t42"
    first = _collect_content("hi", tid)
    second = _collect_content("again", tid)

    assert first in second
