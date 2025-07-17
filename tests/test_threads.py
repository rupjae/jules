from __future__ import annotations

from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.graphs import main_graph


def test_threads(monkeypatch) -> None:
    """Graph should persist messages per thread."""

    def fake_invoke(self, messages):
        return AIMessage(content="pong")

    monkeypatch.setattr(ChatOpenAI, "invoke", fake_invoke)

    main_graph.graph = main_graph.build_graph(MemorySaver())

    thread_id = "t1"
    out1 = main_graph.graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": thread_id}},
    )
    out2 = main_graph.graph.invoke(
        {"messages": [HumanMessage(content="again")]},
        {"configurable": {"thread_id": thread_id}},
    )

    assert len(out2["messages"]) > len(out1["messages"])
