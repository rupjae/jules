from __future__ import annotations

from langchain.schema import AIMessage, HumanMessage
from app.graphs import main_graph


def test_threads(monkeypatch) -> None:
    """Graph should persist messages per thread."""

    class FakeLLM:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def invoke(self, messages):
            return AIMessage(content="pong")

    monkeypatch.setenv("USE_IN_MEMORY", "1")
    monkeypatch.setattr(main_graph, "ChatOpenAI", FakeLLM)

    main_graph.graph = main_graph.build_graph()

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
