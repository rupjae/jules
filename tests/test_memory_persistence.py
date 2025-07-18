"""Ensure conversation context survives *across* new graph instances.

The test simulates a container restart by creating two independent graph
objects that share the same SQLite checkpoint database on disk.  Without the
persistent :pyfunc:`backend.app.checkpointer.get_checkpointer` implementation the
second run would forget the initial user message.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List

from langchain.schema import AIMessage, HumanMessage


def test_thread_memory_survives_restart(tmp_path: Path, monkeypatch) -> None:  # noqa: D401
    """Two separate graph instances share conversation history via checkpoints."""

    # ----------------------------------------------------------------------------
    # Point the backend to a temp SQLite file (or intentionally disable it to
    # test the in-memory fallback).  The environment variable must be set *before*
    # the first import of backend modules that read the setting.
    # ----------------------------------------------------------------------------

    db_file = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("JULES_CHECKPOINT_DB", str(db_file))

    # ----------------------------------------------------------------------------
    # Import *fresh* copies of the graph builder after the env var update.
    # ----------------------------------------------------------------------------
    import backend.app.graphs.main_graph as mg

    # Clear the cached checkpointer to simulate a *new* process.  We reload the
    # module so that the singleton gets re-created from disk instead of reusing
    # the in-memory instance created above.
    import backend.app.checkpointer as cp  # reload to reset singleton
    importlib.reload(cp)

    importlib.reload(mg)

    class EchoLLM:  # deterministic stub
        def __init__(self, *_: object, **__: object):
            pass

        def invoke(self, messages):
            joined = "|".join(m.content for m in messages)
            return AIMessage(content=joined)

    monkeypatch.setattr(mg, "ChatOpenAI", EchoLLM)

    # First invocation – creates checkpoint ------------------------------------
    graph1 = mg.build_graph()
    tid = "thread-42"
    first_prompt = "hi there"

    for step in graph1.stream({"messages": [HumanMessage(content=first_prompt)]}, {"configurable": {"thread_id": tid}}):
        pass  # we don't need the reply yet

    # Second graph simulates *new* process --------------------------------------
    importlib.reload(mg)

    monkeypatch.setattr(mg, "ChatOpenAI", EchoLLM)
    graph2 = mg.build_graph()

    latest = ""
    second_prompt = "remember me?"
    for step in graph2.stream({"messages": [HumanMessage(content=second_prompt)]}, {"configurable": {"thread_id": tid}}):
        msgs: List | None = step.get("llm", {}).get("messages")
        if msgs:
            latest = msgs[-1].content

    # The assistant should have access to the *first* user message.
    assert first_prompt in latest
