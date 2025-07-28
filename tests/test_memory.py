"""End-to-end test for LangGraph persistence.

The test spins up a **temporary** on-disk sqlite database, builds the graph
with the *db_url* pointing at that file and runs **two** turns with a shared
``thread_id``.  The second invocation must see the first user message – i.e.
conversation history has been restored from the saver automatically.
"""

from __future__ import annotations

import asyncio  # noqa: F401 – used for type hints / future extensions
import os
import tempfile
from uuid import uuid4

import pytest


@pytest.mark.anyio
async def test_sqlite_memory_roundtrip():
    """Graph should persist and reload messages across invocations."""

    # Import lazily so the module picks up the *db_url* we pass to
    # ``build_graph`` instead of initialising with the production path.
    from backend.app.graphs import next_gen as ng

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memory.sqlite3")
        db_url = db_path

        graph = ng.build_graph(db_url=db_url)

        thread_id = str(uuid4())

        async def _run(prompt: str):
            state = {"prompt": prompt, "info_packet": None}
            cfg = {"configurable": {"thread_id": thread_id}}
            chunks = [c async for c in graph.stream(state, cfg)]
            return chunks[-1]

        prompt1 = "Hello memory test"
        await _run(prompt1)

        prompt2 = "What did I just say?"
        second_resp = await _run(prompt2)

        assert prompt1 in str(second_resp), "Conversation history not persisted"
