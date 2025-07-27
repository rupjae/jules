"""End-to-end test for LangGraph persistence.

The test spins up a fresh *SqliteSaver* bound to a **temporary** on-disk
database and *monkey-patches* the construction inside
``backend.app.graphs.next_gen.build_graph`` so the production path
("db/jules_memory.sqlite3") is not touched.

It then invokes the compiled graph **twice** with the **same** ``thread_id``
and asserts that the second call sees the first user message – i.e. the
conversation history has been restored from the checkpoint store.
"""

from __future__ import annotations

import os
import tempfile
from uuid import uuid4
from unittest import mock

import pytest

# The module under test
from backend.app.graphs import next_gen as ng


@pytest.mark.anyio
async def test_sqlite_memory_roundtrip():
    """Graph should persist and reload messages across invocations."""

    # ------------------------------------------------------------------
    # 1. Create isolated *temporary* SQLite file
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memory.sqlite3")

        # Patch ``sqlite3.connect`` **inside** the target module so the saver
        # initialises with an in-test database file instead of the production
        # path under *db/*.
        import sqlite3

        def _patched_sqlite_connect(path, *args, **kwargs):  # noqa: D401 – helper
            # When the graph builder requests a connection for its default
            # path we redirect it to the *temp* file.
            if isinstance(path, str) and path.endswith("jules_memory.sqlite3"):
                path = db_path
            return orig_connect(path, *args, **kwargs)

        orig_connect = sqlite3.connect

        with mock.patch("sqlite3.connect", side_effect=_patched_sqlite_connect):

            # ------------------------------------------------------------------
            # 2. Build the graph & run two turns with a shared thread id
            # ------------------------------------------------------------------
            graph = ng.build_graph()

            thread_id = str(uuid4())

            # Helper to exhaust the async *stream* helper into a list so we
            # obtain the final output dictionary (last element).
            async def _run(prompt: str):
                state = {"prompt": prompt, "info_packet": None}
                cfg = {"configurable": {"thread_id": thread_id}}
                chunks = [c async for c in graph.stream(state, cfg)]
                return chunks[-1]

            # First user turn – any prompt is fine, use a unique marker
            prompt1 = "Hello memory test"
            await _run(prompt1)

            # Second user turn – the graph should now recall *prompt1*
            prompt2 = "What did I just say?"
            second_resp = await _run(prompt2)

            # ------------------------------------------------------------------
            # 3. Assert that the first prompt is embedded in the restored history
            # ------------------------------------------------------------------

            assert prompt1 in str(second_resp), "Conversation history not persisted"
