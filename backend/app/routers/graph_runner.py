"""Helper to stream LangGraph executions as async generator."""

from __future__ import annotations

from typing import AsyncGenerator, Any


async def run_graph(
    graph,
    state: dict[str, Any],
    *,
    thread_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield flattened step outputs from *graph* given *state*.

    The function mirrors the convenience wrapper we add in backend.app.graphs.next_gen
    but remains generic so other graphs can use it.
    """

    # Prefer the custom ``stream`` attribute when available (added by our
    # builder).  Fall back to the library-provided ``astream``.
    streamer = getattr(graph, "stream", None) or graph.astream  # type: ignore[attr-defined]

    if thread_id is not None:
        cfg = {"configurable": {"thread_id": thread_id}}
        async for step in streamer(state, config=cfg):  # type: ignore[arg-type]
            yield step
        return

    async for step in streamer(state):  # type: ignore[arg-type]
        yield step
