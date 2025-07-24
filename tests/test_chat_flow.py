"""Unit tests for the next-generation retrieval-aware graph."""

import pytest


@pytest.mark.anyio
async def test_no_search_path(monkeypatch):
    """Graph should bypass retrieval when *need_search* returns False."""

    from backend.app.agents import retrieval_agent as ra
    from backend.app.graphs.next_gen import build_graph

    monkeypatch.setattr(ra, "need_search", lambda _: False)

    graph = build_graph()
    state = {"prompt": "hello", "info_packet": None}

    outputs = [o async for o in graph.stream(state)]

    assert any("content" in o for o in outputs)
    assert outputs[-1].get("info_packet") is None


@pytest.mark.anyio
async def test_search_path(monkeypatch):
    """Graph should attach an info-packet when retrieval is forced."""

    from backend.app.agents import retrieval_agent as ra
    from backend.app.graphs.next_gen import build_graph

    monkeypatch.setattr(ra, "need_search", lambda *_: True)
    monkeypatch.setattr(ra, "search_and_summarise", lambda *_: "• fact A\n• fact B")

    graph = build_graph()
    state = {"prompt": "Please cite sources about X", "info_packet": None}

    outputs = [o async for o in graph.stream(state)]

    pkt = outputs[-1]["info_packet"]

    assert pkt.startswith("•")
    assert len(pkt.split()) <= 150

