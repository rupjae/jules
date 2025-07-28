"""Unit tests for *RetrievalResult* contract and size guarantees.*"""

from dataclasses import dataclass  # noqa: F401 – fixture import


import pytest


@pytest.mark.anyio
async def test_retrieval_result_contract(monkeypatch):
    """The wrapper should expose need_search + info_packet and respect token cap."""

    from backend.app.agents import retrieval_agent as ra

    # Force *need_search* decision to True so the summary path is executed.
    monkeypatch.setattr(ra, "need_search", lambda *_: True)

    # Fake chroma hits – real DB access is not required for this unit test.
    hits = [
        "Passage one about the topic.",
        "Second excerpt with additional details.",
    ]

    async def _fake_chroma(prompt: str, k: int | None = None):  # noqa: D401
        return hits

    monkeypatch.setattr(ra, "chroma_search", _fake_chroma)

    # Skip OpenAI call by pretending the client is unavailable – the internal
    # fallback joins the passages locally.
    monkeypatch.setattr(ra, "_get_client", lambda: None)

    result = await ra.search_and_summarise("dummy prompt")

    assert result.need_search is True
    assert result.info_packet is not None and result.info_packet.startswith("•")

    # Rough token upper-bound: words ≤ 150 (fallback uses 1 word ≈ 1 token)
    assert len(result.info_packet.split()) <= 150
