"""Regression tests for the Max-Marginal-Relevance (MMR) helper.

These tests focus on the *fallback* implementation that is exercised when
LangChain is **not** available.  The helper purposefully oversamples the initial
candidate set and then removes duplicates to make sure the final result list
contains the desired amount of *unique* texts.

The collection is patched to an **in-memory** instance via the shared
``chroma_fake_embed`` fixture from *test_chroma_save_search.py* in order to
keep test execution fast and hermetic.
"""

from __future__ import annotations

from collections import Counter

import pytest

from db import chroma


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Import the deterministic in-memory Chroma fixture from the neighbouring
# module so we don’t duplicate code.

from tests.chroma.test_chroma_save_search import chroma_fake_embed  # type: ignore F401


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _add(text: str, count: int = 1, thread_id: str = "t-mmr") -> None:
    """Helper that persists *count* copies of *text* to the in-memory store."""

    for _ in range(count):
        chroma.save_message(
            chroma.StoredMsg(thread_id=thread_id, role="user", content=text)
        )


def test_mmr_returns_unique_results(chroma_fake_embed: None) -> None:  # type: ignore[arg-type]
    """The fallback MMR implementation must not yield duplicate texts.

    We create several **identical** messages ("hello world") alongside a few
    other unique strings.  When we later ask for *k* results we expect:

    1. The helper returns exactly *k* items.
    2. Each result text is unique.
    3. At most **one** of them equals the duplicated text.
    """

    # Insert 10 duplicates to exaggerate the imbalance.
    _add("hello world", count=10)

    # Plus some distinct distractions.
    for i in range(5):
        _add(f"distractor-{i}")

    k = 3
    import anyio
    results = anyio.run(chroma.search, None, "hello", k)

    assert len(results) == k, "search() must back-fill to *k* unique items"

    texts = [hit.text for hit in results]
    duplicates = Counter(texts)
    assert duplicates.most_common(1)[0][1] == 1, "duplicate texts should be collapsed"
