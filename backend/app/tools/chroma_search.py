"""Thin async wrapper around the project-wide Chroma search helper."""

from __future__ import annotations

from typing import Sequence

from db.chroma import search as _chroma_search


async def chroma_search(query: str, k: int) -> Sequence[str]:
    """Return *k* texts that match *query* ordered by similarity."""

    hits = await _chroma_search(None, query, k)
    # The SearchHit model exposes ``text``.  We only need the plain strings.
    return [h.text for h in hits if h.text]


__all__ = ["chroma_search"]

