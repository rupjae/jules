from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os

import httpx

from jules.logging import trace

API_BASE = os.getenv("JULES_API_BASE", "http://localhost:8000")


@dataclass
class SearchResult:
    text: str
    similarity: float
    role: str
    ts: float


class ChromaSearchTool:
    """Retrieve the *k* most relevant chat messages via the **/api/chat/search** endpoint.

    Args:
        query: Natural-language search string.
        k: Number of results to return (default 5).
        thread_id: Restrict search to a conversation UUID.
        timeout: Network timeout in seconds (default 10).

    Returns:
        List[SearchResult] sorted by `similarity` desc (0-1).

    Example (agent call JSON):
        {"query": "SIEM licensing", "k": 3}
    """

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    @trace
    async def __call__(
        self, query: str, k: int = 5, thread_id: Optional[str] = None
    ) -> List[SearchResult]:
        params: dict[str, str | int] = {"q": query, "k": k}
        if thread_id:
            params["thread_id"] = thread_id

        async with httpx.AsyncClient(base_url=API_BASE, timeout=self.timeout) as client:
            resp = await client.get("/api/chat/search", params=params)
            resp.raise_for_status()

        hits = resp.json()
        return [SearchResult(**hit) for hit in hits]
