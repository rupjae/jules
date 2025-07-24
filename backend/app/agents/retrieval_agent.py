"""Lightweight RAG pre-processor that decides whether external context is
required and, if so, fetches and summarises the relevant passages.

This module purposefully contains **no blocking I/O** in the hot path – Chroma
queries are awaited directly, and the costly large-model summarisation is left
to the OpenAI API.  All functions degrade gracefully in test environments
where the OpenAI key is missing or outbound access is disabled.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..config_agents import get_cfg
from ..tools.chroma_search import chroma_search
from jules.logging import trace

# ---------------------------------------------------------------------------
# Optional runtime dependencies
# ---------------------------------------------------------------------------

try:
    import tiktoken  # type: ignore

    _tiktoken_available = True
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore
    _tiktoken_available = False

try:
    from openai import AsyncOpenAI  # type: ignore

    _openai_available = True
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore
    _openai_available = False


logger = logging.getLogger(__name__)
cfg = get_cfg().retrieval

# Encoding determined once per process; falls back to cl100k_base when
# *tiktoken* is unavailable or the model is unknown.
if _tiktoken_available:
    try:
        ENC = tiktoken.encoding_for_model(cfg.model)
    except Exception:  # pragma: no cover – unknown model id
        ENC = tiktoken.get_encoding("cl100k_base")
else:  # pragma: no cover – missing tiktoken
    ENC = None  # type: ignore


# Instantiate one client lazily.  We avoid constructing it when the dependency
# is missing to keep import time low in minimal environments.
_OAI: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI | None:
    global _OAI
    if _OAI is None and _openai_available:
        try:
            _OAI = AsyncOpenAI()  # type: ignore[arg-type]
        except Exception:
            # Capture issues like missing API key but allow the caller to
            # continue.  We fall back to stub behaviour in that case.
            logger.warning("OpenAI client unavailable; falling back to stub", exc_info=True)
            _OAI = None  # type: ignore
    return _OAI


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

KEYWORDS = {"cite", "source", "reference", "link", "doc", "document"}


def need_search(prompt: str) -> bool:  # noqa: D401 – imperative name preferred
    """Heuristic to decide if *prompt* likely benefits from retrieval."""

    lower = prompt.lower()
    if any(k in lower for k in KEYWORDS):
        return True
    # Consider longer questions as potentially needing more context.
    return len(prompt.split()) > 75


@trace
async def search_and_summarise(prompt: str) -> str:
    """Run semantic search and return a ≤ *cheat_tokens* bullet-point summary."""

    # ---------------------------------------------------------------------
    # 1. Retrieve
    # ---------------------------------------------------------------------
    hits: Sequence[str] = await chroma_search(prompt, k=cfg.k_hits)  # type: ignore[arg-type]

    if not hits:
        return ""

    # ---------------------------------------------------------------------
    # 2. Summarise via OpenAI – revert to a trivial join when the client is
    #    not available (e.g. during unit tests).
    # ---------------------------------------------------------------------
    client = _get_client()
    if client is None:
        joined = "\n".join(f"• {h.strip()}" for h in hits)
        return _trim_tokens(joined, cfg.cheat_tokens)

    system = (
        "You are a concise research assistant. Summarise the passages into "
        "bullet points (no more than 150 tokens)."
    )
    user = "\n\n".join(hits)

    try:
        resp = await client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=cfg.cheat_tokens,
            temperature=0.2,
        )
        summary = (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("OpenAI summarisation failed – using local summary", exc_info=True)
        summary = "\n".join(f"• {h.strip()}" for h in hits)

    return _trim_tokens(summary, cfg.cheat_tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trim_tokens(text: str, limit: int | None) -> str:
    """Return *text* truncated to *limit* tokens (when both libs present)."""

    # ------------------------------------------------------------------
    # Fast-path – when *tiktoken* is available we can accurately trim by token
    # count.  Otherwise fall back to an *approximate* but safe behaviour that
    # never returns more than *limit* space-separated words.  This guarantees
    # the caller gets at most *limit* «tokens» even in minimal environments
    # where the library is missing.
    # ------------------------------------------------------------------

    if limit is None:
        return text

    if not _tiktoken_available:
        # Cheap heuristic: assume 1 word ≈ 1 token.  This intentionally favours
        # safety over accuracy – we may truncate slightly more than necessary
        # but will **never** exceed the requested limit when tiktoken is not
        # installed.
        words = text.split()
        if len(words) <= limit:
            return text
        return " ".join(words[:limit])

    tokens = ENC.encode(text)  # type: ignore[arg-type]
    if len(tokens) <= limit:
        return text
    return ENC.decode(tokens[:limit])  # type: ignore[arg-type]


__all__ = [
    "need_search",
    "search_and_summarise",
]
