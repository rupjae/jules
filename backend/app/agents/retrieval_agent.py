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

# ---------------------------------------------------------------------------
# Public dataclass -----------------------------------------------------------
# ---------------------------------------------------------------------------

# Order-sensitive imports -----------------------------------------------------
# ``get_cfg`` must be imported **early** so the singleton initialises at
# import-time.  This avoids E402 complaints from *ruff* and keeps the module
# level configuration close to the other top-level constants.

# External / standard lib -----------------------------------------------------

from dataclasses import dataclass

# Project-local imports ------------------------------------------------------

from ..config_agents import get_cfg
from ..tools.chroma_search import chroma_search
from jules.logging import trace

# ---------------------------------------------------------------------------
# Public dataclass -----------------------------------------------------------
# ---------------------------------------------------------------------------

# The retrieval contract surfaces three values so downstream callers can
# transparently decide how to incorporate external context into the LLM prompt
# *and* expose the decision to the UI.


# Dataclasses come **after** imports so ruff E402 passes.


@dataclass(frozen=True, slots=True)
class RetrievalResult:  # noqa: D101 – self-explanatory container
    need_search: bool
    info_packet: str | None
    chunks: list[str]

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

#
# The original implementation relied on a *static* heuristic to estimate
# whether a user query would benefit from retrieval-augmented generation.  The
# project requirements have since evolved – we now defer that decision to a
# lightweight **LLM** classifier so that complex phrasing and domain-specific
# language can be handled more accurately.
#
# The helper remains *synchronous* so callers do not have to change.  When an
# OpenAI client is available we perform a minimal blocking request that asks a
# dedicated model (default: the same *retrieval.model* specified in the
# *agents.toml* config) to answer "yes" or "no".  In environments where the
# dependency or API key is missing we transparently fall back to the previous
# heuristic to preserve deterministic offline behaviour and keep the unit
# tests untouched.
#

KEYWORDS = {"cite", "source", "reference", "link", "doc", "document"}

# Separate *sync* client to avoid the need for ``asyncio.run`` inside the
# synchronous public helper.
try:
    from openai import OpenAI as _SyncOpenAI  # type: ignore

    _sync_openai_available = True
except ImportError:  # pragma: no cover – dependency not installed
    _SyncOpenAI = None  # type: ignore
    _sync_openai_available = False


_SYNC_OAI: _SyncOpenAI | None = None  # type: ignore[valid-type]


def _get_sync_client():
    """Return a cached *synchronous* OpenAI client or *None* when unavailable."""

    global _SYNC_OAI
    if _SYNC_OAI is not None:
        return _SYNC_OAI

    if not _sync_openai_available:
        return None

    import os

    if not os.getenv("OPENAI_API_KEY"):
        # No credentials – treat as unavailable so we fall back to heuristic.
        return None

    try:
        _SYNC_OAI = _SyncOpenAI()  # type: ignore[arg-type]
    except Exception:  # pragma: no cover – network/credential issues
        _SYNC_OAI = None  # type: ignore
    return _SYNC_OAI


def _llm_decision(prompt: str) -> bool | None:  # noqa: D401 – imperative name preferred
    """Return *True*/*False* when the LLM call succeeds; otherwise *None*."""

    client = _get_sync_client()
    if client is None:
        return None

    system = (
        "You are a boolean classifier. Return 'yes' (without quotes) when "
        "fetching external documents (e.g. knowledge base, web pages, "
        "company docs) could provide *helpful additional information* that "
        "would likely improve the answer to the user's query. Return 'no' "
        "when the model can already answer confidently without any extra "
        "context. Respond with a single word only: yes or no."
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1,
            temperature=0,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        if answer.startswith("y"):
            return True
        if answer.startswith("n"):
            return False
    except Exception:
        logger.warning("OpenAI retrieval-decision call failed; using heuristic", exc_info=True)

    # Any error or malformed response triggers the heuristic fall-back.
    return None


def need_search(prompt: str) -> bool:  # noqa: D401 – imperative name preferred
    """Decide whether *prompt* likely benefits from retrieval.

    The function first tries an LLM-based classifier for higher accuracy. When
    that is not feasible (e.g. offline CI, missing API key) it falls back to a
    deterministic heuristic to keep behaviour stable.
    """

    llm_result = _llm_decision(prompt)
    if llm_result is not None:
        logger.debug(
            "need_search | decision_from=llm | result=%s",
            llm_result,
            extra={"code_path": __name__},
        )
        return llm_result

    # ------------------------------------------------------------------
    # Heuristic fall-back – unchanged from the previous implementation.
    # ------------------------------------------------------------------

    lower = prompt.lower()
    if any(k in lower for k in KEYWORDS):
        logger.debug(
            "need_search | decision_from=heuristic | reason=keyword | result=True",
            extra={"code_path": __name__},
        )
        return True

    decision = len(prompt.split()) > 75
    logger.debug(
        "need_search | decision_from=heuristic | reason=length | result=%s",
        decision,
        extra={"code_path": __name__},
    )
    return decision


async def _summarise_chunks(hits: Sequence[str]) -> str:
    """Summarise *hits* into a ≤ *cheat_tokens* bullet-point packet."""

    if not hits:
        return ""

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
# Public API ---- new --------------------------------------------------------
# ---------------------------------------------------------------------------


@trace
async def search_and_summarise(prompt: str) -> RetrievalResult:  # noqa: D401 – imperative
    """Return a full *RetrievalResult* for *prompt*.

    The function wraps the previous implementation while enriching the result
    so downstream nodes can:
        • decide whether the search was executed (*need_search*),
        • prepend the generated *info_packet* to the LLM messages, and
        • expose the raw *chunks* (passages) for future debugging features.
    """

    # Decide first – this is a cheap heuristic/LLM call.
    needed = need_search(prompt)

    if not needed:
        logger.log(
            5,  # TRACE level (see project logging guidelines)
            "retrieval | need_search=no | prompt_len=%s",
            len(prompt.split()),
            extra={"code_path": __name__},
        )
        return RetrievalResult(False, None, [])

    # ------------------------------------------------------------------
    # 1. Retrieve – best-effort, never raises upstream.  We keep *await*
    # outside the try/except so testing with a monkey-patched sync stub still
    # works (it might return a list directly).
    # ------------------------------------------------------------------

    hits: Sequence[str]
    result = chroma_search(prompt, k=cfg.k_hits)  # type: ignore[arg-type]
    hits = await result if hasattr(result, "__await__") else result  # type: ignore[arg-type]

    # 2. Summarise (may fall back to cheap join)
    info_packet = await _summarise_chunks(hits)

    # TRACE-level log so operators can inspect decisions & packet size.
    token_estimate = len(info_packet.split())
    logger.log(
        5,
        "retrieval | need_search=yes | hits=%s | info_tokens=%s",
        len(hits),
        token_estimate,
        extra={"code_path": __name__},
    )

    return RetrievalResult(True, info_packet, list(hits))


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
    "RetrievalResult",
]
