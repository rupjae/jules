"""Lightweight retrieval-aware agent.

The agent is intentionally minimal and *stateless* – callers are expected to
pass along any conversational context they deem relevant.  The flow mirrors
the spec in the work-order:

1. Inspect the **user message** and decide whether external knowledge is
   required.
2. If so, emit a function-call for ``search_chroma``.
3. The hosting graph invokes :class:`jules.tools.ChromaSearchTool` with the
   provided arguments.
4. A *second* invocation distils the raw ``SearchResult`` list into a
   **≤150-token** cheat-sheet that downstream agents can slot into their
   prompt.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Sequence

# Optional import – the CI image might not include this extra.
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover – stub when dependency missing
    ChatOpenAI = None  # type: ignore[assignment]

# LangChain is an optional dependency for unit-test speed.  When missing we
# replace it with tiny stubs so type-checking is preserved while avoiding runtime
# crashes.

try:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage  # type: ignore
except ImportError:  # pragma: no cover – CI without langchain extra
    class _BareMsg(str):
        @property
        def content(self):
            return str(self)

    AIMessage = HumanMessage = SystemMessage = _BareMsg  # type: ignore

# ---------------------------------------------------------------------------
# Offline stub – *just enough* to keep the tests happy without hitting the
# network.  The behaviour is extremely dumb but deterministic.
# ---------------------------------------------------------------------------


class _OfflineLLM:
    """Very small stub that emulates `ainvoke` without network access."""

    async def ainvoke(self, messages, **_) -> AIMessage:  # type: ignore[override]
        return AIMessage("NO_SEARCH")  # type: ignore[arg-type]

from jules.config import RetrievalCfg, get_agent_cfg
from jules.tools.search_tool import ChromaSearchTool, SearchResult
import logging
from jules.logging import trace

logger = logging.getLogger(__name__)

# Public API -----------------------------------------------------------------


class RetrievalAgent:
    """LLM-powered tool selector + summariser."""

    def __init__(
        self,
        cfg: Optional[RetrievalCfg] = None,
        *,
        llm: Optional[ChatOpenAI] = None,
        search_tool: Optional[ChromaSearchTool] = None,
    ) -> None:
        self.cfg = cfg or get_agent_cfg("retrieval")  # type: ignore[arg-type]

        if llm is not None:
            self.llm = llm
        else:
            # Provide an offline stub to avoid network calls during unit
            # testing.  If the OPENAI key is configured we fall back to the
            # real model.
            if os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
                self.llm = ChatOpenAI(model=self.cfg.model, temperature=0)  # type: ignore[arg-type]
            else:
                self.llm = _OfflineLLM()
        self.search_tool = search_tool or ChromaSearchTool()

        self._fn_schema = {
            "name": "search_chroma",
            "description": "Searches the Chroma DB for relevant conversation snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to fetch (≤ cfg.k)",
                    },
                },
                "required": ["query"],
            },
        }

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @trace
    async def decide(  # noqa: D401 – imperative mood is fine here
        self, user_message: str, *, thread_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Return *need_search*, and if applicable the search arguments.

        Example return objects:

        * ``{"need_search": False}``
        * ``{"need_search": True, "query": "SIEM licensing", "k": 4}``
        """

        # Fast exit for offline testing – assume no additional context needed.
        if isinstance(self.llm, _OfflineLLM):
            return {"need_search": False}

        sys_prompt = (
            "You read the user's message FIRST.\n"
            "If outside knowledge is needed, CALL function\n"
            "search_chroma(query, k). Otherwise, answer \"NO_SEARCH\"."
        )

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_message),
        ]

        response: AIMessage = await self.llm.ainvoke(
            messages,
            functions=[self._fn_schema],
        )

        # Branch 1 – no external search required
        if response.content and response.content.strip().upper() == "NO_SEARCH":
            logger.info(
                "SEARCH_DECISION",
                extra={
                    "code_path": __name__,
                    "need_search": False,
                },
            )
            return {"need_search": False}

        # Branch 2 – function call
        call = response.additional_kwargs.get("function_call")  # type: ignore[attr-defined]
        if not call:
            # Fallback to *no search* to avoid hard failure
            return {"need_search": False}

        args = json.loads(call.get("arguments", "{}"))
        query: str = args.get("query", user_message)
        k: int = int(args.get("k", self.cfg.k))
        k = max(1, min(k, self.cfg.k))

        logger.info(
            "SEARCH_DECISION",
            extra={
                "code_path": __name__,
                "need_search": True,
                "query": query,
                "k": k,
            },
        )

        return {"need_search": True, "query": query, "k": k, "thread_id": thread_id}

    @trace
    async def cheat_sheet_from_results(
        self, results: Sequence[SearchResult]
    ) -> str:
        """Generate a ≤summary_tokens-token bullet-point cheat-sheet."""

        if not results:
            return ""

        docs_text = "\n\n".join(f"- {hit.text}" for hit in results)
        prompt = (
            "Summarise the following context into concise Markdown bullet points. "
            f"Stay under {self.cfg.summary_tokens} tokens.\n\n{docs_text}"
        )

        # Initialise here so it is *always* defined for the ``finally`` block
        summary: str = ""
        try:
            # -----------------------------------------------------------------
            # Offline path – deterministic join capped by cfg.summary_tokens.
            # -----------------------------------------------------------------
            if isinstance(self.llm, _OfflineLLM):
                joined = "\n".join(f"- {hit.text}" for hit in results[: self.cfg.k])

                tokens = joined.split()
                if len(tokens) > self.cfg.summary_tokens:
                    joined = " ".join(tokens[: self.cfg.summary_tokens]) + " …"

                summary = joined

            # -----------------------------------------------------------------
            # Online path – delegate summarisation to the LLM and *then* enforce
            # the token budget post-hoc.
            # -----------------------------------------------------------------
            else:
                msg = await self.llm.ainvoke(
                    [SystemMessage(content=prompt)],
                    max_tokens=self.cfg.summary_tokens,
                )

                summary = msg.content.strip()

                # Hard truncate using tiktoken if available to guarantee the
                # ≤summary_tokens contract.
                try:
                    import tiktoken  # type: ignore

                    enc = tiktoken.encoding_for_model(self.cfg.model)
                    tokens = enc.encode(summary)
                    if len(tokens) > self.cfg.summary_tokens:
                        summary = enc.decode(tokens[: self.cfg.summary_tokens]) + " …"
                except Exception:
                    # Soft-fail: keep raw text on any tiktoken issue.
                    pass

            return summary
        finally:
            # Telemetry must *never* raise – swallow any error defensively.
            try:
                cheat_tokens = len(summary.split()) if summary else 0
                logger.info(
                    "SEARCH_SUMMARY",
                    extra={"code_path": __name__, "cheat_tokens": cheat_tokens},
                )
            except Exception:
                pass


    # ------------------------------------------------------------------
    # Convenience one-shot helper used outside LangGraph
    # ------------------------------------------------------------------

    @trace
    async def run(
        self, user_message: str, *, thread_id: Optional[str] = None
    ) -> dict[str, Any]:
        """End-to-end helper combining *decide* + *summary* stages."""

        decision = await self.decide(user_message, thread_id=thread_id)
        if not decision.get("need_search"):
            decision["cheat_sheet"] = ""
            return decision

        query = decision["query"]
        k = decision["k"]

        hits = await self.search_tool(query=query, k=k, thread_id=thread_id)
        cheat_sheet = await self.cheat_sheet_from_results(hits)

        return {
            "need_search": True,
            "cheat_sheet": cheat_sheet,
        }
