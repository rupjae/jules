"""LangGraph *v6* – retrieval-aware pipeline.

Nodes:
1. ``input`` – packs the raw user message into graph state.
2. ``retrieval_decide`` – lightweight RetrievalAgent decides whether we need
   external context.
3. Conditional edge → ``chroma`` – only taken when a search is required.
4. ``retrieval_summary`` – same RetrievalAgent summarises the search hits.
5. ``jules`` – heavyweight responder producing the final answer.
"""

from __future__ import annotations

from typing import Annotated, List, Optional

# Optional import
try:
    from langchain.schema import BaseMessage  # type: ignore
except ImportError:  # pragma: no cover – optional dependency
    class BaseMessage:  # type: ignore
        ...
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from jules.agents.retrieval import RetrievalAgent
from jules.agents.jules import JulesAgent
from jules.tools.search_tool import ChromaSearchTool, SearchResult
import logging
from jules.logging import trace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    user_message: str
    need_search: bool
    query: str
    k: int
    search_hits: List[SearchResult]
    cheat_sheet: str
    reply: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _input_node(message: str, _state: GraphState | None = None) -> GraphState:  # type: ignore[override]
    return {"user_message": message}


_retrieval_agent = RetrievalAgent()


@trace
def _retrieval_decide(state: GraphState, _: RunnableConfig | None = None) -> GraphState:  # noqa: D401
    """Node: decide if search is needed."""
    import asyncio

    res = asyncio.run(_retrieval_agent.decide(state["user_message"]))
    return res  # type: ignore[return-value]


_search_tool = ChromaSearchTool()


@trace
def _chroma_node(state: GraphState, _: RunnableConfig | None = None) -> GraphState:  # noqa: D401
    import asyncio

    hits = asyncio.run(_search_tool(query=state["query"], k=state["k"]))
    return {"search_hits": hits}


@trace
def _retrieval_summary(state: GraphState, _: RunnableConfig | None = None) -> GraphState:  # noqa: D401
    import asyncio

    cheat_sheet = asyncio.run(
        _retrieval_agent.cheat_sheet_from_results(state.get("search_hits", []))
    )
    return {"cheat_sheet": cheat_sheet}


_jules_agent = JulesAgent()


@trace
def _jules_node(state: GraphState, _: RunnableConfig | None = None) -> GraphState:  # noqa: D401
    import asyncio

    reply = asyncio.run(
        _jules_agent(state["user_message"], cheat_sheet=state.get("cheat_sheet", ""))
    )
    return {"reply": reply}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph[GraphState]:
    sg: StateGraph[GraphState] = StateGraph(GraphState)

    sg.add_node("retrieval_decide", _retrieval_decide)
    sg.add_node("chroma", _chroma_node)
    sg.add_node("retrieval_summary", _retrieval_summary)
    sg.add_node("jules", _jules_node)

    # Entry point packs user input – LangGraph entry nodes don't take arguments
    sg.set_entry_point("retrieval_decide")

    # Conditional edge – only run chroma when search is required
    sg.add_conditional_edges(
        "retrieval_decide",
        lambda s: "chroma" if s.get("need_search") else "retrieval_summary",
    )

    sg.add_edge("chroma", "retrieval_summary")
    sg.add_edge("retrieval_summary", "jules")
    sg.add_edge("jules", END)

    return sg.compile()


graph = build_graph()
