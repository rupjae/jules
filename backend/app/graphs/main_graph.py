"""Application LangGraph setup."""

from __future__ import annotations

from typing import Annotated, List

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage
import datetime
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from typing import Any

# NOTE: this module now supports multiple graph versions (v5, v6, …).  The
# v5 implementation lives below unchanged, while newer versions are lazily
# imported to avoid pulling heavyweight deps at import-time.

from ..config import get_settings
from ..checkpointer import get_checkpointer

settings = get_settings()


def _merge_messages(
    existing: List[BaseMessage], new: List[BaseMessage] | None
) -> List[BaseMessage]:
    """Reducer that appends new messages to history."""
    return existing + (new or [])


class GraphState(TypedDict):
    """State shared across graph nodes."""

    messages: Annotated[List[BaseMessage], _merge_messages]


def _llm_node(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    """Generate assistant reply using the configured LLM."""
    llm = ChatOpenAI(
        model="gpt-4.1",
        # temperature=0.2,
        openai_api_key=settings.openai_api_key,
    )
    # Generate AI response and annotate with timestamp
    raw_msg: AIMessage = llm.invoke(state["messages"])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # preserve content and attach timestamp metadata
    ai_msg = AIMessage(
        content=raw_msg.content,
        additional_kwargs={
            **getattr(raw_msg, "additional_kwargs", {}),
            "timestamp": now,
        },
    )
    return {"messages": [ai_msg]}


def _build_v5_graph() -> StateGraph[GraphState]:
    """Construct the *legacy* v5 conversation graph."""

    checkpointer = get_checkpointer()
    sg: StateGraph[GraphState] = StateGraph(GraphState)
    sg.add_node("llm", _llm_node)
    sg.set_entry_point("llm")
    sg.add_edge("llm", END)
    return sg.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Graph registry helpers
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Any] = {}


def _ensure_v5() -> None:
    if "v5" not in _REGISTRY:
        _REGISTRY["v5"] = _build_v5_graph()


def _ensure_v6() -> None:
    """Import and cache the v6 graph lazily to avoid heavy deps on startup."""

    if "v6" in _REGISTRY:
        return

    from jules.graph_v6 import graph as v6_graph  # local import – heavy

    _REGISTRY["v6"] = v6_graph


def get_graph(req_version: str | None = None):  # noqa: D401 – helper accessor
    """Return the compiled graph matching *req_version*.

    Fallback order:
    1. Explicit match (case-insensitive).
    2. Default to ``v5`` when the supplied version is ``None`` or unknown.
    """

    version = (req_version or "v5").lower()

    # Always make sure v5 is present.
    _ensure_v5()

    if version == "v6":
        _ensure_v6()

    return _REGISTRY.get(version, _REGISTRY["v5"])


# Keep legacy *build_graph()* alias for existing imports (returns v5).


def build_graph():  # type: ignore[override]
    return get_graph("v5")
