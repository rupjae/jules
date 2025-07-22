from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph

try:  # newer langgraph
    from langgraph.nodes import LLMNode, ToolNode  # type: ignore
except Exception:  # fallback for older versions
    from langgraph.prebuilt.tool_node import ToolNode  # type: ignore

    class LLMNode:  # pragma: no cover - placeholder
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("LLMNode unavailable")


from jules.tools import ChromaSearchTool, SearchResult
from jules.config import get_agent_cfg

cfg = get_agent_cfg()


class ChatState(TypedDict, total=False):
    """State shared across graph nodes."""

    user_msg: str
    search_call: dict
    retrieval: list[SearchResult]
    info_packet: str
    assistant_reply: str


# --- Tool -------------------------------------------------
search_tool = ChromaSearchTool()
SEARCH_NODE = ToolNode([search_tool.__call__], name="chroma_search")

# --- Retrieval Agent -------------------------------------
retrieval_llm = LLMNode(
    model_name=cfg["retrieval_agent"]["model"],
    system_prompt=(
        Path("prompts/retrieval_agent.txt")
        .read_text()
        .format(
            top_k=cfg["retrieval_agent"]["top_k"],
            max_tokens=cfg["retrieval_agent"]["max_tokens_packet"],
        )
    ),
)

# --- Jules ------------------------------------------------
jules_llm = LLMNode(
    model_name=cfg["jules"]["model"],
    system_prompt=Path("prompts/jules.txt").read_text(),
)


def route_after_retrieval(result: dict) -> str:
    """Route to tool when the LLM emitted a function call."""

    return "tool" if result.get("type") == "function_call" else "skip"


def build_graph() -> StateGraph[ChatState]:
    """Construct the retrieval-augmented conversation graph."""

    sg: StateGraph[ChatState] = StateGraph(ChatState)
    sg.add_node("retrieval_agent", retrieval_llm)
    sg.add_node("chroma_search", SEARCH_NODE)
    sg.set_entry_point("retrieval_agent")
    sg.add_node("jules", jules_llm)

    sg.add_conditional_edges(
        "retrieval_agent",
        route_after_retrieval,
        {"tool": "chroma_search", "skip": "jules"},
    )
    sg.add_edge("chroma_search", "retrieval_agent")
    sg.add_edge("retrieval_agent", "jules")
    sg.set_finish_point("jules")
    return sg.compile()
