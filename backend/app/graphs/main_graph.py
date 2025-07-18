"""Application LangGraph setup.

This module tries to use the *new* ``langchain-openai`` distribution first and
falls back to the deprecated import from ``langchain_community`` when the
package is not available in the runtime environment.  The conditional import
removes the deprecation warning once projects migrate, while keeping the code
working today without adding a hard dependency.
"""

from __future__ import annotations

from typing import Annotated, List


# ---------------------------------------------------------------------------
# ChatOpenAI conditional import
# ---------------------------------------------------------------------------

try:  # Prefer the new, dedicated distribution.
    from langchain_openai import ChatOpenAI  # type: ignore
except ModuleNotFoundError:  # Fallback options
    try:
        from langchain_community.chat_models import ChatOpenAI  # type: ignore  # noqa: F401
    except (
        ModuleNotFoundError
    ):  # Last-ditch stub – keeps package importable in dev envs

        class ChatOpenAI:  # type: ignore
            """Stub that surfaces a helpful error if instantiated without deps."""

            def __init__(self, *_: object, **__: object) -> None:  # noqa: D401
                raise ImportError(
                    "Install either 'langchain-openai' or 'langchain-community' to use ChatOpenAI."
                )

    # Expose "langchain_openai.ChatOpenAI" even when we fell back so external
    # imports and monkey-patches remain functional.
    import sys
    import types

    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = ChatOpenAI  # type: ignore[attr-defined]
    sys.modules.setdefault("langchain_openai", module)

# ``langchain`` and ``langgraph`` are heavy optional dependencies.  In CI or
# local-dev environments without them we fall back to *very* light-weight stub
# implementations so that the rest of the backend can start and the unit tests
# run.

try:
    from langchain.schema import AIMessage, BaseMessage  # type: ignore
except ModuleNotFoundError:  # Stub – minimal surface used in this module.
    class _Msg:  # pyright: ignore
        def __init__(self, content: str):
            self.content = content

        def __repr__(self) -> str:  # pragma: no cover
            return f"<StubMessage {self.content!r}>"

    AIMessage = _Msg  # type: ignore
    BaseMessage = _Msg  # type: ignore


try:
    from langgraph.graph import END, StateGraph  # type: ignore
except ModuleNotFoundError:  # Stub enough for type checkers / tests
    class _SG:  # pyright: ignore
        def __init__(self, *_: object, **__: object):
            pass

        def add_node(self, *_: object, **__: object):
            pass

        def set_entry_point(self, *_: object, **__: object):
            pass

        def add_edge(self, *_: object, **__: object):
            pass

        def compile(self, *_: object, **__: object):
            return self

        # The real graph has a .stream() generator; for stubs we just yield the
        # final state so that callers still iterate and gather a response.
        def stream(self, state: dict, *_: object, **__: object):
            yield {"llm": {"messages": [AIMessage("(stub) no llm")]}}

    END = "__end__"  # type: ignore
    StateGraph = _SG  # type: ignore

try:
    from langchain_core.runnables import RunnableConfig  # type: ignore
except ModuleNotFoundError:
    class RunnableConfig:  # type: ignore
        pass
from typing_extensions import TypedDict

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
    # When no API key is configured we fall back to an *echo* stub so that the
    # dev UI still shows a response instead of hanging indefinitely.
    if not settings.openai_api_key:
        user_msg = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"(stub) You said: {user_msg}")]}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=settings.openai_api_key,
    )
    ai_msg: AIMessage = llm.invoke(state["messages"])
    return {"messages": [ai_msg]}


def build_graph() -> StateGraph[GraphState]:
    """Construct the conversation graph."""
    checkpointer = get_checkpointer()
    sg: StateGraph[GraphState] = StateGraph(GraphState)
    sg.add_node("llm", _llm_node)
    sg.set_entry_point("llm")
    sg.add_edge("llm", END)
    return sg.compile(checkpointer=checkpointer)


graph = build_graph()
