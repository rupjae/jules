"""Application LangGraph setup."""

from __future__ import annotations

from typing import Annotated, List

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage
import datetime
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
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
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=settings.openai_api_key,
    )
    # Generate AI response and annotate with timestamp
    raw_msg: AIMessage = llm.invoke(state["messages"])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # preserve content and attach timestamp metadata
    ai_msg = AIMessage(content=raw_msg.content,
                       additional_kwargs={**getattr(raw_msg, 'additional_kwargs', {}),
                                          'timestamp': now})
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
