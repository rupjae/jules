"""Chat router exposing the memory-aware LangGraph endpoint.

The router is mounted with prefix "/api" so the final path is /api/chat.
It streams tokens via Server-Sent Events (SSE) and keeps per-thread
conversation context using the configured LangGraph checkpoint saver.
"""

from __future__ import annotations

from typing import AsyncGenerator, List
import logging
import asyncio
from uuid import UUID, uuid4
import time
import anyio
from db.chroma import save_message, StoredMsg
from db import sqlite

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain.schema import HumanMessage, SystemMessage
from pathlib import Path
from sse_starlette.sse import EventSourceResponse

from ..config import Settings, get_settings
import datetime


router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)
THREAD_ID_HEADER = "X-Thread-ID"


async def _authorize(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Bearer-token guard. Disabled when JULES_AUTH_TOKEN is unset."""

    if settings.auth_token is None:
        return

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")

    if scheme.lower() != "bearer" or token != settings.auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )


@router.get("/chat")
async def chat_endpoint(
    request: Request, settings: Settings = Depends(get_settings)
):  # noqa: D401
    """Stream chat completions via SSE with thread-scoped memory."""

    await _authorize(request, settings)

    prompt: str = request.query_params.get("message", "")
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message is required",
        )

    raw_id = request.headers.get(THREAD_ID_HEADER) or request.query_params.get(
        "thread_id"
    )
    if raw_id is not None:
        try:
            thread_id = str(UUID(raw_id, version=4))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid thread_id") from exc
    else:
        thread_id = str(uuid4())

    graph = request.app.state.graph

    async def generator() -> AsyncGenerator[str, None]:
        """Run the *blocking* graph.exec in a thread pool and stream tokens."""

        # Provide the **new** user message only.  LangGraph will automatically
        # hydrate the previous conversation from the configured checkpointer
        # (SQLite or in-memory) when a checkpoint for *thread_id* exists.  This
        # avoids the earlier "checkpoint['v'] is None" migration bug.

        # Load system prompt and include it before the user message
        prompts_path = Path(__file__).parent.parent / "prompts" / "root.system.md"
        try:
            system_text = prompts_path.read_text(encoding="utf-8").strip()
        except Exception:
            system_text = ""
        # Prepare messages: system prompt followed by user message with timestamp
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        messages: list = []
        if system_text:
            messages.append(SystemMessage(content=system_text))
        messages.append(
            HumanMessage(content=prompt, additional_kwargs={"timestamp": now})
        )
        user_state = {"messages": messages}

        loop = asyncio.get_running_loop()

        def _run_sync() -> List[str]:
            latest_tokens: List[str] = []
            cfg = {"configurable": {"thread_id": thread_id}}
            for step in graph.stream(user_state, cfg):
                messages: List | None = step.get("llm", {}).get("messages")
                if messages:
                    latest_tokens = list(messages[-1].content)
            return latest_tokens

        tokens: List[str] = await loop.run_in_executor(None, _run_sync)
        for ch in tokens:
            yield ch

    # Attach the (possibly freshly generated) thread id so clients can persist it.
    return EventSourceResponse(generator(), headers={"X-Thread-ID": thread_id})


@router.get("/chat/history")
async def chat_history(request: Request, settings: Settings = Depends(get_settings)):
    """Return the full message history for a given thread_id."""
    await _authorize(request, settings)
    raw_id = request.headers.get(THREAD_ID_HEADER) or request.query_params.get(
        "thread_id"
    )
    if raw_id is None:
        raise HTTPException(status_code=400, detail="thread_id is required")
    try:
        thread_id = str(UUID(raw_id, version=4))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid thread_id") from exc
    # Load saved state for this thread via the SqliteSaver get_tuple
    saver = request.app.state.checkpointer
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        tup = saver.get_tuple(cfg)
        if tup is None:
            msgs = []
        else:
            cp = tup.checkpoint
            if isinstance(cp, dict):
                # channel_values holds per-channel state
                channel_vals = cp.get("channel_values", {})
                msgs = channel_vals.get("messages", []) or []
            else:
                msgs = getattr(cp, "messages", []) or []
    except Exception:
        # error loading history, return empty
        msgs = []
    # Convert to serializable form
    result = []
    from langchain.schema import AIMessage

    for m in msgs:
        # determine sender and extract timestamp if present
        sender = "assistant" if isinstance(m, AIMessage) else "user"
        ts = getattr(m, "additional_kwargs", {}).get("timestamp")
        entry: dict = {"sender": sender, "content": m.content}
        if ts is not None:
            entry["timestamp"] = ts
        result.append(entry)
    return result


@router.post("/chat/message")
async def post_message(
    request: Request,
    thread_id: str,
    role: str,
    content: str,
    settings: Settings = Depends(get_settings),
):
    """Persist a message and index it in Chroma."""

    await _authorize(request, settings)

    msg = StoredMsg(
        id=str(uuid4()),
        thread_id=thread_id,
        role=role,
        content=content,
        ts=time.time(),
    )
    await sqlite.insert(sqlite.ChatMessage(**msg.model_dump()))
    try:
        await anyio.to_thread.run_sync(save_message, msg)
    except Exception:
        # Swallow vector store errors
        logger.warning("Chroma save failed", exc_info=True)
    return {"id": msg.id}


@router.get("/chat/search")
async def chat_search(
    request: Request,
    thread_id: str,
    query: str,
    settings: Settings = Depends(get_settings),
):
    """Vector search for messages within *thread_id*."""

    await _authorize(request, settings)
    from db.chroma import search as chroma_search

    try:
        hits = chroma_search(thread_id, query, k=8)
    except Exception:
        raise HTTPException(status_code=503, detail="vector search unavailable")

    return hits
