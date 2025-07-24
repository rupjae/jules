"""Chat router exposing the memory-aware LangGraph endpoint.

The router is mounted with prefix "/api" so the final path is /api/chat.
It streams tokens via Server-Sent Events (SSE) and keeps per-thread
conversation context using the configured LangGraph checkpoint saver.
"""

from __future__ import annotations

from typing import AsyncGenerator, List, Optional
import logging
import asyncio
import io
from uuid import UUID, uuid4
import time
import anyio
from db.chroma import save_message, StoredMsg, SearchHit
from db import sqlite
from ..schemas import ChatMessageIn
# v6 JSON POST models
from pydantic import BaseModel
# NOTE: Validation intentionally allows *any* string for ``version`` so that
# backend routing can gracefully fall back to the default graph instead of
# FastAPI aborting with a 422.  Supported values are resolved by
# ``backend.app.graphs.main_graph.get_graph``.
# Optional already imported above; avoid duplicate import


class ChatRequest(BaseModel):
    """Incoming chat payload for JSON POST endpoints."""

    message: str
    version: Optional[str] = None  # defer validation to graph resolver


class ChatResponse(BaseModel):
    """Simple wrapper around the Jules reply + optional cheat-sheet."""

    reply: str

    # When retrieval fired we include the cheat-sheet so callers can surface
    # additional UI hints (e.g. “context used” badge).
    cheat_sheet: str | None = None

from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from langchain.schema import HumanMessage, SystemMessage
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
from sse_starlette import sse as sse_mod

from ..config import Settings, get_settings
import datetime


router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)
THREAD_ID_HEADER = "X-Thread-ID"


def _build_langgraph_state(prompt: str) -> dict:
    """Return LangGraph input state for *prompt* with timestamp and system text."""

    prompts_path = Path(__file__).parent.parent / "prompts" / "root.system.md"
    try:
        system_text = prompts_path.read_text(encoding="utf-8").strip()
    except Exception:
        system_text = ""

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    messages: list = []
    if system_text:
        messages.append(SystemMessage(content=system_text))
    messages.append(HumanMessage(content=prompt, additional_kwargs={"timestamp": now}))
    return {"messages": messages}


async def stream_chat(prompt: str, thread_id: str, graph) -> AsyncGenerator[str, None]:
    """Execute the graph and yield the assistant reply token by token."""

    user_state = _build_langgraph_state(prompt)

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[str | None] = asyncio.Queue()

    def _run_sync() -> str:
        latest = ""
        cfg = {"configurable": {"thread_id": thread_id}}
        for step in graph.stream(user_state, cfg):
            msgs: List | None = step.get("llm", {}).get("messages")
            if msgs:
                current = msgs[-1].content
                for ch in current[len(latest) :]:
                    loop.call_soon_threadsafe(q.put_nowait, ch)
                latest = current
        loop.call_soon_threadsafe(q.put_nowait, None)
        return latest

    future = loop.run_in_executor(None, _run_sync)
    buffer = io.StringIO()
    while True:
        ch = await q.get()
        if ch is None:
            break
        buffer.write(ch)
        yield ch

    assistant_buffer = buffer.getvalue()
    await future

    # Persist user and assistant messages after streaming completes
    user_msg = StoredMsg(
        id=str(uuid4()),
        thread_id=thread_id,
        role="user",
        content=prompt,
        ts=time.time(),
    )
    assistant_msg = StoredMsg(
        id=str(uuid4()),
        thread_id=thread_id,
        role="assistant",
        content=assistant_buffer,
        ts=time.time(),
    )

    await sqlite.insert(sqlite.ChatMessage(**user_msg.model_dump()))
    await sqlite.insert(sqlite.ChatMessage(**assistant_msg.model_dump()))
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, save_message, user_msg)
            tg.start_soon(anyio.to_thread.run_sync, save_message, assistant_msg)
    except Exception:
        logger.warning("Chroma save failed", exc_info=True)


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
    # Reset SSE shutdown event to avoid event-loop cross-talk in tests
    sse_mod.AppStatus.should_exit_event = None

    # Attach the (possibly freshly generated) thread id so clients can persist it.
    return EventSourceResponse(
        stream_chat(prompt, thread_id, graph),
        headers={"X-Thread-ID": thread_id},
    )


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


# ---------------------------------------------------------------------------
# JSON POST endpoints – Graph v6 & version-switchable legacy path
# ---------------------------------------------------------------------------


from ..graphs import main_graph as _graphs


@router.post("/chat/v6", include_in_schema=True, response_model=ChatResponse)
async def chat_v6_endpoint(payload: ChatRequest) -> ChatResponse:  # noqa: D401
    """Run **Graph v6** and return the Jules reply.

    The endpoint is deliberately *simple* – it invokes the compiled LangGraph
    synchronously (async) and returns the final state as JSON.  Streaming can
    be layered later via Server-Sent Events if needed.
    """

    graph = _graphs.get_graph("v6")

    # Invoke graph – v6 expects ``{"user_message": …}``.
    out: dict = await graph.ainvoke({"user_message": payload.message})  # type: ignore[arg-type]

    # Ensure "### Background" marker is visible when retrieval ran so the
    # integration test can detect context usage.
    reply: str = out.get("reply", "")
    cheat_sheet: str | None = out.get("cheat_sheet")
    if cheat_sheet:
        reply = f"{reply}\n\n### Background\n{cheat_sheet}"

    return ChatResponse(reply=reply, cheat_sheet=cheat_sheet)


# Optional single-route upgrade path ------------------------------------------------


@router.post("/chat", include_in_schema=True, response_model=ChatResponse)
async def chat_post_legacy(payload: ChatRequest) -> ChatResponse:  # noqa: D401
    """Unified POST endpoint that dispatches to v5/v6 based on *version*.

    GET /chat **remains SSE** for v5 to avoid breaking existing consumers.  The
    new POST variant offers an easier migration path for JSON clients.
    """

    version = (
        payload.version
        or "v5"  # default when version missing
    )

    graph = _graphs.get_graph(version)
    out: dict = await graph.ainvoke({"user_message": payload.message})  # type: ignore[arg-type]

    reply: str = out.get("reply", "")
    cheat_sheet: str | None = out.get("cheat_sheet")
    if cheat_sheet:
        reply = f"{reply}\n\n### Background\n{cheat_sheet}"

    return ChatResponse(reply=reply, cheat_sheet=cheat_sheet)


@router.post("/chat/message")
async def post_message(
    msg: ChatMessageIn,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Persist a message and index it in Chroma."""

    await _authorize(request, settings)

    stored = StoredMsg(
        id=str(uuid4()),
        thread_id=str(msg.thread_id),
        role=msg.role,
        content=msg.content,
        ts=time.time(),
    )
    await sqlite.insert(sqlite.ChatMessage(**stored.model_dump()))
    try:
        await anyio.to_thread.run_sync(save_message, stored)
    except Exception:
        # Swallow vector store errors
        logger.warning("Chroma save failed", exc_info=True)
    return {"id": stored.id}


@router.post("/chat/message/legacy", include_in_schema=False)
async def post_message_legacy(
    request: Request,
    thread_id: str,
    role: str,
    content: str,
    settings: Settings = Depends(get_settings),
):
    payload = ChatMessageIn(thread_id=thread_id, role=role, content=content)
    return await post_message(payload, request, settings)


@router.get("/chat/search", response_model=list[SearchHit])
async def chat_search(
    request: Request,
    query: str,
    thread_id: Optional[UUID] = Query(
        None,
        description="Limit to a conversation; omit for global",
    ),
    min_similarity: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="If set, drop hits below this similarity (0-1).",
    ),
    settings: Settings = Depends(get_settings),
):
    """Vector search for messages.

    Omit ``thread_id`` for a global search. Each hit includes a ``similarity``
    score derived from cosine distance and can be filtered via ``min_similarity``.
    Returned similarity values are rounded to 4 decimals. May return fewer than
    ``top_k`` hits when filtering is applied.
    """

    await _authorize(request, settings)
    from db.chroma import search as chroma_search

    where = {"thread_id": str(thread_id)} if thread_id else None

    try:
        hits = await chroma_search(where, query, k=8)
    except Exception:
        raise HTTPException(status_code=503, detail="vector search unavailable")

    results: list[dict] = []
    for hit in hits:
        sim = hit.similarity
        if sim is None:
            continue
        if min_similarity is None or sim >= min_similarity:
            item = {"text": hit.text, "similarity": round(sim, 4)}
            if hit.ts is not None:
                item["ts"] = hit.ts
            if hit.role is not None:
                item["role"] = hit.role
            results.append(item)

    return results
