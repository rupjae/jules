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

from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from langchain.schema import HumanMessage, SystemMessage
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
from sse_starlette import sse as sse_mod

from ..config import Settings, get_settings
from .graph_runner import run_graph
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
# ---------------------------------------------------------------------------
# RAG stream endpoint (new /chat/stream path)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Streaming endpoint (GET-based for native browser EventSource support)
# ---------------------------------------------------------------------------


@router.get("/chat/stream")
async def chat_stream(request: Request, prompt: str = Query(..., description="User prompt")):
    """Stream assistant tokens (and a final *info_packet*) via SSE.

    The endpoint is **GET**-only so that browsers can establish an EventSource
    without CORS or polyfill workarounds.  The *prompt* is provided as a query
    string parameter – body payloads are explicitly rejected by FastAPI's
    signature (no pydantic model argument).
    """

    # Build LangGraph instance – created at startup and cached on app state
    graph = request.app.state.graph
    state = {"prompt": prompt, "info_packet": None}

    async def event_generator():
        last_info: str | None = None
        # When running with real streaming the UI progressively renders
        # *partial* tokens.  In non-streaming scenarios (e.g. stub mode) the graph
        # may only emit one final *content* message – capture it so we can
        # still return a meaningful payload to the EventSource consumer.
        last_content: str | None = None

        # Flag to track whether we already streamed *partial* tokens.  When we
        # did, sending the *full* message again at the end would duplicate the
        # content on the client (the UI concatenates every incoming chunk).
        streamed_partials = False

        async for output in run_graph(graph, state):
            # 1. Progressive tokens
            if "partial" in output:
                streamed_partials = True
                # Starlette will automatically prepend "data: " and append a
                # final blank line when the yielded value is *str*.
                yield output["partial"]

            # 2. Final assistant message emitted when streaming is unavailable.
            #    Capture it so we can forward **once** the graph finishes – but
            #    only when we have *not* streamed any partial tokens (to avoid
            #    duplication on the client side).
            if "content" in output:
                last_content = output["content"]

            # 3. Keep track of the most recent info_packet so we can forward
            #    it once the graph is done.
            if "info_packet" in output:
                last_info = output["info_packet"]

        # ------------------------------------------------------------------
        # Graph finished – emit any *content* captured from a non-streaming
        # run **first**, followed by the optional *info_packet*.
        # ------------------------------------------------------------------

        # Only forward the *full* assistant message when we have **not**
        # already streamed incremental tokens.  When *streamed_partials* is
        # True the UI has the complete text assembled locally and would show
        # a duplicated copy if we sent the message again.

        if last_content is not None and not streamed_partials:
            yield last_content

        # Emit single *info_packet* event by streaming the
        # *event* and *data* lines as one logical block (no blank line until
        # after the data) so they belong to the same SSE event.

        # According to the SSE spec each event is separated by a *blank line*.
        # Therefore we must not send the terminating "\n\n" after the
        # ``event:`` directive alone – doing so would create an event without
        # a *data:* field (observed as "data: null" in the UI).  Instead we
        # write both lines back-to-back and finish with the mandatory blank
        # line.

        # Emit *info_packet* only when the graph produced one – callers may
        # disable that feature and rely on the default *None* value.  In that
        # case we avoid sending a meaningless event that would clutter the
        # client-side event stream.


        # Always emit a *single* info_packet event so the frontend can reliably
        # detect when the assistant has finished streaming and clear its
        # loading indicator.  When the graph did not produce extra metadata we
        # explicitly send the JSON literal `null` (as a string) which the
        # browser forwards as the text "null".  The React hook converts that
        # value back to the JavaScript `null` primitive.

        payload = None if last_info is None else last_info
        # ServerSentEvent will JSON-serialise non-str data, therefore convert
        # `None` to the *string* "null" so the frontend can easily interpret
        # it (the hook already maps the literal back to JavaScript null).
        data_field = "null" if payload is None else payload

        yield {"event": "info_packet", "data": data_field}

        # Keep the SSE connection alive with periodic comments so the browser
        # does **not** aggressively reconnect (which manifests as an endless
        # request loop on the server).  The client will close the connection
        # on its side once it processed the final info_packet.

        try:
            while True:
                await asyncio.sleep(15)
                # SSE comment line – ignored by the client but prevents idle
                # TCP timeouts and keeps the *readyState* at OPEN.
                yield ":\n\n"
        except asyncio.CancelledError:
            # Client disconnected – exit quietly.
            return

    return EventSourceResponse(event_generator())
