"""Chat router exposing the memory-aware LangGraph endpoint.

The router is mounted with prefix "/api" so the final path is /api/chat.
It streams tokens via Server-Sent Events (SSE) and keeps per-thread
conversation context using the configured LangGraph checkpoint saver.
"""

from __future__ import annotations

from typing import AsyncGenerator, List
import asyncio
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain.schema import HumanMessage
from sse_starlette.sse import EventSourceResponse

from ..config import Settings, get_settings


router = APIRouter(prefix="/api")

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

        user_state = {"messages": [HumanMessage(content=prompt)]}

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
