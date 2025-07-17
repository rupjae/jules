"""Chat router exposing the memory-aware LangGraph endpoint.

The router is mounted with prefix "/api" so the final path is /api/chat.
It streams tokens via Server-Sent Events (SSE) and keeps per-thread
conversation context using the configured LangGraph checkpoint saver.
"""

from __future__ import annotations

from typing import AsyncGenerator, List
import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain.schema import HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from ..config import Settings, get_settings
from ..graphs.main_graph import graph


router = APIRouter(prefix="/api")


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

    # Use header-supplied thread id or create one.
    thread_id = request.headers.get("X-Thread-ID") or str(uuid4())

    async def generator() -> AsyncGenerator[str, None]:
        """Run the (blocking) graph in a background thread and stream tokens."""

        state = {
            "messages": [
                SystemMessage(content="You are Jules, a helpful AI assistant."),
                HumanMessage(content=prompt),
            ]
        }

        loop = asyncio.get_running_loop()

        def _run_sync() -> List[str]:
            """Execute graph.stream and collect the assistant's latest message."""
            latest_tokens: List[str] = []
            for step in graph.stream(state, {"configurable": {"thread_id": thread_id}}):
                messages: List | None = step.get("llm", {}).get("messages")
                if messages:
                    latest_tokens = list(messages[-1].content)
            return latest_tokens

        # Run the blocking graph in default executor
        tokens: List[str] = await loop.run_in_executor(None, _run_sync)
        for ch in tokens:
            yield ch

    # Attach the (possibly freshly generated) thread id so clients can persist it.
    return EventSourceResponse(generator(), headers={"X-Thread-ID": thread_id})
