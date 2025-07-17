"""Chat router for Jules.

Provides a `/chat` endpoint that streams tokens via Server-Sent Events (SSE).
"""

import asyncio
from typing import AsyncGenerator, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from ..config import Settings, get_settings


class SSECallbackHandler(AsyncCallbackHandler):
    """Captures streamed tokens and yields them through an async queue."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):  # type: ignore[override]
        await self.queue.put(token)

    async def done(self) -> None:
        await self.queue.put("__END__")


router = APIRouter(prefix="/api")


async def _authorize(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Simple bearer-token check if JULES_AUTH_TOKEN is configured."""

    if settings.auth_token is None:
        return  # auth disabled

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
    """Stream chat completions via SSE.

    Expects JSON body: { "message": "Hello Jules" }
    """

    await _authorize(request, settings)

    prompt: str = request.query_params.get("message", "")

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message is required",
        )

    # Async queue & callback for streaming tokens
    cb = SSECallbackHandler()

    async def generator() -> AsyncGenerator[str, None]:
        # Build conversation
        messages: List = [
            SystemMessage(content="You are Jules, a helpful AI assistant."),
            HumanMessage(content=prompt),
        ]

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=True,
            callbacks=[cb],
            openai_api_key=settings.openai_api_key,
        )

        # Run model concurrently so we can stream tokens
        async def _run_llm():
            await llm.agenerate([messages])
            await cb.done()

        task = asyncio.create_task(_run_llm())

        try:
            while True:
                token = await cb.queue.get()
                if token == "__END__":
                    break
                yield token
        finally:
            task.cancel()

    return EventSourceResponse(generator())
