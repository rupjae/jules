from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from typing import AsyncGenerator
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status
from langchain.schema import HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from .config import Settings, get_settings
from .graphs.main_graph import graph


app = FastAPI(title="Jules API", version="0.1.0")

# Allow frontend origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount built frontend (static export) at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _authorize(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Validate optional bearer token."""
    if settings.auth_token is None:
        return

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or token != settings.auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )


@app.get("/api/chat")
async def chat_endpoint(
    request: Request, settings: Settings = Depends(get_settings)
):  # noqa: D401
    """Stream chat completions via SSE using per-thread memory."""

    await _authorize(request, settings)

    prompt: str = request.query_params.get("message", "")
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message is required",
        )

    thread_id = request.headers.get("X-Thread-ID") or str(uuid4())

    async def generator() -> AsyncGenerator[str, None]:
        state = {
            "messages": [
                SystemMessage(content="You are Jules, a helpful AI assistant."),
                HumanMessage(content=prompt),
            ]
        }

        async for step in graph.astream(
            state, {"configurable": {"thread_id": thread_id}}
        ):
            messages = step.get("llm", {}).get("messages")
            if messages:
                for ch in messages[-1].content:
                    yield ch
        yield "__END__"

    return EventSourceResponse(generator())
