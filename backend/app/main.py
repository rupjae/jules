from __future__ import annotations

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings
from jules.logging import configure_logging

# `memory.py` lives at project root.  Use absolute import so it works regardless
# of where the `app` package is located on the import path (Docker image copies
# only the *app* folder, not the whole backend tree).

from .checkpointer import get_checkpointer
from .graphs.next_gen import build_graph

# Routers
from .routers import chat as chat_router


settings = get_settings()
configure_logging(settings.debug)
app = FastAPI(title="Jules API", version="1.0.0")


async def _init_graph() -> None:
    app.state.checkpointer = get_checkpointer()
    app.state.graph = build_graph()


app.add_event_handler("startup", _init_graph)


# Allow frontend origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount built frontend (static export) at root
# NOTE: static mount must come *after* API routers so it doesn't shadow them.

# Register application routers
app.include_router(chat_router.router)


@app.get("/health")  # type: ignore[misc]
async def health() -> dict[str, str]:
    """Simple health endpoint for monitoring."""
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


# Mount built frontend (static export) *after* all API routes so paths like
# "/api/chat" are correctly resolved before the catch-all static files route.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
