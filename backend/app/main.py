from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings

# `memory.py` lives at project root.  Use absolute import so it works regardless
# of where the `app` package is located on the import path (Docker image copies
# only the *app* folder, not the whole backend tree).

from .checkpointer import get_checkpointer
from .graphs.main_graph import build_graph

# Routers
from .routers import chat as chat_router


app = FastAPI(title="Jules API", version="0.1.0")


@app.on_event("startup")
async def _init_graph() -> None:
    app.state.checkpointer = get_checkpointer()
    app.state.graph = build_graph()


# Allow frontend origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount built frontend (static export) at root *only when the directory exists*
# This guards the test environment where the compiled assets are absent.

import os


if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Register application routers
app.include_router(chat_router.router)


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


# Mount built frontend (static export) *after* all API routes so paths like
# "/api/chat" are correctly resolved before the catch-all static files route.
# Duplicate mount removed – the conditional mount above handles it.
