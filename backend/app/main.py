from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings

# Routers
from .routers import chat as chat_router


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
# NOTE: static mount must come *after* API routers so it doesn't shadow them.

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


# ---------------------------------------------------------------------------
# NOTE:
# The memory-aware chat endpoint previously lived here. It has been migrated to
# backend/app/routers/chat.py so we can keep all API routes in a single module
# and avoid accidental duplicates. The implementation in the router still
# relies on the same LangGraph graph object and continues to support per-thread
# memory.  This stub remains only to preserve the module-level helpers above.
# ---------------------------------------------------------------------------
# Mount built frontend (static export) *after* all API routes so paths like
# "/api/chat" are correctly resolved before the catch-all static files route.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
