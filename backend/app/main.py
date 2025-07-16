from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Serve static frontend
from fastapi.staticfiles import StaticFiles

from .routers import chat


app = FastAPI(title="Jules API", version="0.1.0")

# Allow frontend origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)

# Mount built frontend (static export) at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}
