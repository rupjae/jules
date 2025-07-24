"""Ensure Graph v5 can be disabled via the JULES_ENABLE_V5 env flag."""

from __future__ import annotations


import os
import pytest

import httpx


# We *must* set the env var *before* the FastAPI app (and thus the router
# module) is imported so the flag is picked up correctly.  Pytest guarantees
# module-level fixtures and code are evaluated in test order, so we mutate the
# environment first, then import the application.


@pytest.mark.anyio
async def test_disable_v5_flag(monkeypatch):
    """POST /api/chat with default (v5) version should fail when flag is off."""

    monkeypatch.setenv("JULES_ENABLE_V5", "false")

    # Import *after* env patch so the router sees the correct value.
    from backend.app.main import app  # noqa: WPS433 – local import for patch ordering

    # Patch graph builder to avoid heavy LangGraph construction / OpenAI calls.
    import backend.app.routers.chat as chat_router  # noqa: WPS433

    class _StubGraph:  # noqa: WPS110 – test helper
        async def ainvoke(self, *_args, **_kwargs):  # noqa: D401
            return {"reply": "stub"}

    def _fake_get_graph(version: str):  # noqa: D401
        return _StubGraph()

    monkeypatch.setattr(chat_router._graphs, "get_graph", _fake_get_graph, raising=True)

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        r = await client.post("/api/chat", json={"message": "hello"})

        assert r.status_code == 410

        # The dedicated v6 endpoint should still work fine (stubbed).
        ok = await client.post("/api/chat/v6", json={"message": "hi"})
        assert ok.status_code == 200
