"""Regression tests for the *show_retrieval* query flag.

The chat streaming endpoint must *not* raise a ``KeyError`` when the flag is
absent and has to correctly accept common truthy string variants.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.app.routers import chat as chat_router


class _StubGraph:  # Minimal async-streaming stub used by the tests
    async def stream(self, state, config=None):  # noqa: D401, ANN001
        # Pretend the LLM answered "Hi" and always return an info-packet so the
        # retrieval event path is exercised.
        yield {"content": "Hi"}
        yield {"info_packet": "• fact A"}
        yield {"search_decision": True}


@pytest.mark.parametrize("flag_value", [None, "true", "1", "yes", "false"])
def test_show_retrieval_flag_does_not_500(flag_value: str | None) -> None:
    """Endpoint should return 200 regardless of the *show_retrieval* flag."""

    app = FastAPI()
    app.include_router(chat_router.router)

    # Inject stub LangGraph so the router can run without heavy deps or network
    # calls.
    app.state.graph = _StubGraph()  # type: ignore[attr-defined]

    with TestClient(app) as client:
        params: dict[str, str] = {"prompt": "test"}
        if flag_value is not None:
            params["show_retrieval"] = flag_value

        with client.stream("GET", "/api/chat/stream", params=params) as resp:
            # The streaming response returns HTTP 200 and yields lines over
            # time.  We only assert the status code – parsing the SSE stream
            # is covered elsewhere.  Exiting the context closes the
            # connection so the test does not hang on the infinite keep-alive
            # loop.
            assert resp.status_code == 200
