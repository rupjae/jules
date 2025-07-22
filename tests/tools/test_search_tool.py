from __future__ import annotations

import httpx
import pytest
import respx

from jules.tools import ChromaSearchTool


FIXTURE = [
    {"text": "dummy 1", "similarity": 0.92, "role": "user", "ts": 1},
    {"text": "dummy 2", "similarity": 0.90, "role": "assistant", "ts": 2},
]


@pytest.mark.anyio("asyncio")
@respx.mock
async def test_happy_path() -> None:
    route = respx.get("http://localhost:8000/api/chat/search").mock(
        return_value=httpx.Response(200, json=FIXTURE)
    )
    tool = ChromaSearchTool()
    results = await tool("foo", k=2)
    assert route.called
    assert len(results) == 2
    assert results[0].similarity == 0.92


@pytest.mark.anyio("asyncio")
@respx.mock
async def test_thread_filter_param() -> None:
    respx.get("http://localhost:8000/api/chat/search").mock(
        return_value=httpx.Response(200, json=FIXTURE)
    )
    tool = ChromaSearchTool()
    await tool("foo", thread_id="abc")
    last_request = respx.calls.last.request
    assert "thread_id=abc" in str(last_request.url)


@pytest.mark.anyio("asyncio")
@respx.mock
async def test_http_error() -> None:
    respx.get("http://localhost:8000/api/chat/search").mock(
        return_value=httpx.Response(404)
    )
    tool = ChromaSearchTool()
    with pytest.raises(httpx.HTTPStatusError):
        await tool("oops")
