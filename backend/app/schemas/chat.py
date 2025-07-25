from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, UUID4


class ChatMessageIn(BaseModel):  # type: ignore[misc]
    """Incoming Pydantic chat message payload."""

    thread_id: UUID4
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(..., max_length=32000)
