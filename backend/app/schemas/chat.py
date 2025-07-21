from __future__ import annotations

from pydantic import BaseModel, Field, UUID4


class ChatMessageIn(BaseModel):
    """Incoming chat message payload."""

    thread_id: UUID4
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=8000)
