"""Application settings.

Values are sourced from environment variables (.env file loaded automatically
by python-dotenv when the application starts).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load variables from a .env file if present
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime configuration.

    The OpenAI key is *optional* in local-dev / CI scenarios so that the
    backend can start without external credentials.  When the key is missing
    we fall back to a stubbed LLM that simply echoes the user input.  This
    keeps the entire stack functional – the frontend receives a response –
    while still enforcing the presence of a real key in production (see the
    guard in :pyfunc:`langgraph` node).
    """

    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    # Basic auth token for protecting the chat endpoint.
    # Send `Authorization: Bearer <token>`.
    auth_token: Optional[str] = Field(None, env="JULES_AUTH_TOKEN")

    class Config:
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
