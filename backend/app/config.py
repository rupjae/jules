"""Application settings.

Values are sourced from environment variables (.env file loaded automatically
by python-dotenv when the application starts).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional import – the tests run in a minimal environment without external
# dependencies.  Gracefully degrade when *python-dotenv* is absent.
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover – fallback stub

    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False
try:
    from pydantic_settings import BaseSettings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – minimal dev/CI envs
    try:
        from pydantic import BaseSettings  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover – ultimate fallback

        class BaseSettings(object):  # type: ignore
            """Very small stub mimicking pydantic BaseSettings behaviour."""

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def dict(self):  # noqa: D401
                return self.__dict__
# ``Field`` is only a thin wrapper around default metadata; provide stub if
# Pydantic is unavailable.
try:
    from pydantic import Field  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – minimal envs

    def Field(default, *_, **__):  # type: ignore
        return default

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
    # Path to the persistent LangGraph checkpoint database. When the optional
    # ``langgraph-checkpoint-sqlite`` extra is **not** installed the value is
    # ignored and the backend silently falls back to an in-memory saver.
    checkpoint_db: str = Field("data/checkpoints.sqlite", env="JULES_CHECKPOINT_DB")

    class Config:
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
