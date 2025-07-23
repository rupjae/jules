"""Application settings.

Values are sourced from environment variables (.env file loaded automatically
by python-dotenv when the application starts).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings  # type: ignore
from pydantic import Field  # type: ignore

# Load variables from a .env file if present
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime configuration."""

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    # Basic auth token for protecting the chat endpoint.
    # Send `Authorization: Bearer <token>`.
    auth_token: Optional[str] = Field(None, env="JULES_AUTH_TOKEN")
    # Path to the persistent LangGraph checkpoint database.
    checkpoint_db: str = Field("data/checkpoints.sqlite", env="JULES_CHECKPOINT_DB")

    # ---------------------------------------------------------------------
    # Vector search parameters (defaults tuned for small-scale demos).
    # ---------------------------------------------------------------------
    # Number of final results returned to callers.
    SEARCH_TOP_K: int = Field(8, env="SEARCH_TOP_K")
    # Oversampling factor when performing MMR – the actual number of vectors
    # fetched from the DB is k * oversample to give the algorithm more
    # candidates to choose from.
    SEARCH_MMR_OVERSAMPLE: int = Field(4, env="SEARCH_MMR_OVERSAMPLE")
    # λ trade-off parameter between similarity (1→purely relevance-based) and
    # diversity (0→purely novelty-based).
    SEARCH_MMR_LAMBDA: float = Field(0.5, env="SEARCH_MMR_LAMBDA")

    class Config:
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
