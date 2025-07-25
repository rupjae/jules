"""Application settings.

Values are sourced from environment variables (.env file loaded automatically
by python-dotenv when the application starts).
"""

from __future__ import annotations

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


class Settings(BaseSettings):  # type: ignore[misc]
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
    # Vector search configuration ------------------------------------------------
    SEARCH_TOP_K: int = Field(8, ge=1, env="SEARCH_TOP_K")
    # Oversampling factor – we first fetch TOP_K * oversample candidates for MMR.
    SEARCH_MMR_OVERSAMPLE: int = Field(4, ge=1, env="SEARCH_MMR_OVERSAMPLE")
    # λ ∈ [0,1] – 0→novelty-only, 1→relevance-only.
    SEARCH_MMR_LAMBDA: float = Field(0.5, ge=0.0, le=1.0, env="SEARCH_MMR_LAMBDA")

    debug: bool = Field(False, validation_alias="JULES_DEBUG")

    class Config:
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
