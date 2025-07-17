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


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    # Basic auth token for protecting the chat endpoint.
    # Send `Authorization: Bearer <token>`.
    auth_token: Optional[str] = Field(None, env="JULES_AUTH_TOKEN")
    postgres_uri: str = Field(..., env="POSTGRES_URI")

    class Config:
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
