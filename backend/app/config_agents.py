"""Load agent configuration from config/agents.toml.

The file lives at project_root/config/agents.toml so we resolve the path two
levels up from *this* file (backend/app/ → backend/ → project root).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# TOML loader – stdlib `tomllib` is available from Python 3.11 which this
# project already targets (see *pyproject.toml*).  We still fall back to the
# external *tomli* package when running under an older interpreter to keep the
# module import-safe in ancillary tooling that might use a different Python.
# ---------------------------------------------------------------------------

try:
    import tomllib  # type: ignore
except ImportError:  # pragma: no cover – Python < 3.11
    import tomli as tomllib  # type: ignore

# Path to the shared TOML configuration file.
AGENTS_FILE = Path(__file__).resolve().parents[2] / "config" / "agents.toml"


class AgentCfg(BaseModel):
    """Configuration for a single agent."""

    model: str
    k_hits: int | None = Field(default=None, alias="k_hits")
    cheat_tokens: int | None = None


class AgentsConfig(BaseModel):
    """Full agents configuration tree."""

    retrieval: AgentCfg
    jules: AgentCfg


@lru_cache(maxsize=1)
def get_cfg() -> AgentsConfig:
    """Return cached AgentsConfig parsed from *config/agents.toml*."""

    text: str = AGENTS_FILE.read_text(encoding="utf-8")
    data: Any = tomllib.loads(text)
    return AgentsConfig(**data)  # type: ignore[arg-type]


__all__ = [
    "get_cfg",
    "AgentsConfig",
    "AgentCfg",
]
