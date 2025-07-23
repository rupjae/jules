"""Lightweight loader for *config/agents.toml*.

The file is expected to live at project root so it is resolved relative to
``Path(__file__).parents[2]`` – i.e. *../..*.  This mirrors the layout we
use elsewhere (see ``backend/app/prompts``).

Two sections are currently recognised:

* ``[retrieval]`` – config for the lightweight **RetrievalAgent**.
* ``[jules]`` – config for the heavyweight responder.

Additional tables will be ignored so we can extend the spec without breaking
older binaries.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Mapping

import tomllib
from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "RetrievalCfg",
    "JulesCfg",
    "AgentsConfig",
    "get_agent_cfg",
]


# Resolve config path relative to package or via env override.

_ENV_PATH = os.getenv("JULES_AGENTS_TOML")

if _ENV_PATH:
    TOML_PATH = Path(_ENV_PATH).expanduser().resolve()
else:
    TOML_PATH = (Path(__file__).resolve().parent.parent / "config" / "agents.toml").resolve()


class RetrievalCfg(BaseModel):
    """Config block for the retrieval agent."""

    model: str = Field(..., description="OpenAI chat model id")
    k: int = Field(..., gt=0, description="Max Chroma results")
    summary_tokens: int = Field(
        150, gt=0, lt=4097, description="Token budget for cheat-sheet"
    )


class JulesCfg(BaseModel):
    """Config block for the heavyweight Jules responder."""

    model: str = Field(..., description="OpenAI chat model id")
    temperature: float = Field(0.3, ge=0, le=2)
    max_tokens: int = Field(1024, ge=16, le=4096)


class AgentsConfig(BaseModel):
    """Top-level config reflecting *agents.toml*."""

    retrieval: RetrievalCfg
    jules: JulesCfg

    @classmethod
    def load(cls, path: Path | None = None) -> "AgentsConfig":
        """Load the TOML file from *path* (defaults to *config/agents.toml*)."""

        p = path or TOML_PATH
        if not p.exists():
            raise FileNotFoundError(p)
        data: Mapping[str, Any]
        with p.open("rb") as fp:
            data = tomllib.load(fp)
        try:
            return cls.model_validate(data, strict=True)  # type: ignore[arg-type]
        except ValidationError as exc:  # pragma: no cover – shows in pytest output
            # Re-raise with a nicer path hint
            raise ValueError(f"Invalid agents.toml: {exc}") from exc


_CACHE: AgentsConfig | None = None


def _get_cfg() -> AgentsConfig:
    global _CACHE
    if _CACHE is None:
        _CACHE = AgentsConfig.load()
    return _CACHE


def get_agent_cfg(name: str) -> RetrievalCfg | JulesCfg:
    """Return subsection *name* (e.g. ``"retrieval"``).

    The result is typed – callers should cast appropriately:

    ```python
    retrieval = get_agent_cfg("retrieval")  # -> RetrievalCfg
    ```
    """

    cfg = _get_cfg()
    if hasattr(cfg, name):
        return getattr(cfg, name)
    raise KeyError(f"Unknown agent config '{name}'.")
