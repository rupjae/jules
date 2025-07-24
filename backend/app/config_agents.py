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

# ---------------------------------------------------------------------------
# Locate the shared *config/agents.toml* file.
#
# When the project is executed straight from the source checkout the file sits
# exactly two levels above this module (project_root/config/agents.toml).  Once
# the package is installed – e.g. inside Docker where the code ends up in
# */app/app/* – that relative distance changes and the naïve resolution would
# incorrectly point at */config/agents.toml* (i.e. one level above the FS
# root).  To make the lookup resilient we walk up the directory tree until we
# find the first matching *config/agents.toml* file.  Should that search fail
# we fall back to the conventional *config/agents.toml* location under the
# current working directory so that unit-tests or one-off scripts can still run
# with a local config.
# ---------------------------------------------------------------------------


def _discover_agents_file() -> Path:
    """Return the first *config/agents.toml* found when walking upwards.

    Starting from the directory that contains *this* file we inspect every
    parent for a *config/agents.toml* file and return the first hit.  The
    algorithm guarantees termination at the filesystem root.  If no file is
    found we do **not** raise immediately to keep import-time side-effects
    minimal; instead we provide a sensible default under the current working
    directory and let the caller decide how to handle a potential
    *FileNotFoundError*.
    """

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "agents.toml"
        if candidate.is_file():
            return candidate

    # Fall back to *./config/agents.toml* relative to CWD.  This makes ad-hoc
    # invocations (e.g. `pytest`) behave intuitively when the repository root
    # is the working directory.
    return Path.cwd() / "config" / "agents.toml"


# Path to the shared TOML configuration file discovered at import time.
AGENTS_FILE = _discover_agents_file()


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

    try:
        text: str = AGENTS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Provide a minimal but functional default when the config file is not
        # shipped with the deployment artifact (e.g. Docker image that copied
        # only the *backend/* directory).  These defaults mirror the
        # repository-tracked *config/agents.toml* so that behaviour stays
        # consistent between local development and production unless the user
        # explicitly overrides the values via that TOML file.

        text = """
[retrieval]
model = "gpt-4o-mini"
k_hits = 5
cheat_tokens = 150

[jules]
model = "gpt-4o"
"""

    data: Any = tomllib.loads(text)
    return AgentsConfig(**data)  # type: ignore[arg-type]


__all__ = [
    "get_cfg",
    "AgentsConfig",
    "AgentCfg",
]
