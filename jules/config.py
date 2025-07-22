from __future__ import annotations

from pathlib import Path
import tomllib

_cfg_cache: dict | None = None


def get_agent_cfg() -> dict:
    """Return agent configuration loaded from ``config/agents.toml``."""
    global _cfg_cache
    if _cfg_cache is None:
        path = Path(__file__).parent.parent / "config" / "agents.toml"
        _cfg_cache = tomllib.loads(path.read_text())
    return _cfg_cache
