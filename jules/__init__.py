"""Root package for Jules."""

# Re-export config helpers for convenience so downstream code can simply do:
# ``from jules import get_agent_cfg``.

from importlib import import_module as _imp

_cfg_mod = _imp("jules.config")

get_agent_cfg = _cfg_mod.get_agent_cfg  # type: ignore[attr-defined]
RetrievalCfg = _cfg_mod.RetrievalCfg
JulesCfg = _cfg_mod.JulesCfg

__all__ = [
    "get_agent_cfg",
    "RetrievalCfg",
    "JulesCfg",
]
