# Allow `import app.*` to resolve to `backend.app.*` for brevity.

from importlib import import_module as _import_module
import sys as _sys

if "app" not in _sys.modules:
    _sys.modules["app"] = _import_module(__name__ + ".app")

__all__ = [
    "app",
]
