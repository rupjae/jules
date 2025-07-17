"""Re-export get_checkpointer so code can import it either as:

    from memory import get_checkpointer          # production / Docker path
    from app.memory import get_checkpointer      # local testing

The indirection avoids brittle import errors when the project layout changes
between the source tree (monorepo) and the Docker image (only ``backend/app``
gets copied).
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast


# ---------------------------------------------------------------------------
# Attempt to import the top-level ``memory`` module first (present in the repo
# root and explicitly copied into the Docker image). Fallback to a local stub
# when it doesn’t exist (e.g., in certain test runners).
# ---------------------------------------------------------------------------

try:
    _memory: ModuleType = import_module("memory")
except ModuleNotFoundError:  # pragma: no cover – only hit in edge cases

    class _Stub:
        """Minimal drop-in replacement so the app can still start."""

        def get_checkpointer(self) -> Any:  # noqa: ANN401 – dynamic
            from functools import lru_cache

            @lru_cache(maxsize=1)
            def _stub():  # noqa: D401
                class _DummySaver:  # noqa: D401 – minimal interface
                    """In-memory stand-in for LangGraph MemorySaver."""

                    def __init__(self) -> None:  # noqa: D401
                        self._store: dict[str, dict] = {}
                        # Keep track of writes so newer LangGraph versions can
                        # introspect them at the end of the execution loop.
                        self._writes: list[dict] = []

                    # Expected by graph execution ---------------------------
                    def get_latest_checkpoint(self, thread_id: str):  # noqa: D401
                        return self._store.get(thread_id)

                    def put_checkpoint(self, thread_id: str, state: dict):  # noqa: D401
                        self._store[thread_id] = state
                        # Mirror behaviour of the real MemorySaver to satisfy
                        # the framework's expectations.
                        self._writes.append({"thread_id": thread_id, "state": state.copy()})

                    # The LangGraph runtime expects a *callable* attribute
                    # ``put_writes`` – not a list – and uses
                    # ``inspect.signature`` to check whether it accepts a
                    # ``task_path`` keyword. Provide a pass-through method that
                    # meets those requirements.

                    def put_writes(  # type: ignore[override]  # noqa: D401
                        self, *, task_path: str | None = None
                    ) -> list[dict]:
                        """Return the list of stored write records."""

                        return self._writes

                    def get_next_version(self, thread_id: str):  # noqa: D401
                        versions = self._store.setdefault(thread_id, {}).setdefault("_v", [])
                        ver = len(versions) + 1
                        versions.append(ver)
                        return ver
                    def get_tuple(self, *args, **kwargs):  # noqa: D401, ANN001
                        """Return minimal object exposing `.checkpoint`."""

                        class _CheckpointTuple:  # noqa: D401 – internal helper
                            def __init__(self, checkpoint):  # noqa: ANN001
                                self.checkpoint = checkpoint

                        thread_id = None
                        if args:
                            cfg = args[0]
                            thread_id = getattr(cfg, "thread_id", None) or getattr(cfg, "path", None)
                        state = self.get_latest_checkpoint(thread_id) if thread_id else None
                        return _CheckpointTuple(state)  # type: ignore[return-value]

                    # Newer LangGraph versions inspect this list after the
                    # graph run; we initialise it in __init__.

                return _DummySaver()

            return _stub()

    _memory = cast(ModuleType, _Stub())


# Public re-exports ----------------------------------------------------------

get_checkpointer = _memory.get_checkpointer  # type: ignore[attr-defined]
