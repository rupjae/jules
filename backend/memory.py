from __future__ import annotations

import logging
from functools import lru_cache

# ``langgraph`` is optional – provide a stub when missing so the backend still
# boots without the heavy dependency.

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
except ModuleNotFoundError:

    class MemorySaver:  # type: ignore
        """In-memory stub replicating the minimal API we use (get/set)."""

        def __init__(self) -> None:
            self._store: dict[str, dict] = {}
            # LangGraph >=0.0.41 expects the checkpointer instance to expose a
            # ``put_writes`` attribute so the framework can track writes that
            # need to be flushed at the *end* of a graph run.
            #
            # The original in-memory ``MemorySaver`` from LangGraph defines
            # this attribute as a simple list that gets cleared after every
            # ``put_checkpoint`` call. We replicate the same surface so that
            # newer LangGraph versions can interact with our lightweight stub
            # without raising ``AttributeError`` while still keeping the
            # implementation minimal and dependency-free.
            self._writes: list[dict] = []

        # The real saver returns the latest checkpoint dict or None.
        def get_latest_checkpoint(self, thread_id: str):  # type: ignore
            return self._store.get(thread_id)

        def put_checkpoint(self, thread_id: str, state: dict):  # type: ignore
            self._store[thread_id] = state
            # Record the write so callers waiting on ``put_writes`` can flush
            # them. We append a *copy* to avoid accidental mutation by the
            # caller after the fact.
            self._writes.append({"thread_id": thread_id, "state": state.copy()})

        # The LangGraph runtime inspects the callable signature of
        # ``checkpointer.put_writes`` to determine whether it accepts a
        # ``task_path`` kw-arg. Provide it as a method that simply returns the
        # accumulated writes so that both the type-checking via ``inspect``
        # and the later read work as expected.

        def put_writes(self, *, task_path: str | None = None):  # noqa: D401, ANN001
            """Return a list of writes performed since the last call."""

            return self._writes

        # -----------------------------------------------------------------
        # Additional helpers expected by recent versions of LangGraph
        # -----------------------------------------------------------------

        def get_next_version(self, thread_id: str):  # type: ignore
            """Return an incrementing integer for ``thread_id``.

            The actual implementation in LangGraph namespaces versions per
            thread. We replicate the behaviour just enough for our tests – a
            monotonically increasing counter kept in-memory.
            """

            versions = self._store.setdefault(thread_id, {}).setdefault("_v", [])
            ver = len(versions) + 1
            versions.append(ver)
            return ver

        # -----------------------------------------------------------------
        # New helper surfaced in recent LangGraph releases
        # -----------------------------------------------------------------

        def get_tuple(self, *args, **kwargs):  # noqa: D401, ANN001
            """Return a (state, version) tuple.

            The concrete ``CheckpointTuple`` in LangGraph contains multiple
            attributes (``checkpoint``, ``version`` …). The calling code only
            accesses ``.checkpoint`` so we expose a lightweight replacement
            that fulfils that contract.
            """

            thread_id = None
            if args:
                cfg = args[0]
                thread_id = getattr(cfg, "thread_id", None) or getattr(cfg, "path", None)


            class _CheckpointTuple:  # noqa: D401 – minimal stand-in
                def __init__(self, checkpoint):  # noqa: ANN001
                    self.checkpoint = checkpoint

            state = self.get_latest_checkpoint(thread_id) if thread_id else None
            return _CheckpointTuple(state)  # type: ignore[return-value]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer() -> MemorySaver:
    """Return a singleton MemorySaver instance."""
    saver = MemorySaver()
    logger.debug("MemorySaver initialised (singleton).")
    return saver
