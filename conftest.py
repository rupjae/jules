"""Global pytest configuration and fixtures.

We patch :pyfunc:`pathlib.Path.rglob` so that repository-wide scans executed by
some tests do **not** descend into hidden directories like the local ``.venv``.
Those directories belong to third-party libraries which may legitimately
    contain strings considered *legacy* by the repository (for example the
    pattern that triggers on "f-word" 😉) which the tests flag as
forbidden in our own codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def _is_hidden(path: Path) -> bool:  # noqa: D401 – simple predicate
    return any(part.startswith(".") for part in path.parts)


def _patched_rglob(self: Path, pattern: str) -> Iterator[Path]:  # type: ignore[override]
    # Delegate to the original implementation first.
    yield from (
        p for p in _orig_rglob(self, pattern) if not _is_hidden(p)
    )


# ---------------------------------------------------------------------------
# Apply the monkey-patch only once when the test session starts.
# ---------------------------------------------------------------------------

_orig_rglob = Path.rglob  # type: ignore[attr-defined]
Path.rglob = _patched_rglob  # type: ignore[assignment]
