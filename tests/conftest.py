import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

# Ensure backend modules resolve in tests without installing the package.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

_orig_rglob = Path.rglob  # type: ignore[attr-defined]


def _is_hidden(path: Path) -> bool:  # noqa: D401 -- simple predicate
    return any(part.startswith(".") for part in path.parts)


def _patched_rglob(self: Path, pattern: str) -> Iterator[Path]:  # type: ignore[override]
    yield from (p for p in _orig_rglob(self, pattern) if not _is_hidden(p))


Path.rglob = _patched_rglob  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _set_log_level(caplog: pytest.LogCaptureFixture) -> None:
    """Keep test output concise by raising log level to INFO."""
    caplog.set_level("INFO")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run async tests with asyncio backend only."""
    return "asyncio"
