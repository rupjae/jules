import os
import sys
from pathlib import Path

import pytest

# Ensure backend modules resolve in tests without installing the package.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _set_log_level(caplog: pytest.LogCaptureFixture) -> None:
    """Keep test output concise by raising log level to INFO."""
    caplog.set_level("INFO")
