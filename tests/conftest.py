import os
import sys
from pathlib import Path

# Ensure backend modules resolve in tests without installing the package.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
