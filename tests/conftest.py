import os
import sys
from pathlib import Path

# Ensure backend modules resolve in tests without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
