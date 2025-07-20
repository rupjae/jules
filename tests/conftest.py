import os
import sys
from pathlib import Path

import pytest
import requests

# Ensure backend modules resolve in tests without installing the package.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture(scope="session")
def chroma_service(docker_ip, docker_services):
    """Start Chroma for integration tests."""
    docker_services.start("chroma")
    port = docker_services.port_for("chroma", 8000)

    def _ready() -> bool:
        try:
            r = requests.get(f"http://{docker_ip}:{port}/api/v1/heartbeat")
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    docker_services.wait_until_responsive(timeout=30.0, pause=0.5, check=_ready)
    os.environ["CHROMA_HOST"] = docker_ip
    os.environ["CHROMA_PORT"] = str(port)
    return f"http://{docker_ip}:{port}"
