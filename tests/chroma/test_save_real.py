import os
import pytest
from db import chroma

pytestmark = pytest.mark.integration

if os.environ.get("ENABLE_INT_TESTS") != "1":
    pytest.skip("integration tests disabled", allow_module_level=True)


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "docker-compose.yml")


def test_save_real(docker_services):
    docker_services.start("chroma")
    port = docker_services.port_for("chroma", 8000)
    docker_services.wait_for_service("chroma", 8000)

    os.environ["CHROMA_HOST"] = "localhost"
    os.environ["CHROMA_PORT"] = str(port)

    msg = chroma.StoredMsg(thread_id="i1", role="user", content="hi")
    chroma.save_message(msg)
    assert chroma._get_collection().count() == 1

