from backend.app.config import Settings


def test_env_flag_true(monkeypatch):
    monkeypatch.setenv("JULES_DEBUG", "1")
    assert Settings().debug is True
