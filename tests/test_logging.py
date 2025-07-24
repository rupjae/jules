from jules.logging import configure_logging, TRACE
import logging


def test_trace_level(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configure_logging(True)
    assert logging.getLogger().level == TRACE
