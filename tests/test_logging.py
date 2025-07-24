import json
import logging

from rich.logging import RichHandler

from jules.logging import TRACE, configure_logging


def test_trace_level(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    configure_logging(True)
    path = configure_logging(True)
    root = logging.getLogger()
    assert root.level == TRACE
    rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 1
    logger = logging.getLogger("t")
    logger.info("hi", extra={"code_path": "t"})
    json_path = path.with_suffix(".jsonl")
    line = json_path.read_text().splitlines()[0]
    payload = json.loads(line)
    for key in ["ts_epoch", "level", "logger", "msg", "code_path"]:
        assert key in payload and payload[key] != ""
