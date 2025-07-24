"""Logging helpers with TRACE level support."""

from __future__ import annotations

import functools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, Tuple, cast

from rich.logging import RichHandler

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

P = ParamSpec("P")
R = TypeVar("R")


def trace(func: Callable[P, R]) -> Callable[P, R]:
    """Log entry and exit of *func* at TRACE level."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = logging.getLogger(func.__module__)
        path = f"{func.__module__}.{func.__name__}"
        logger.log(TRACE, "-> %s", path, extra={"code_path": path})
        try:
            return func(*args, **kwargs)
        finally:
            logger.log(TRACE, "<- %s", path, extra={"code_path": path})

    return wrapper


class JsonLinesHandler(logging.Handler):
    """Write log records as JSON lines."""

    STANDARD_KEYS = {
        "ts_epoch",
        "level",
        "logger",
        "msg",
        "code_path",
        "trace_id",
        "exc_type",
        "exc_msg",
    }

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._fh = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        payload = {
            "ts_epoch": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "code_path": getattr(record, "code_path", record.pathname),
        }
        if record.exc_info:
            exc_type, exc_val, _ = cast(
                Tuple[type[BaseException], BaseException, object], record.exc_info
            )
            payload.update({"exc_type": exc_type.__name__, "exc_msg": str(exc_val)})
        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            payload["trace_id"] = trace_id
        for key, value in record.__dict__.items():
            if key not in payload and key not in {
                "args",
                "msg",
                "exc_info",
                "levelno",
                "levelname",
                "name",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                payload[key] = value

        json.dump(payload, self._fh)
        self._fh.write("\n")
        self._fh.flush()


def _purge_old_logs(directory: Path, keep: int = 10) -> None:
    pairs = sorted(directory.glob("jules-*.log"))
    excess = len(pairs) - keep
    if excess <= 0:
        return
    for path in pairs[:excess]:
        jsonl = path.with_suffix(".jsonl")
        path.unlink(missing_ok=True)
        jsonl.unlink(missing_ok=True)


def configure_logging(debug: bool = False) -> Path:
    """Configure console + file logging and return the log file path."""

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    existing = cast(str | None, getattr(root, "_configured", None))
    if existing:
        return Path(existing)

    _purge_old_logs(log_dir)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"jules-{ts}.log"
    jsonl_path = log_path.with_suffix(".jsonl")

    level = TRACE if debug else logging.INFO
    root.setLevel(level)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d - %(message)s"
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(log_path, encoding="utf-8"),
        JsonLinesHandler(jsonl_path),
    ]

    for h in handlers:
        h.setLevel(level)
        if not isinstance(h, JsonLinesHandler):
            h.setFormatter(logging.Formatter(fmt))
        root.addHandler(h)

    root._configured = str(log_path)
    return log_path


__all__ = ["TRACE", "trace", "configure_logging"]
