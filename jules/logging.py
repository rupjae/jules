"""Logging helpers with TRACE level support."""

from __future__ import annotations

import functools
import logging
from typing import Callable, ParamSpec, TypeVar

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


__all__ = ["TRACE", "trace"]
