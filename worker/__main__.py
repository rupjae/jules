from __future__ import annotations

import logging

from backend.app.config import Settings
from jules.logging import configure_logging

from .cli import app


def main() -> None:
    """Entry point for the worker CLI."""
    configure_logging(Settings().debug)
    logging.getLogger("rq.worker").setLevel(logging.getLogger().level)
    app()


if __name__ == "__main__":
    main()
