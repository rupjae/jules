import logging
import os

from jules.logging import configure_logging

from .cli import app


def main() -> None:
    configure_logging(debug=os.getenv("JULES_DEBUG") in {"1", "true", "yes"})
    logging.getLogger("rq.worker").setLevel(logging.getLogger().level)
    app()


if __name__ == "__main__":
    main()
