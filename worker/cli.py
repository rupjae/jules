from __future__ import annotations

import os
import time
import typer

app = typer.Typer(add_completion=False)


@app.command()  # type: ignore[misc]
def run_worker(
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable TRACE logging")
) -> None:
    """Start the background job worker."""
    os.environ["JULES_DEBUG"] = "1" if debug else "0"

    if debug:
        time.sleep(3600)
    else:
        raise NotImplementedError("Worker implementation pending")


if __name__ == "__main__":
    app()
