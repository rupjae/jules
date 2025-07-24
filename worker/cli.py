from __future__ import annotations

import os
import typer  # type: ignore[import]

app = typer.Typer(add_completion=False)


@app.command()
def run_worker(
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable TRACE logging")
) -> None:
    """Start the background job worker."""
    os.environ["JULES_DEBUG"] = "1" if debug else "0"
    import time

    if debug:
        time.sleep(3600)
    else:
        raise NotImplementedError("Worker implementation pending")


if __name__ == "__main__":
    app()
