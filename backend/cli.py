from __future__ import annotations

import os
import typer

app = typer.Typer(add_completion=False)


@app.command()  # type: ignore[misc]
def run_server(
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable TRACE logging")
) -> None:
    """Start the FastAPI development server."""
    os.environ["JULES_DEBUG"] = "1" if debug else "0"
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=debug)


if __name__ == "__main__":
    app()
