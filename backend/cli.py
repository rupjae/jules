from __future__ import annotations

import os
import typer  # type: ignore[import]

app = typer.Typer(add_completion=False)


@app.command()
def run_server(
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable TRACE logging")
) -> None:
    """Start the FastAPI development server."""
    os.environ["JULES_DEBUG"] = "1" if debug else "0"
    import uvicorn  # type: ignore[import]

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=debug)


if __name__ == "__main__":
    app()
