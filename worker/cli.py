import os
import typer

app = typer.Typer(add_completion=False)


@app.command()
def run_worker(
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable TRACE logging")
) -> None:
    """Start the background job worker."""
    os.environ["JULES_DEBUG"] = "1" if debug else "0"
    # Placeholder for actual worker start logic
    while True:
        pass


if __name__ == "__main__":
    app()
