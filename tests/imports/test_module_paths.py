"""Guard tests for critical import paths."""


def test_settings_importable() -> None:  # noqa: D401
    """Ensure ``app.config`` is importable after refactors."""

    from importlib import import_module

    assert import_module("app.config")

