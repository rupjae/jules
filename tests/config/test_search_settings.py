"""Unit tests for search-related settings validation."""

import os

import pytest

from app.config import Settings


def _clear_env(keys: list[str]):
    """Helper to remove env vars after test to avoid leakage."""
    for k in keys:
        os.environ.pop(k, None)


def test_env_override():
    os.environ["SEARCH_TOP_K"] = "3"
    try:
        s = Settings()
        assert s.SEARCH_TOP_K == 3
    finally:
        _clear_env(["SEARCH_TOP_K"])


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_top_k_bounds(bad):
    os.environ["SEARCH_TOP_K"] = bad
    try:
        with pytest.raises(ValueError):
            Settings()
    finally:
        _clear_env(["SEARCH_TOP_K"])
