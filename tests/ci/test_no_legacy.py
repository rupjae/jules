from pathlib import Path

FORBIDDEN = ["legacy", "shim"]


def test_no_legacy_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in repo_root.rglob("*"):
        if "tests" in path.parts:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        text = str(path)
        for token in FORBIDDEN:
            assert (
                token not in text.lower()
            ), f"Forbidden path segment '{token}' in {path}"
