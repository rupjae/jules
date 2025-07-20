from pathlib import Path

FORBIDDEN = ["legacy", "shim"]
ALLOWED = {
    ".github/workflows/no-legacy.yml",
    "tests/ci/test_no_legacy.py",
    "tests/test_no_legacy_patterns.py",
}


def test_no_legacy_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in repo_root.rglob("*"):
        rel = str(path.relative_to(repo_root))
        if rel in ALLOWED:
            continue
        if "tests" in path.parts:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        text = str(path)
        for token in FORBIDDEN:
            assert (
                token not in text.lower()
            ), f"Forbidden path segment '{token}' in {path}"
