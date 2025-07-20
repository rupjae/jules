from pathlib import Path

BANNED = [
    "ModuleNotFoundError",
    "langchain_community",
    "compatibility shim",
    "MemorySaver",
    "fallback",
]


def test_no_legacy_code() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for py in repo_root.rglob("*.py"):
        if "tests" in py.parts:
            continue
        text = py.read_text()
        for pattern in BANNED:
            assert pattern not in text, f"Found legacy pattern '{pattern}' in {py}"
