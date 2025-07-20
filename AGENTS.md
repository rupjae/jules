# AGENTS Coding Guidelines (v0.2 – June 2025)

*This file is the single source of truth for contributor‑facing rules.  
If anything here conflicts with code comments, **this document wins**.*

---

## 1 · Logging

| Target | Handler | Format |
|--------|---------|--------|
| Console | `rich.logging.RichHandler` | *Rich default (timestamp, coloured level, file:line)* |
| File (`logs/tradingbot‑YYYYMMDD‑HHMMSS.log`) | `logging.FileHandler` | `%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d - %(message)s` |
| JSONL (`logs/tradingbot‑YYYYMMDD‑HHMMSS.jsonl`) | custom `JsonLinesHandler` | keys: `ts_epoch`, `level`, `logger`, `msg`, `code_path`, *(opt)* `trace_id`, `exc_type`, `exc_msg` |

* **`code_path` is mandatory** on every record—either pass it in `extra=` or rely on the `@trace` decorator.  
* Keep the latest **10** `.log` + `.jsonl` pairs; `configure_logging()` auto‑purges older ones.  
* A custom **TRACE** level (`5`) powers `@trace` entry/exit logs. `--debug` lifts root to TRACE, `--debug‑module foo.bar` raises just that tree to DEBUG.

---

## 2 · Testing

* All new code ships with unit tests (`pytest`, no `unittest`).
* Flaky sleep‑based assertions are forbidden—use `anyio.abc.Clock`.
* Golden‑file fixtures live under `tests/fixtures/…`; keep them small (<25 kB).

---

## 3 · Documentation

* Update `README.md` and **any affected `docs/…` files** in the same PR.
* Public‑facing modules carry docstrings; internal helpers do so when non‑trivial.

---

## 4 · Versioning & Releases

### 4.1 Bumping  
```bash
poetry version {patch|minor|major|<exact>}
```
### 4.2 In‑code access

from importlib.metadata import version
__version__ = version("tradingbot")


⸻

## 5 · Changelog

Follow Keep a Changelog.
Move items from Unreleased → dated section on release; bump pyproject.toml too.

⸻

## 6 · Coding Guidelines

### 6.1 Agent Design
	•	Prefer composition over inheritance.
	•	Avoid speculative abstractions—YAGNI.

### 6.2 Project Conventions
	•	pathlib.Path for all FS ops.
	•	CLI via typer, never raw argparse / click.
	•	Full type hints; pass --strict mypy in CI.

### 6.3 Style
	•	black & ruff clean.
	•	f‑strings only; no % or .format.
	•	Docstrings on every public function.

⸻

## 7 · Exception Handling
	•	Catch and log at async boundaries; include contextual metadata (symbol, broker, …).
	•	No silent pass; log with exc_info=True.
	•	Fail fast on non‑recoverables, graceful shutdown elsewhere.

⸻

## 8 · Dependencies
	•	Pin exact versions in pyproject.toml.
	•	poetry export --without-hashes | pip-audit -r - before merge.
	•	No new libs without approval.

⸻

## 9 · CLI UX
	•	Every option has --help.
	•	Default execution is side‑effect‑free unless explicitly overridden.
	•	Provide safe defaults for --symbol, --broker, --debug.

⸻

## 10 · LLM Integration

### 10.1 OpenAI Endpoint
We use the v2 “/responses” endpoint via tradingbot.openai.OpenAIAsyncClient.
Legacy helpers are gone; do not call OpenAI directly.

### 10.2 Builder Helpers
Always construct message / tool blocks with tradingbot.openai.builders.* to avoid drift.

### 10.3 Prompt Hygiene
	•	Keep prompts short, grounded, and schema‑validated.
	•	Log raw + parsed responses at DEBUG when TB_LOG_LEVEL=TRACE.

⸻

## 11 · External API References

Docs live under api_docs/{openai_api_docs|bitget_api_docs}.
When touching an API:
	1.	Cite the doc section in your PR.
	2.	Add/update typed models + tests using provider examples.
	3.	Commit spec diffs alongside code.

### 11.1 Updating Vendored Docs
	•	OpenAI – pull latest OpenAPI spec, regen Markdown with widdershins.
	•	Bitget – replace Markdown dump from portal; note revision date.

⸻

## 12 · When Unsure

Match existing patterns; otherwise propose via PR comment or minimal stub.

⸻

## 13 · Legacy Code
        •       This is a green‑field project. Avoid compatibility shims or
                fallbacks for missing dependencies.
        •       Remove or reject code meant solely to support retired APIs or
                behaviour.

⸻

Last edited: 2025‑07‑20

