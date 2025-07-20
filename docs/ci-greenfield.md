# Green-Field CI Enforcement

This project rejects any pull request that reintroduces legacy code or vulnerable dependencies.

## Local setup

1. Install pre-commit:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
2. Run all hooks manually with `pre-commit run --all-files` before pushing.

## What gets blocked?

- Any file or directory containing `legacy` or `shim`.
- Imports of `oldlib` or `jules.legacy`.
- Dependencies with known vulnerabilities or disallowed licenses.

If you hit a block, remove the offending code. Overrides are not supported—open an issue instead.
