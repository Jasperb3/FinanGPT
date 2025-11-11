# Testing & Harness Notes

## Running Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=.:src pytest -q
```

Pytest needs **both** the project root and `src/` on `sys.path` because:
- Production code imports modules via `src.*`.
- Wrapper modules at the repo root (e.g., `chat.py`, `ingest.py`) are the ones tests patch/mimic.

## Design Learnings

- Keep wrapper modules exporting the symbols tests expect so monkeypatching works without contacting real services.
- When reorganising modules (e.g., moving logic under `src/`), plan for test harness implications early—decide whether tests import from `src.*` or wrappers, and supply `conftest.py`/`PYTHONPATH` guidance accordingly.
- If you eventually want to run with just `PYTHONPATH=src`, restructure production imports to avoid `src.` prefixes or package the project as an installable distribution.
