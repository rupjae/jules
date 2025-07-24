# Debugging Jules

Jules supports a global debug mode that lifts all services to TRACE level.
Enable it via the `JULES_DEBUG` environment variable or by using the
`docker-compose.debug.yml` overlay.

When debug mode is active the API server runs with `uvicorn --reload` and every
JSON line log entry includes a `code_path` field for easy grepping.

## Using the environment variable

```bash
JULES_DEBUG=1 docker compose up
```

## Using the overlay file

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up
```

## Log locations

Each container writes coloured console output and two log files inside
`/app/logs`. Old pairs beyond the latest ten are removed automatically.
When running locally you can grep for TRACE lines:

```bash
grep "\"level\": " logs/*.jsonl | grep TRACE
```

Docker retains up to 50 MB × 5 rotated log files per service when the overlay
is used.
