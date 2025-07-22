# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added
- Green-field enforcement via CI and pre-commit.
- Streaming `/api/chat` exchanges are now persisted to Chroma after the
  assistant reply completes.
- SSE responses now emit tokens incrementally and write to Chroma concurrently.
  Closes WO-4.

### Changed
- `/api/chat/message` now accepts a JSON body. The query parameter version lives under `/api/chat/message/legacy` until v0.5.
- Allowed chat roles expanded to `user`, `assistant`, `system`, and `tool` with a 32k content limit.
- Default `CHROMA_HOST` is now `chroma` and set in `docker-compose.yml`.


### Fixed
- Agent now recalls prior turns when the same `X-Thread-ID` is reused.
- Chroma container health check now uses `nc` instead of missing `wget`.
- Conversation continues when the same `thread_id` is passed as a query parameter.
- Added `@types/uuid` to resolve TypeScript build error.

### Security
- Bumped `starlette` to 0.47.2 to address GHSA-2c2j-9gv5-cj73.

### Removed
- Postgres checkpoint backend and related configuration.
