# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Changed
* Added validated MMR tuning environment variables (`SEARCH_TOP_K`, `SEARCH_MMR_OVERSAMPLE`, `SEARCH_MMR_LAMBDA`) with sensible defaults.
* Bumped LangChain to `0.3.26`.

* BREAKING: `/api/chat/search` no longer returns `distance`. `similarity` field is now rounded to 4 decimals.
* Clients that need raw distance should compute it client-side or read it from earlier API versions.

### Added
- Green-field enforcement via CI and pre-commit.
- Streaming `/api/chat` exchanges are now persisted to Chroma after the
  assistant reply completes.
- SSE responses now emit tokens incrementally and write to Chroma concurrently.
  Closes WO-4.
- `/api/chat/search`
    - `thread_id` now optional (global search)
    - NEW: returns `similarity` (0-1) and accepts `min_similarity` filter

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
