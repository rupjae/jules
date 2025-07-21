# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added
- Green-field enforcement via CI and pre-commit.

### Changed
- `/api/chat/message` now accepts a JSON body. The query parameter version lives under `/api/chat/message/legacy` until v0.5.
- Default `CHROMA_HOST` is now `chroma` and set in `docker-compose.yml`.


### Fixed
- Agent now recalls prior turns when the same `X-Thread-ID` is reused.
- Conversation continues when the same `thread_id` is passed as a query parameter.
- Added `@types/uuid` to resolve TypeScript build error.

### Removed
- Postgres checkpoint backend and related configuration.
