# Changelog

All notable changes to this project will be documented in this file.

## Unreleased


## [1.0.1] – 2025-07-24

### Fixed
* GET-based SSE endpoint for browser compatibility.
* Missing `search` key in `ChatState` TypedDict.
* Correct SSE framing for `info_packet` event.

### Added
* Configurable RetrievalAgent heuristics.
* Extra unit tests for search gating.

## [1.0.0] – 2025-07-24

### Added
* Retrieval-aware two-stage RAG pipeline (`backend.app.graphs.next_gen`).
* Config-driven `config/agents.toml` with model + search parameters.
* Info-packet UI toggle (frontend) plus SSE `info_packet` event.

### Removed
* All legacy *graph_v* code, docs, and tests.


### Changed
* Added validated MMR tuning environment variables (`SEARCH_TOP_K`, `SEARCH_MMR_OVERSAMPLE`, `SEARCH_MMR_LAMBDA`) with sensible defaults.
* Pinned LangChain monolith to `0.3.26` (includes MMR); removed duplicate `langchain-core`/`langchain-community` pins. `langchain-openai` kept as it houses `ChatOpenAI`.

* **Vector retrieval** – switched `/api/chat/search` to **Max-Marginal Relevance (MMR)**. When LangChain is unavailable the code falls back to the previous dense-KNN + de-dup logic. The `where` filter and `SEARCH_*` tuning knobs are unchanged.

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
- Corrected settings import path from `backend.app.config` to `app.config` across
  the codebase and added a regression test.
- Agent now recalls prior turns when the same `X-Thread-ID` is reused.
- Chroma container health check now uses `nc` instead of missing `wget`.
- Conversation continues when the same `thread_id` is passed as a query parameter.
- Added `@types/uuid` to resolve TypeScript build error.

### Security
- Bumped `starlette` to 0.47.2 to address GHSA-2c2j-9gv5-cj73.

### Removed
- Postgres checkpoint backend and related configuration.
