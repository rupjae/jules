# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Fixed
- Agent now recalls prior turns when the same `X-Thread-ID` is reused.
- Conversation continues when the same `thread_id` is passed as a query parameter.

### Removed
- Postgres checkpoint backend and related configuration.
