# Jules 🗣️🤖

[![Keeping Jules Green](https://img.shields.io/badge/Keeping%20Jules%20Green-docs-green)](docs/ci-greenfield.md)

An AI-powered chatbot built with LangChain + LangGraph on a FastAPI backend and a Next.js (React, TypeScript, Material-UI) frontend.

## Project layout

```
.
├── backend/          # FastAPI + LangChain services
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   └── routers/
│   │       └── chat.py
│   ├── Dockerfile
│   └── (Python dependencies managed via Poetry)
├── frontend/         # Next.js (TypeScript) client
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── public/
│   ├── src/
│   │   ├── pages/
│   │   │   └── index.tsx
│   │   └── components/
│   │       └── Chat.tsx
│   └── Dockerfile
├── docker-compose.yml
└── .env              # OpenAI key and config (never commit)
```

## Quick start (development)

1. Copy `.env.example` → `.env` and fill in:
   - `OPENAI_API_KEY=<your OpenAI key>` (required)
   - `JULES_AUTH_TOKEN=<optional bearer token>` (commented out by default; leave unset to disable auth)
   - `CHROMA_HOST=chroma` target Chroma service host
 - `CHROMA_TIMEOUT_MS=100` optional request timeout
  - **Vector search parameters** (optional – leave defaults unless you need to tune)
    - `SEARCH_TOP_K` (default `8`): number of documents returned by `/api/chat/search`.
    - `SEARCH_MMR_OVERSAMPLE` (default `4`): when using Maximal Marginal Relevance, we first fetch `TOP_K * OVERSAMPLE` candidates and then re-rank.
    - `SEARCH_MMR_LAMBDA` (default `0.5`): trade-off between relevance (`1.0`) and novelty (`0.0`).
2. Install Python deps & run backend:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install poetry
poetry install --no-interaction --no-root
uvicorn app.main:app --reload --port 8000
```

Conversation history is stored in a SQLite database under `data/` so it
persists across restarts.

Then open http://localhost:8000 to chat with **Jules**.

### API Endpoints
• **GET /api/chat?thread_id=<id>&message=<text>**
  Streams chat completions via Server-Sent Events.  On the first request for a new thread,
  the `X-Thread-ID` response header provides the generated session ID.  Include the same
  `thread_id` (query or `X-Thread-ID` header) on subsequent calls to continue the conversation.
  Once the assistant reply finishes streaming, both the user prompt and the response are
  automatically persisted to SQLite and Chroma so they appear in subsequent searches.
• **POST /api/chat/message**
  Persist a single chat message to SQLite and Chroma. Accepts a JSON body `{"thread_id": "<uuid>", "role": "user|assistant|system|tool", "content": "<text>"}` (max 32k chars).
  The legacy query-parameter variant remains available at `/api/chat/message/legacy` and will be removed after v0.5.
  Example:
  ```bash
  curl -X POST \
    -H 'Content-Type: application/json' \
    -d '{"thread_id":"123e4567-e89b-12d3-a456-426614174000","role":"user","content":"hi"}' \
    http://localhost:8000/api/chat/message
  ```
• **GET /api/chat/history?thread_id=<id>**
  Returns the full conversation history as JSON:
  ```
  > NOTE: Both `/api/chat/history` and `/api/chat/search` responses now include a `"timestamp"` field (ISO-8601 UTC) for each message.
  [
    {"sender": "user",      "content": "Hello!"},
    {"sender": "assistant", "content": "Hi there!"},
    ...
  ]
  ```
• **GET /api/chat/search?thread_id=<id>&query=<text>**
Vector similarity search backed by Chroma. Omit `thread_id` for a global
search. Results are chosen via **Max-Marginal Relevance (MMR)** to balance
relevance and diversity (configurable through `SEARCH_MMR_*` settings). When
LangChain is not installed the server transparently falls back to the previous
dense K-NN search with light de-duplication.  Each hit includes a `similarity`
score (0–1, higher means closer) rounded to 4 decimals and may be filtered via
`min_similarity`. Example:

```
/api/chat/search?query=hello&min_similarity=0.8
```

Similarity ranges 0–1 (higher = closer).

Returns `[{'text': 'hello', 'similarity': 0.6476,
'timestamp': 1620000000.0, 'role': 'user'}]`.
The backend will return the generated `X-Thread-ID` header on the very first
request so that clients can persist it.  Subsequent calls should repeat the
same ID either via the `X-Thread-ID` header or as a `thread_id` query
parameter to continue the conversation thread.

## Docker (single image)

```bash
docker compose up --build
```

Open your browser:

• http://localhost:3000  → Next.js dev UI with hot-reload
• http://localhost:8000  → FastAPI API (and the static build when dev server is stopped)
• http://chroma:8000  → Chroma vector database HTTP API (Docker network)

The compose file mounts `backend/app`, `db`, and `jules` into the container so
any local changes reload automatically.

## MMR search settings

Jules uses *Maximal Marginal Relevance* (MMR) re-ranking to avoid returning
near-duplicate chunks when answering questions. Three environment variables let
you fine-tune the behaviour without touching code:

| Variable | Type & Bounds | Default | Description |
|----------|--------------|---------|-------------|
| `SEARCH_TOP_K` | integer ≥ 1 | `8` | Final number of documents returned to the caller. |
| `SEARCH_MMR_OVERSAMPLE` | integer ≥ 1 | `4` | We first fetch `TOP_K × OVERSAMPLE` candidates, then run MMR to pick the best *TOP_K*. |
| `SEARCH_MMR_LAMBDA` | float 0-1 | `0.5` | Trade-off between relevance (`1.0`) and novelty (`0.0`). |

All three variables are validated at startup using Pydantic. Supplying an
out-of-range value (e.g. `SEARCH_MMR_LAMBDA=1.3`) aborts launch with a clear
error so mis-configured deployments fail fast.

### Vector Store

The `chroma` service acts as the vector store side-car.  Data persists under
`./data/chromadb` on the host so the index survives container restarts.  Remove
that directory to wipe all embeddings:

The project pins **Chroma 0.5.23** on both the Python client and the Docker
image with telemetry disabled so no outbound calls are made.

```bash
make clean-vector-store
```

## UI Theming

Jules ships with the **Catppuccin Macchiato** palette but does **not** enable it
by default to avoid inflating the CSS bundle for users that prefer the
standard Material-UI look.  Opt-in is controlled via an environment variable
at build (or run) time:

```bash
# Next.js build using the Catppuccin theme
NEXT_PUBLIC_THEME=catppuccin npm --prefix frontend run build
```

When `NEXT_PUBLIC_THEME` equals `catppuccin`, the global `_app.tsx` file
imports `src/styles/catppuccin-macchiato.css` automatically.  If the variable
is unset or set to a different value, no extra stylesheet is loaded and the
baseline bundle size remains unchanged.

### Accessibility

The colour combinations used in the Macchiato palette have been verified with
the WebAIM Contrast Checker and meet **WCAG 2.1 AA** (contrast ≥ 4.5:1) for
normal text.  When introducing new UI elements make sure that any additional
foreground/background pairs continue to pass or document known exceptions.

Run `npm --prefix frontend run build:tokens` to re-generate CSS variables and
TypeScript exports after editing `src/design-tokens/*.json`.

### Tailwind usage

```tsx
// tailwind.config.js
const palette = require('@catppuccin/palette');
module.exports = {
  darkMode: 'class',
  theme: { colors: { ctp: palette.flavors.macchiato.colors } }
};
```

### Chakra usage

```ts
import { ChakraProvider } from '@chakra-ui/react';
import catppuccinTheme from '../src/theme/catppuccin';

<ChakraProvider theme={catppuccinTheme}>...</ChakraProvider>
```

## License

MIT
