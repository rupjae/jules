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
│   ├── requirements.txt
│   └── Dockerfile
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
2. Install Python deps & run backend:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Conversation history is stored in a SQLite database under `data/` so it
persists across restarts.

Then open http://localhost:8000 to chat with **Jules**.

### API Endpoints
- **POST /api/chat?thread_id=<id>&message=<text>**
  Streams chat completions via Server-Sent Events.  On the first request for a new thread,
  the `X-Thread-ID` response header provides the generated session ID.  Include the same
  `thread_id` (query or `X-Thread-ID` header) on subsequent calls to continue the conversation.
- **GET /api/chat/history?thread_id=<id>**
  Returns the full conversation history as JSON:
  ```
  > NOTE: Both `/api/chat/history` and `/api/chat/search` responses now include a `"timestamp"` field (ISO-8601 UTC) for each message.
  [
    {"sender": "user",      "content": "Hello!"},
    {"sender": "assistant", "content": "Hi there!"},
    ...
  ]
  ```
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

### Vector Store

The `chroma` service acts as the vector store side-car.  Data persists under
`./.chroma-data` on the host so the index survives container restarts.  Remove
that directory to wipe all embeddings:

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
