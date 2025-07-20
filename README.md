# Jules 🗣️🤖

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

1. Copy `.env.example` → `.env` and add your `OPENAI_API_KEY`.
2. Install Python deps & run backend:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Conversation memory is stored in-process using a singleton MemorySaver, so it
persists only while the server is running.

Then open http://localhost:8000 to chat with **Jules**.
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

## License

MIT
