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

The `./data` directory holds `checkpoints.sqlite` and `jules_memory.sqlite`.

Then open http://localhost:8000 to chat with **Jules**.
Clients must pass an `X-Thread-ID` header to resume a conversation.

## Docker (single image)

```bash
# Build (first time) and start all services: backend + Next.js dev server
docker compose up --build
```

Open your browser:

• http://localhost:3000  → Next.js dev UI with hot-reload
• http://localhost:8000  → FastAPI API (and the static build when dev server is stopped)

## License

MIT
