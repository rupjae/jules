# ---------- Stage 1: base image with Python + Node ---------- #
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps and Node 20
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg build-essential \
    # NodeSource setup
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ---------- Stage 2: build frontend ---------- #
FROM base AS frontend-builder

WORKDIR /workspace

# Copy frontend package manifests first for cache
# Copy the entire frontend source and install deps
COPY frontend ./frontend

RUN cd frontend \
    && npm install \
    && npm run build


# ---------- Stage 3: production image ---------- #
FROM base AS production

WORKDIR /app

# Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry poetry-plugin-export \
    && poetry export --without-hashes -f requirements.txt -o requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Copy backend code
COPY backend/app ./app
# Support local helper packages
COPY db ./db
COPY jules ./jules

# Copy exported static site from the previous stage
COPY --from=frontend-builder /workspace/frontend/out ./static

# FastAPI will serve static files under /static mounted at root

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
