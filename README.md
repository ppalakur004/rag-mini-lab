# Mini RAG Lab — Grounded Expense-Policy Assistant

CLI Retrieval-Augmented Generation app over `policy.md`. It chunks the policy into six sections, embeds them locally, stores text + vectors + metadata in PostgreSQL/pgvector, retrieves the three nearest chunks by cosine distance, and answers with citations from those excerpts only.

## Prerequisites

- Python 3.11+
- Docker Desktop
- Ollama app running locally with `mistral:7b` pulled
  (`http://127.0.0.1:11434` must respond)

## Setup

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d
```

Wait until Postgres is healthy on `localhost:5433`.

If `docker compose up` fails to bind-mount files from a path under `~/Desktop`, add this project directory to Docker Desktop file sharing (Settings → Resources → File sharing), or start an equivalent `pgvector/pgvector:pg16` container that publishes `5433` without mounting from `~/Desktop`.

## Ingest

```bash
python -m rag_lab ingest policy.md
```

Expected: `ingested 6 chunks into postgresql://rag:rag@127.0.0.1:5433/rag_lab`

## Ask

```bash
python -m rag_lab ask "How much can I spend on food each day?"
```

Prints JSON with `answer`, `citation`, and up to three `retrieved_chunks` (numeric distances).

## Saved output for the six required questions

```bash
python -m rag_lab eval
```

Writes `tests/output/required_questions.json`.

## Schema

See `migrations/001_create_chunks.sql`. Retrieval SQL is:

```sql
ORDER BY embedding <=> :query_vector ASC
LIMIT 3;
```
