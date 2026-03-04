# Conversational AI System

FastAPI + LangGraph + OpenAI-compatible chat endpoints + tool calling, with **PAHF as the only memory system**.

## Architecture

`Frontend (React + Vite) -> FastAPI -> LangGraph -> UniversalChat + PAHFMemoryService -> PAHF MemoryBank (SQLite/FAISS)`

LangGraph execution:

`memory_retrieval_node -> assistant_generation_node -> memory_extraction_node -> memory_update_node`

## Project Structure

```text
backend/                 FastAPI app + LangGraph + PAHF integration
PAHF/                    Official PAHF implementation (local clone)
frontend/                React + Vite + TypeScript frontend
tests/                   Pytest tests
docs/PAHF_MEMORY.md      PAHF memory architecture and API
docs/FRONTEND.md         Frontend architecture notes
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Conda env `servicebot` activated
- Local PAHF repo present at `./PAHF`
- PAHF requirements already installed in `servicebot`

## Setup

1. Backend deps:

```bash
pip install -r requirements.txt
```

2. PAHF deps (from local `PAHF/`):

```bash
pip install -r PAHF/requirements.txt
```

3. Frontend deps:

```bash
cd frontend
npm install
cd ..
```

4. Configure env:

```bash
copy .env.example .env
```

Required model settings:

- `MODEL_NAME`
- `BASE_URL`
- `API_KEY`

PAHF memory settings:

- `PAHF_BACKEND=sqlite|faiss` (default `sqlite`)
- `PAHF_SQLITE_DB_PATH`
- `PAHF_FAISS_PATH`
- `PAHF_TOP_K`
- `PAHF_SIMILARITY_THRESHOLD`
- `PAHF_QUERY_ENCODER`
- `PAHF_CONTEXT_ENCODER`
- `PAHF_ENABLE_PRE_CLARIFICATION`
- `PAHF_ENABLE_POST_CORRECTION`

## Run

One-command dev:

```bash
python run_all.py
```

Backend only:

```bash
python run_backend.py
```

## API Endpoints Used By Frontend

- `GET /health`
- `GET /api/v1/models` (fallback `GET /v1/models`)
- `POST /api/v1/chat/completions`
- `GET /api/v1/prompt-scenes`
- PAHF memory endpoints:
  - `POST /api/v1/memory`
  - `GET /api/v1/memory?user_id=...`
  - `GET /api/v1/memory/{memory_id}?user_id=...`
  - `PUT /api/v1/memory/{memory_id}`
  - `POST /api/v1/memory/search`
  - `POST /api/v1/memory/find-similar`

## Memory (PAHF)

PAHF memory behavior in this repo:

- Person-isolated memory (`user_id -> person_id`)
- DRAGON+ embedding retrieval via PAHF memory banks
- Pre-action clarification gate
- Post-action correction/update loop
- Similarity-based update/overwrite (not append-only)
- SQLite and FAISS backend options (SQLite default)

Detailed design: [docs/PAHF_MEMORY.md](/F:/OneDrive/desktop/项目/智能客服/docs/PAHF_MEMORY.md)

## Verification Checklist

1. Health check returns `ok`.
2. New user has empty PAHF memory list.
3. Chat creates memory automatically from durable preference/fact.
4. Later chat retrieves memory (visible in trace `retrieved_memories`).
5. Correction message updates existing memory via similarity-based merge/update.

## Demo Conversation (Memory Evolution)

1. User: `My name is Xiaohao and my shoe size is 30.`
2. User: `What is my shoe size?`
   - System should retrieve prior memory.
3. User: `Actually my shoe size is 31.`
   - System should update existing similar memory.
4. User: `What is my shoe size now?`
   - System should use updated memory.

## Tests

Run all backend tests:

```bash
pytest
```

Focused PAHF suites:

```bash
pytest tests/test_graph.py tests/test_pahf_memory_api.py -q
```

## Troubleshooting

- If `PAHF_BACKEND=faiss` fails with missing module, switch to `sqlite` or install a compatible FAISS build.
- First PAHF memory request can be slow due to DRAGON+ encoder loading.
- See [docs/PAHF_MEMORY.md](/F:/OneDrive/desktop/项目/智能客服/docs/PAHF_MEMORY.md) for detailed troubleshooting.
