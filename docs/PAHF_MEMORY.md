# PAHF Memory System

## 1. What This Is

This project now uses **PAHF (arXiv:2602.16173)** as the **only** memory system.

The previous custom memory stack (short-term manager, custom vector index, memory CRUD repository, rule extractor) has been removed.

PAHF components used from local `./PAHF`:

- `PAHF.memory.banks.MemoryBank`
- `PAHF.memory.banks.SQLiteMemoryBank`
- `PAHF.memory.banks.FAISSMemoryBank`
- `PAHF.memory.banks.DragonPlusEmbedding` (DRAGON+ retriever embeddings)

## 2. Architecture In This Repo

Text diagram:

`FastAPI (/api/v1/chat, /v1/chat/completions)`  
`-> LangGraph: memory_retrieval_node -> assistant_generation_node -> memory_extraction_node -> memory_update_node`  
`-> PAHFMemoryService (backend/pahf_memory/service.py)`  
`-> PAHF MemoryBank backend (SQLite or FAISS)`

## 3. user_id -> person_id Mapping

- External API and frontend still send `user_id`.
- Internally this is mapped 1:1 to PAHF `person_id`.
- All PAHF retrieval/search/update/add calls are executed within that `person_id` namespace.

## 4. Retrieval + Injection Flow

Before model generation:

1. `memory_retrieval_node` retrieves PAHF memories for `person_id=user_id`.
2. Retrieved results are formatted as PAHF memory context text.
3. Prompt builder injects:
   - `### PAHF Memory Context`
   - `### Retrieved PAHF Memories`
4. Pre-action clarification check runs (PAHF-style Ask-human gate):
   - If clarification is needed, assistant replies with a clarification question.
   - Otherwise normal generation continues.

## 5. Memory Update Flow (Clarification/Correction)

After generation:

1. `memory_extraction_node` runs PAHF-style post-action extraction:
   - Detects durable personal info / preferences / constraints / corrections.
   - Produces a concise memory summary candidate.
2. `memory_update_node` applies PAHF-style similarity update:
   - `find_similar_memory(...)`
   - If no similar memory: `add(...)`
   - If similar memory exists:
     - same-topic check (LLM)
     - merge summary (`integration_prompt`) + `update_memory(...)`
     - else add as new memory

This preserves PAHF’s overwrite/update semantics rather than append-only behavior.

## 6. Storage Backends & Persistence

Supported backends:

- `sqlite` (default): uses `SQLiteMemoryBank`
- `faiss`: uses `FAISSMemoryBank`

Persistence paths:

- SQLite file: `PAHF_SQLITE_DB_PATH`
- FAISS files: `PAHF_FAISS_PATH` with `.index` and `.docs`

Fail-fast behavior:

- If `PAHF_BACKEND=faiss` and `faiss` is unavailable, startup fails with explicit error.
- No silent switch to SQLite.

## 7. Configuration

Environment variables:

- `PAHF_BACKEND=sqlite|faiss`
- `PAHF_SQLITE_DB_PATH=./data/pahf/pahf_memory.db`
- `PAHF_FAISS_PATH=./data/pahf/pahf_memory`
- `PAHF_TOP_K=5`
- `PAHF_SIMILARITY_THRESHOLD=0.45`
- `PAHF_QUERY_ENCODER=facebook/dragon-plus-query-encoder`
- `PAHF_CONTEXT_ENCODER=facebook/dragon-plus-context-encoder`
- `PAHF_EMBED_DEVICE=` (optional, e.g. `cuda` or `cpu`)
- `PAHF_ENABLE_PRE_CLARIFICATION=true|false`
- `PAHF_ENABLE_POST_CORRECTION=true|false`
- `PAHF_LLM_MODEL=` (optional; defaults to `MODEL_NAME`)

## 8. API Reference

Base: `/api/v1/memory`

Note: PAHF `MemoryBank` does not provide native delete semantics in this integration, so no delete endpoint is exposed.

### Add

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"demo_user\",\"text\":\"My shoe size is 30.\"}"
```

### List

```bash
curl "http://localhost:8000/api/v1/memory?user_id=demo_user"
```

### Get By ID

```bash
curl "http://localhost:8000/api/v1/memory/1?user_id=demo_user"
```

### Update/Overwrite

```bash
curl -X PUT http://localhost:8000/api/v1/memory/1 \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"demo_user\",\"text\":\"My shoe size is 31.\"}"
```

### Search

```bash
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"demo_user\",\"query\":\"shoe size\",\"top_k\":5}"
```

### Find Similar

```bash
curl -X POST http://localhost:8000/api/v1/memory/find-similar \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"demo_user\",\"text\":\"Actually my shoe size is 31.\",\"threshold\":0.45}"
```

## 9. Verification Walkthrough

1. Start with empty memory list for a user.
2. Send chat: `My name is Xiaohao and my shoe size is 30.`
3. List memory; one PAHF memory should exist.
4. Ask chat: `What is my shoe size?`
5. Check chat trace `retrieved_memories`; the stored memory should be retrieved.
6. Send correction: `Actually my shoe size is 31.`
7. List memory again; similar-memory update should overwrite/merge instead of creating duplicates.

## 10. Troubleshooting

- **`ImportError: FAISS backend requested but 'faiss' module is not installed.`**
  - Use `PAHF_BACKEND=sqlite`, or install a FAISS build compatible with your Python/OS.
- **DRAGON+ model download errors**
  - Verify network access to HuggingFace model endpoints or provide local model cache.
- **Slow first request**
  - First embedding call loads DRAGON+ encoders; warm up with one request during startup.
- **No memory update in trace**
  - Check `PAHF_ENABLE_POST_CORRECTION=true`.
  - Inspect trace fields: `memory_candidate`, `memory_update`.
