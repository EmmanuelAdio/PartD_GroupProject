# PartD Group Project

This repository currently includes working ingestion and retrieval services for RAG:
1. Read JSON source files from `data/`
2. Normalize + chunk into retrieval-friendly text blocks
3. Add domain/entity tags (heuristic or LLM-assisted)
4. Generate embeddings (deterministic local or OpenAI)
5. Upsert chunk records into MongoDB (`open_day_knowledge.kb_chuncks`)
6. Plan retrieval queries with an LLM-first Processor Agent
7. Run hybrid retrieval (vector + lexical + metadata filters)
8. Merge/rerank evidence and return answerer-ready items

## Current status

Implemented:
- `app/main.py`
- `app/orchestrator.py`
- `services/ingestion_service.py`
- `services/embedding_service.py`
- `services/llm_services.py`
- `services/mongo_repo.py`
- `services/index_manager.py`
- `services/retriever_service.py`
- `agents/processor_agent.py`
- `schemas/models.py`
- `test_ingestion_service.py`
- `test_retrieval_service.py`
- `test_index_manager.py`
- `test_processor_agent.py`

## Setup

Run from repo root:
`c:\Users\Emman\OneDrive\Documents\GitHub\PartD_GroupProject`

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env`:
```env
MONGODB_URI="your_mongodb_connection_string"
OPENAI_API_KEY="your_openai_api_key"
```

Notes:
- `MONGODB_URI` is required for Mongo upload tests.
- `OPENAI_API_KEY` is required for `--embedder openai` and `--tagger llm`.
- `OPEN_API_KEY` is also accepted as an alias for OpenAI key lookup.
- `app/main.py` now loads `.env` from project root explicitly, with a built-in fallback parser when `python-dotenv` is not installed.

## Processor agent overview

`ProcessorAgent` (`agents/processor_agent.py`) is implemented as an LLM-first query planner.

It:
- normalizes the raw user query
- builds a schema-aware prompt for retrieval planning
- calls `LLMService` for JSON output
- parses output into `RetrievalQuery`
- applies small post-processing for retrieval safety/consistency

It does not:
- query MongoDB
- retrieve documents
- generate final answers
- run embeddings
- ingest documents

## Retrieval service overview

`RetrieverService` supports hybrid retrieval over MongoDB chunk records:
- Vector retrieval using Atlas Vector Search (`$vectorSearch`) when index is ready.
- Lexical retrieval using Atlas Search (`$search`) when text index is ready.
- Hard metadata filters on `domain`, optional `source_id`, `version`.
- Soft rerank boosts on `section` and `entity_tags` (instead of hard pre-filtering).
- Merge + rerank with overlap boost for chunks returned by both channels.
- Deduplication by `chunk_id`.

Fallback behavior (for resilience):
- If vector index is not ready/unavailable: falls back to Python cosine scan over filtered docs.
- If Atlas Search index is not ready/unavailable: falls back to Mongo `$text`.
- If `$text` is unavailable: falls back to regex/token scan.

Index management:
- `AtlasIndexManager` checks, creates, and reconciles Atlas vector/text search indexes.
- `MongoRepo.ensure_indexes()` also creates a native Mongo text index:
  `chunk_text_search_idx` on `text`, `title`, `entity_tags` for lexical fallback.

## FastAPI integration (app.main)

`app.main` supports both:
- CLI-based ingestion runs
- FastAPI endpoints for ingestion and querying when launched by uvicorn

Runtime behavior in FastAPI mode:
- Startup creates both `IngestionOrchestrator` and `QueryOrchestrator`.
- Embedder backend is selected automatically:
  - `openai` when `OPENAI_API_KEY` or `OPEN_API_KEY` exists
  - `fake` otherwise
- This keeps startup resilient if OpenAI credentials are missing.

### 1) Initial ingestion from CLI (all data files)

```bash
python -m app.main --embedder fake --tagger heuristic
```

Expected result:
- First run ingests all JSON sources and returns non-zero `total_new_records`.
- Example observed: `total_new_records: 570`.

### 2) Incremental rerun (skip unchanged sources)

Run the same command again:
```bash
python -m app.main --embedder fake --tagger heuristic
```

Expected result:
- `ingested_sources` should be empty.
- Sources appear in `skipped_sources` with reason `unchanged_source_and_pipeline`.
- `total_new_records: 0`.

How skip works:
- Source-level manifest check:
  source file hash + pipeline hash + existing docs.
- Chunk-level check:
  existing `chunk_id`s are filtered before tagging/embedding to avoid cost.

### 3) Force full reingest

```bash
python -m app.main --embedder fake --tagger heuristic --full-reingest
```

### 4) Ingest one file only

```bash
python -m app.main --file accommodation_halls.json --embedder fake --tagger heuristic
```

### 5) Use OpenAI embeddings and/or LLM tagging

```bash
python -m app.main --embedder openai --tagger llm --llm-model gpt-4o-mini
```

### 6) Run FastAPI server with uvicorn

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

If port `8000` is in use, use another port:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Available endpoints:
- `GET /health`
- `GET /status`
- `POST /ingest/all`
- `POST /ingest/file`
- `POST /ingest/payload`
- `POST /query`

## FastAPI endpoint usage

Base URL:
`http://127.0.0.1:8000`

### Health + status endpoints

1. Health check:
```bash
curl http://127.0.0.1:8000/health
```

2. Retrieval/index status:
```bash
curl http://127.0.0.1:8000/status
```

`/status` includes:
- Mongo collection doc count
- Index health (`vector_index_found`, `text_index_found`, statuses)
- Any health errors (including dimension mismatch warnings)

### Data endpoints (ingestion)

1. Ingest all files from `data/`:
```bash
curl -X POST "http://127.0.0.1:8000/ingest/all?incremental=true"
```

2. Ingest a single file:
```bash
curl -X POST "http://127.0.0.1:8000/ingest/file" \
  -H "Content-Type: application/json" \
  -d "{\"file_path\":\"accommodation_halls.json\",\"incremental\":true}"
```

3. Ingest JSON payload directly:
```bash
curl -X POST "http://127.0.0.1:8000/ingest/payload" \
  -H "Content-Type: application/json" \
  -d "{\"source_id\":\"manual_payload\",\"title\":\"Manual Payload\",\"incremental\":false,\"data\":{\"items\":[{\"text\":\"Example\"}]}}"
```

### Query endpoint

1. Query with default planner `top_k`:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"How much is Butler Court accommodation?\"}"
```

2. Query with explicit `top_k` override:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"UCAS requirements for Computer Science\",\"top_k\":10}"
```

Response includes:
- `processor_plan`
- `retrieval_run.attempts_log`
- `retrieval_run.evidence`
- `retrieval_run.diagnostics`

## Ingestion test commands

### 1) Local test for all JSON files

Command:
```bash
python test_ingestion_service.py --embedder fake --tagger heuristic
```

What it does:
- Ingests all `data/*.json` except preview files.
- Uses deterministic local embeddings.
- Uses heuristic domain/entity tagging.

Expected output example:
```text
[accommodation_halls.json]
  records: 44
  first chunk domain: accommodation

[all_ug_courses.json]
  records: 521
  first chunk domain: courses

[contextual_offer_faqs.json]
  records: 2
  first chunk domain: admissions

[lboro_sport_faqs.json]
  records: 3
  first chunk domain: sports

[LOCAL SUMMARY] total records built: 570
```

### 2) Local test for one file

Command:
```bash
python test_ingestion_service.py --file accommodation_halls.json --embedder fake --tagger heuristic
```

Expected output example:
```text
[accommodation_halls.json]
  records: 44
  first chunk id: ...
  first chunk domain: accommodation
```

### 3) Local test + write preview records

Command:
```bash
python test_ingestion_service.py --embedder fake --tagger heuristic --write-previews
```

What it writes:
- `data/chunk_previews/<source>_chunk_records_preview.json`

Expected output includes:
```text
wrote preview: data\chunk_previews\accommodation_halls_chunk_records_preview.json
```

## Mongo integration commands

### 4) Upload all JSON files to MongoDB (recommended baseline)

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder fake --tagger heuristic --clear-source-before-upload
```

What it does:
- Ingests all JSON files.
- Clears existing docs per `source_id` before upload.
- Upserts into `open_day_knowledge.kb_chuncks`.
- Prints local vs Mongo counts by source.

Expected output example:
```text
[MONGO UPLOAD TEST]
  database: open_day_knowledge
  collection: kb_chuncks
  embedder: fake
  tagger: heuristic
  files to ingest: 4

[accommodation_halls.json]
  local records built: 44
  records in Mongo by source_id=accommodation_halls: 44

[all_ug_courses.json]
  local records built: 521
  records in Mongo by source_id=all_ug_courses: 521

[contextual_offer_faqs.json]
  local records built: 2
  records in Mongo by source_id=contextual_offer_faqs: 2

[lboro_sport_faqs.json]
  local records built: 3
  records in Mongo by source_id=lboro_sport_faqs: 3

[MONGO SUMMARY]
  total local records built: 570
  total Mongo records across ingested source_ids: 570
```

### 5) Upload one file to MongoDB

Command:
```bash
python test_ingestion_service.py --file accommodation_halls.json --test-mongo-upload --embedder fake --tagger heuristic --clear-source-before-upload
```

### 6) Upload with LLM tagging enabled

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder fake --tagger llm --llm-model gpt-4o-mini --clear-source-before-upload
```

Notes:
- Uses `LLMService` for chunk tagging.
- Falls back to heuristics if LLM tagging fails on a chunk.

### 7) Use OpenAI embeddings

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder openai --tagger heuristic --clear-source-before-upload
```

### 8) Override Mongo target

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --mongo-db open_day_knowledge --mongo-collection kb_chuncks
```

## Retrieval and index test commands

### 9) Check Atlas index health only

Command:
```bash
python test_index_manager.py --embedder fake --mongo-db open_day_knowledge --mongo-collection kb_chuncks
```

### 10) Create/reconcile Atlas indexes (fake embeddings)

Command:
```bash
python test_index_manager.py --embedder fake --mongo-db open_day_knowledge --mongo-collection kb_chuncks --ensure-indexes
```

### 11) Create/reconcile Atlas indexes (OpenAI embeddings)

Command:
```bash
python test_index_manager.py --embedder openai --embedding-model text-embedding-3-small --mongo-db open_day_knowledge --mongo-collection kb_chuncks --ensure-indexes
```

### 12) Run retrieval test suite (default sample queries)

Command:
```bash
python test_retrieval_service.py --embedder fake --mongo-db open_day_knowledge --mongo-collection kb_chuncks
```

Expected output includes:
- index health summary
- query-level fallback diagnostics
- ranked evidence snippets

Example diagnostic line:
```text
fallback_used: YES (vector_mode=fallback_cosine, text_mode=mongo_text)
```

### 13) Run retrieval test for a single query

Command:
```bash
python test_retrieval_service.py --embedder fake --query "Butler Court" --top-k 5
```

### 14) Run retrieval test with structured filters

Command:
```bash
python test_retrieval_service.py --embedder fake --query "UCAS entry requirements" --domain courses --domain admissions --entity-tag UCAS --top-k 6
```

### 15) Auto-create/reconcile indexes from retrieval test script

Command:
```bash
python test_retrieval_service.py --embedder fake --auto-create-indexes
```

### 16) Index check only via retrieval script

Command:
```bash
python test_retrieval_service.py --embedder fake --check-indexes-only
```

### 17) Full real-embedding path (recommended production-like flow)

1. Re-ingest all data with OpenAI embeddings:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder openai --tagger heuristic --clear-source-before-upload --mongo-db open_day_knowledge --mongo-collection kb_chuncks
```
2. Ensure Atlas indexes with OpenAI dimensions:
```bash
python test_index_manager.py --embedder openai --embedding-model text-embedding-3-small --mongo-db open_day_knowledge --mongo-collection kb_chuncks --ensure-indexes
```
3. Run retrieval tests with OpenAI embedder:
```bash
python test_retrieval_service.py --embedder openai --embedding-model text-embedding-3-small --mongo-db open_day_knowledge --mongo-collection kb_chuncks --auto-create-indexes
```

## Processor + retrieval integration test commands

### 18) Processor plans query, then retrieval runs on that plan

Command:
```bash
python test_processor_agent.py --query "How much is Butler Court accommodation?"
```

What it does:
- Generates a `RetrievalQuery` plan via `ProcessorAgent`.
- Prints the plan JSON.
- Runs `RetrieverService` with that exact plan.
- Prints diagnostics and ranked evidence.

### 19) Same flow with fake embedder (no embedding API calls)

Command:
```bash
python test_processor_agent.py --query "How much is Butler Court accommodation?" --embedder fake --mongo-db open_day_knowledge --mongo-collection kb_chuncks
```

### 20) Override planned top-k for retrieval experiments

Command:
```bash
python test_processor_agent.py --query "UCAS entry requirements for Computer Science" --top-k-override 10
```

### 21) Write processor+retrieval report JSON

Command:
```bash
python test_processor_agent.py --query "How much is Butler Court accommodation?" --write-report reports/processor_butler_test.json
```

### 22) Use OpenAI embedder at retrieval time (production-like query embedding path)

Command:
```bash
python test_processor_agent.py --query "How much is Butler Court accommodation?" --embedder openai --embedding-model text-embedding-3-small
```

### 23) Auto-create/reconcile indexes from processor test script

Command:
```bash
python test_processor_agent.py --query "How much is Butler Court accommodation?" --auto-create-indexes
```

### 24) See all processor test CLI options

Command:
```bash
python test_processor_agent.py --help
```

## Version numbers in ingestion records

Where version comes from:
- `ChunkRecord.version` is set by `IngestionService(version=...)`.
- In `test_ingestion_service.py`, the service is currently created with `version="test-v2"`.
- If not set explicitly, `IngestionService` defaults to `version="v1"`.

What this means now:
- Version values are currently manual labels, not auto-generated.
- They are useful to track which ingestion logic produced a given record set.

Recommended versioning rule (next step):
1. Use semantic labels: `ingest-v1`, `ingest-v2`, etc.
2. Bump when chunking/tagging/embedding logic changes.
3. Keep old data queryable by filtering on `version`.

## What is still missing in this ingestion pipeline

1. Strong idempotency key strategy across model changes
Currently `chunk_id` is content-derived; changing chunk format can create new ids unexpectedly. A stable source+path strategy would help.

2. Atlas index operations are environment-dependent
Atlas Search/Vector index APIs vary by cluster tier/permissions/driver support.
`AtlasIndexManager` now handles helper + command fallbacks, but Atlas permissions are still required.

3. Retry/backoff around external API calls
OpenAI embedding/tagging calls should have retry policy and better error telemetry.

4. Cost and throughput controls
No rate limiting, token budgeting, or batching policy tuning for LLM tagging at scale.

5. Quality evaluation for tags
No automated evaluation set yet to measure domain/entity tagging precision/recall.

6. Ingestion observability
No run report persisted yet (start/end time, records processed, failures by source).

7. Lifecycle tooling
No explicit rollback/rebuild command per source and version beyond manual deletes.

## Troubleshooting

### `MONGODB_URI is not set`
- Add it to `.env` or shell environment.

### `OPENAI_API_KEY is not set` / `OPEN_API_KEY is not set`
- Required for OpenAI embeddings and LLM calls.
- FastAPI startup will now fall back to `fake` embedder and non-LLM planning when key is missing.
- If you expect OpenAI mode, verify with:
  - `GET /status`
  - or inspect runtime env before server start.

### `python-dotenv` not installed
- `.env` still loads in FastAPI mode because `app/main.py` includes a fallback parser.
- For CLI tools or notebooks, install for consistency:
  `pip install python-dotenv`

### Atlas dimension mismatch
Error example:
`Embedder produces 64-dim vectors but Atlas index expects 1536-dim`

Cause:
- Query-time embedder and stored/indexed embeddings were built with different backends.

Fix:
1. Ensure OpenAI key is loaded (`OPENAI_API_KEY`/`OPEN_API_KEY`).
2. Re-ingest with OpenAI embeddings:
   `python test_ingestion_service.py --test-mongo-upload --embedder openai --tagger heuristic --clear-source-before-upload`
3. Recreate/reconcile indexes with OpenAI dims:
   `python test_index_manager.py --embedder openai --embedding-model text-embedding-3-small --ensure-indexes`
4. Restart FastAPI and re-check `/status`.

### Mongo DNS/timeout errors
- Check internet access.
- Check Atlas network allowlist/firewall.
- Check URI correctness.

### Count mismatch (local vs Mongo)
- Re-run with `--clear-source-before-upload`.
- Ensure `--mongo-collection kb_chuncks`.
- Check if previous versions/data exist in same source_id.

### Atlas index errors / not READY
- Run:
  `python test_index_manager.py --embedder fake --ensure-indexes`
- If using OpenAI embeddings, ensure index dims match:
  `python test_index_manager.py --embedder openai --embedding-model text-embedding-3-small --ensure-indexes`
- If status is `PENDING`, wait a few minutes and check again.

### Retrieval always using fallbacks
- Check diagnostics in retrieval output (`vector_mode`, `text_mode`, `fallback_used`).
- Run index health check:
  `python test_retrieval_service.py --check-indexes-only`
- Confirm embedding dimensions match stored data and query embedder.

### FastAPI `500` errors from endpoints
Use this quick triage:
1. `GET /health` to confirm server is running.
2. `GET /status` to inspect Mongo/index readiness and dimension errors.
3. Check server logs for stack trace.
4. Verify `.env` keys and restart uvicorn after any env change.

### Credential leakage handling
- If an OpenAI key was ever committed, logged, or pasted in traces, rotate/revoke it immediately.
- Generate a new key and update `.env`.
