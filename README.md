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
9. Generate grounded final answers from retrieved evidence with an Answerer Agent
10. Evaluate answer grounding/relevance/safety with a hybrid Evaluator Agent
11. Apply orchestrator decisions (`pass`, `revise`, `ask_clarification`, `fallback`) before final API output

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
- `agents/answerer_agent.py`
- `agents/evaluator_agent.py`
- `schemas/models.py`
- `test_ingestion_service.py`
- `test_retrieval_service.py`
- `test_index_manager.py`
- `test_processor_agent.py`
- `test_answerer_agent.py`
- `test_evaluator_agent.py`

## Setup

Run from repo root:
`c:\Users\Emman\OneDrive\Documents\GitHub\PartD_GroupProject`

Install dependencies:
```bash
pip install -r requirements.txt
```

Recommended interpreter for this repo:
```bash
.venv\Scripts\python.exe
```

Create `.env`:
```env
MONGODB_URI="your_mongodb_connection_string"
OPENAI_API_KEY="your_openai_api_key"
FRONTEND_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

Notes:
- `MONGODB_URI` is required for Mongo upload tests.
- `OPENAI_API_KEY` is required for `--embedder openai` and `--tagger llm`.
- `OPEN_API_KEY` is also accepted as an alias for OpenAI key lookup.
- `pytest` is required for the unit test files (install with `pip install pytest`).
- `FRONTEND_ORIGINS` controls FastAPI CORS origins for browser calls (comma-separated).
- `app/main.py` now loads `.env` from project root explicitly, with a built-in fallback parser when `python-dotenv` is not installed.
- Keep provider keys server-side only. Never put `OPENAI_API_KEY` into frontend env files.

## Run the project (backend + frontend)

This is the fastest way to run the full chat app locally.

### Prerequisites

- Python 3.10+ (recommended)
- Node.js 20.19+ or 22.12+ (Vite 7 requirement)
- npm

### 1) Start the backend (FastAPI)

From the repo root:
```bash
cd c:\Users\Emman\OneDrive\Documents\GitHub\PartD_GroupProject
pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend will be available at:
- `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/health`

### 2) Start the frontend (Vite avatar UI)

Open a second terminal:
```bash
cd c:\Users\Emman\OneDrive\Documents\GitHub\PartD_GroupProject\cs-avatar
npm install
```

Create `cs-avatar/.env` if it does not exist:
```env
VITE_API_BASE_URL="http://127.0.0.1:8000"
```

Then run:
```bash
npm run dev
```

Frontend will usually start on:
- `http://localhost:5173`

### 3) Verify frontend-backend connection

1. Ensure backend terminal shows uvicorn running on port `8000`.
2. Open the Vite URL (`http://localhost:5173`).
3. Send a chat question in the avatar UI.
4. You should receive a response from `POST /query`.

If browser calls fail due to CORS, verify project root `.env` includes:
```env
FRONTEND_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

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

## Answerer agent overview

`AnswererAgent` (`agents/answerer_agent.py`) is the grounded generation step that produces a draft answer before evaluator verification.

It:
- receives the raw user query plus retrieved `EvidenceItem`s
- builds a RAG-style prompt over the retrieved evidence
- asks `LLMService` for structured answer JSON when OpenAI is available
- returns an `AnswerResult` with answer text, confidence, and citations
- falls back to deterministic evidence summarization when no LLM key is configured

It does not:
- retrieve documents
- query MongoDB directly
- plan retrieval filters
- ingest or embed documents

## Evaluator agent overview

`EvaluatorAgent` (`agents/evaluator_agent.py`) verifies the draft answer before the orchestrator returns the final response.

Design:
- Layer 1: deterministic rule checks (always on, fast).
- Layer 2: optional LLM judge (only for ambiguous/borderline cases when an LLM is available).

Inputs:
- `user_query`
- `RetrievalQuery` plan (if available)
- retrieved `EvidenceItem[]`
- answer draft (`AnswerResult`)

Rule checks currently cover:
- evidence presence and evidence strength thresholds
- grounding heuristics (citation validity, numeric support, lexical overlap)
- relevance to query focus (including price/location/requirements)
- clarity/usefulness for open-day visitors
- safety/uncertainty handling under weak evidence

Output:
- `EvaluationResult` with:
  - `verdict`: `pass | revise | ask_clarification | fallback`
  - booleans: `grounded`, `relevant`, `clear`, `safe`
  - `issues`, `suggested_action`, `suggested_filters`, `clarification_question`, `notes`

Orchestrator policy:
- `pass`: keep answer
- `revise`: one retrieval/answer retry with suggested filter adjustments
- `ask_clarification`: return clarification prompt
- `fallback`: return safe fallback with official link

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

### 7) Run avatar frontend (Vite)

From `cs-avatar/`, create `.env`:
```env
VITE_API_BASE_URL="http://127.0.0.1:8000"
```

Start Vite:
```bash
cd cs-avatar
npm install
npm run dev
```

Local chat startup flow:
1. Start FastAPI on `127.0.0.1:8000`.
2. Start Vite on `localhost:5173`.
3. Open the Vite URL and send a chat message from the avatar UI.

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

3. Browser/frontend shape (`debug=false`, default):
```json
{
  "query": "...",
  "answer": "...",
  "grounded": true,
  "confidence": 0.98,
  "citations": []
}
```

4. Debug/full shape (`debug=true`) includes multi-agent internals:
- `answerer_run`
- `evaluator_run`
- `orchestration_decision`
- `processor_plan`
- `retrieval_run.attempts_log`
- `retrieval_run.evidence`
- `retrieval_run.diagnostics`

## Evaluation and load testing

This repository now includes three separate testing/evaluation paths for the chatbot:
- `scripts/eval_accuracy.py`: offline quality evaluation with RAGAS.
- `scripts/eval_evaluator.py`: offline evaluator/orchestration behavior logging.
- `locustfile.py`: API load test against `POST /query`.

These are intentionally different:
- `eval_accuracy.py` does not call the HTTP API. It imports and runs `QueryOrchestrator.run()` directly inside Python, then scores the outputs with RAGAS.
- `eval_evaluator.py` also does not call the HTTP API. It imports and runs `QueryOrchestrator.run()` directly, then logs evaluator decisions, retries, fallback usage, and timing.
- `locustfile.py` does call the live API. It sends real HTTP `POST /query` requests to FastAPI and measures latency, throughput, and failures under concurrent load.

### Benchmark dataset

Both evaluation scripts use the built-in benchmark set in:
- `scripts/benchmark_questions.py`

Each benchmark item contains:
- `id`
- `category`
- `question`
- `ground_truth_answer`

The benchmark currently covers:
- accommodation
- undergraduate courses
- contextual offers / admissions / policy

### Timing instrumentation

`QueryOrchestrator.run()` now returns additive timing metadata in:

```json
"timing_ms": {
  "processor": 0.0,
  "retriever": 0.0,
  "answerer": 0.0,
  "evaluator": 0.0,
  "total": 0.0
}
```

The evaluation scripts flatten this into row-level columns such as:
- `processor_time_ms`
- `retriever_time_ms`
- `answerer_time_ms`
- `evaluator_time_ms`
- `total_time_ms`

### Accuracy evaluation with RAGAS

Purpose:
- run the benchmark questions through the full orchestrator
- capture generated answers and retrieved contexts
- compute report-ready RAGAS metrics

Main script:
- `scripts/eval_accuracy.py`

Output files:
- `results/accuracy_eval_<timestamp>.csv`
- `results/accuracy_eval_<timestamp>.json`

The CSV contains one row per benchmark item. The JSON contains:
- run metadata
- summary metrics
- row-level results
- raw orchestration outputs

Typical columns include:
- benchmark metadata: `id`, `category`, `question`, `ground_truth_answer`
- answer data: `generated_answer`, `retrieved_context_count`
- RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- aggregate metric: `overall_metric_score`
- timings: `total_time_ms`, `processor_time_ms`, `retriever_time_ms`, `answerer_time_ms`, `evaluator_time_ms`
- evaluator/orchestration: `evaluator_decision`, `retry_count`
- error fields: `success`, `error_message`, `metric_error_message`

Run the full benchmark:

```bash
.venv\Scripts\python.exe scripts\eval_accuracy.py
```

Run only one category:

```bash
.venv\Scripts\python.exe scripts\eval_accuracy.py --category accommodation
```

Run a subset of benchmark IDs:

```bash
.venv\Scripts\python.exe scripts\eval_accuracy.py --ids accommodation_01,accommodation_02
```

Run a shorter smoke test:

```bash
.venv\Scripts\python.exe scripts\eval_accuracy.py --limit 3
```

Useful options:
- `--category <category>`
- `--ids <comma,separated,ids>`
- `--limit <n>`
- `--top-k <n>`
- `--output-dir <dir>`
- `--save-mongo`
- `--mongo-db <name>`
- `--mongo-collection <name>`

OpenAI requirements:
- `OPENAI_API_KEY` or `OPEN_API_KEY` must be set to run RAGAS scoring.

Runtime notes:
- This script can take several minutes because it runs the orchestrator and then the RAGAS scoring layer.
- The script prints the active RAGAS config at startup:
  - LLM model
  - embedding model
  - question count

Summary output includes:
- tests run
- batch total runtime
- average faithfulness
- average answer relevancy
- average context precision
- average context recall
- average overall metric score
- scored-row counts for each metric

Interpreting `None` metrics:
- If the orchestrator ran but all averages are `None`, check:
  - `RAGAS warning:` in terminal output
  - `ragas_error` in the JSON
  - `metric_error_message` in row data
  - scored-row counts such as `Scored rows for faithfulness: 0/30`
- If scored-row counts are zero, the orchestrator likely succeeded but the RAGAS layer did not produce valid numeric outputs.

Environment note about RAGAS:
- The locally installed `ragas` version may expose both legacy and collections-based APIs.
- In this repository, `scripts/eval_accuracy.py` is written to work with the installed environment and may use legacy-compatible metric objects and embedding wrappers internally.
- This is why the script should be treated as the source of truth for the current project environment rather than generic RAGAS examples from external docs.

### Evaluator effectiveness evaluation

Purpose:
- measure how the Evaluator and orchestration policy behave on the benchmark set
- count `pass`, `revise`, `ask_clarification`, and `fallback` style outcomes
- log retries, fallback usage, and timing

Main script:
- `scripts/eval_evaluator.py`

Output files:
- `results/evaluator_eval_<timestamp>.csv`
- `results/evaluator_eval_<timestamp>.json`

The CSV and JSON log one row per benchmark item and include fields such as:
- `evaluator_decision`
- `initial_verdict`
- `final_verdict`
- `effective_verdict`
- `runtime_action`
- `retry_count`
- `fallback_used`
- `clarification_requested`
- `revised`
- `retry_improved_result`
- `final_answer`
- answer/evaluator metadata
- retrieval attempt metadata
- timing breakdowns
- success/error flags

Run the full evaluator benchmark:

```bash
.venv\Scripts\python.exe scripts\eval_evaluator.py
```

Run a filtered category:

```bash
.venv\Scripts\python.exe scripts\eval_evaluator.py --category undergraduate_courses
```

Run a small subset:

```bash
.venv\Scripts\python.exe scripts\eval_evaluator.py --limit 5
```

Summary output includes:
- batch total runtime
- evaluator outcome counts
- total tests run
- successful vs errored runs
- fallback count
- clarification-requested count
- revised count
- average retry count
- average total time

This script is useful when you want to answer questions like:
- How often did the evaluator pass answers immediately?
- How often did retries happen?
- How often did the system fall back to a safe response?
- How much time does the evaluator/orchestration layer add?

### Difference between `eval_accuracy.py` and `eval_evaluator.py`

`eval_accuracy.py` is for answer quality:
- evaluates the final answer against retrieved context and ground truth
- uses RAGAS metrics
- best for report-ready quality scoring

`eval_evaluator.py` is for evaluator behavior:
- logs evaluator and orchestration decisions
- does not compute RAGAS metrics
- best for analyzing retries, fallback logic, and clarification behavior

Neither of these scripts uses `POST /query`. They both run the orchestrator directly inside Python.

### Locust load testing

Purpose:
- test the live FastAPI endpoint under concurrent load
- measure latency, throughput, and failures

Main file:
- `locustfile.py`

What it does:
- picks random benchmark questions from `scripts/benchmark_questions.py`
- sends `POST /query` requests with:

```json
{
  "query": "...",
  "debug": false
}
```

- marks failures when:
  - status code is not `200`
  - JSON is invalid
  - API returns an error payload
  - response has no `answer`

#### 1) Start the backend

Locust requires the API server to be running first.

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

#### 2) Run Locust with the web UI

```bash
.venv\Scripts\locust.exe -f locustfile.py --host http://127.0.0.1:8000
```

Then open:
- `http://127.0.0.1:8089`

Suggested smoke settings:
- users: `5`
- spawn rate: `1`
- run time: `2-5 minutes`

Suggested heavier settings:
- users: `20`
- spawn rate: `2-4`
- run time: `5-10 minutes`

#### 3) Run Locust headless and save results

Example:

```bash
.venv\Scripts\locust.exe -f locustfile.py --host http://127.0.0.1:8000 --headless --users 10 --spawn-rate 2 --run-time 5m --csv results\locust_query_test --html results\locust_query_test.html
```

This generates:
- `results\locust_query_test_stats.csv`
- `results\locust_query_test_stats_history.csv`
- `results\locust_query_test_failures.csv`
- `results\locust_query_test_exceptions.csv`
- `results\locust_query_test.html`

Recommended comparison set for reporting:

```bash
.venv\Scripts\locust.exe -f locustfile.py --host http://127.0.0.1:8000 --headless --users 5 --spawn-rate 1 --run-time 5m --csv results\locust_5_users --html results\locust_5_users.html
.venv\Scripts\locust.exe -f locustfile.py --host http://127.0.0.1:8000 --headless --users 10 --spawn-rate 2 --run-time 5m --csv results\locust_10_users --html results\locust_10_users.html
.venv\Scripts\locust.exe -f locustfile.py --host http://127.0.0.1:8000 --headless --users 20 --spawn-rate 4 --run-time 5m --csv results\locust_20_users --html results\locust_20_users.html
```

Useful Locust metrics for reports:
- requests per second
- median response time
- 95th percentile response time
- 99th percentile response time
- failure rate

### Which test should I use?

Use `eval_accuracy.py` when you need:
- answer-quality metrics
- context/faithfulness scoring
- exportable CSV/JSON results for report graphs

Use `eval_evaluator.py` when you need:
- evaluator decision counts
- retry/fallback/clarification analysis
- orchestration behavior summaries

Use `locustfile.py` when you need:
- API performance testing
- concurrent user simulation
- latency/throughput/failure measurements under load

### Typical evaluation workflow

1. Make sure MongoDB and OpenAI credentials are configured in `.env`.
2. Run a small accuracy smoke test:

```bash
.venv\Scripts\python.exe scripts\eval_accuracy.py --limit 3
```

3. Run the full accuracy benchmark:

```bash
.venv\Scripts\python.exe scripts\eval_accuracy.py
```

4. Run the evaluator behavior benchmark:

```bash
.venv\Scripts\python.exe scripts\eval_evaluator.py
```

5. Start FastAPI and run the Locust comparison set if you also need performance data.

6. Use the generated files in `results/` for graphs and write-up:
- `accuracy_eval_*.csv`
- `accuracy_eval_*.json`
- `evaluator_eval_*.csv`
- `evaluator_eval_*.json`
- `locust_*_stats.csv`
- `locust_*_stats_history.csv`
- `locust_*_failures.csv`
- `locust_*.html`

### Notebook analysis and figure generation

This repository also includes notebook-first analysis for the evaluation and load-test outputs:
- `notebooks/eval_analysis.ipynb`
- `notebooks/load_test_analysis.ipynb`

These notebooks are designed to:
- load the latest matching CSV outputs automatically by default
- keep the report tables and printed summaries in the notebook output
- save chart PNG files into `results/figures/`
- also display grouped/overview figures directly in notebook output cells

Current notebook behavior:
- `eval_analysis.ipynb` keeps the preview tables, metric tables, timing tables, and evaluator tables, then shows:
  - section overview figures
  - grouped figures for related plots
  - a final `Generated Figure Gallery` section that displays the saved evaluation figures in one place
- `load_test_analysis.ipynb` keeps the summary tables and exception tables, then shows:
  - inline overview figures for load summary, time-series behavior, and failures/exceptions
  - grouped figures for related load-test plots

Figure-saving behavior:
- Saved figures go to `results/figures/`.
- Figure filenames are deterministic so rerunning the same notebook does not create clutter from repeated copies.
- A figure-signature manifest is stored at:
  - `results/figures/_figure_manifest.json`
- If the same chart is regenerated from the same data, the existing PNG is reused instead of being saved again.
- If the data changes, the stable PNG is updated in place.

Notebook usage:

1. Start Jupyter from the repo root with the project interpreter:
```bash
.venv\Scripts\python.exe -m notebook
```

2. Open either notebook and select the `.venv` kernel.

3. Run the notebook from top to bottom.

4. Check:
- inline tables in notebook output
- inline grouped/overview figures in notebook output
- saved PNG files in `results/figures/`

If notebook figures do not appear:
- restart the kernel
- rerun the setup/import cell first
- rerun the plotting cells

The notebooks rely on the project analysis helpers in:
- `scripts/analysis_helpers.py`

Those helpers now provide:
- stable figure naming
- signature-based figure deduplication
- save-and-display behavior for notebook plotting

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

## Evaluator + orchestration test commands

### 25) Run evaluator unit tests

Command:
```bash
python -m pytest test_evaluator_agent.py
```

### 26) Run orchestration policy + API shape unit tests

Command:
```bash
python -m pytest test_query_orchestrator_unit.py test_query_api_shape.py
```

What these validate:
- Evaluator verdicts and rule-check behavior
- Optional LLM-judge gating logic
- One-retry cap for `revise`
- Debug payload shape including `evaluator_run`

## Planned evaluator improvements / TODOs

These are sensible next-step improvements for the multi-agent runtime, but they are not fully implemented yet:

1. Proper clarification memory across turns
Current clarification behavior is mostly single-turn. A future improvement is to store pending clarification state so follow-up answers such as `Butler Court` can be linked back to the earlier unresolved question instead of relying only on lightweight frontend context hints.

2. Stronger evaluator branch coverage and two-turn clarification benchmarking
Current evaluator benchmarking records verdicts, retries, clarification requests, and fallback usage, but a future improvement is to make branch coverage more explicit for `revise`, `ask_clarification`, and `fallback`, and to add a two-turn clarification benchmark that checks whether the system handles clarification follow-ups correctly.

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

### Atlas TLS handshake failures at startup
If startup fails with `ServerSelectionTimeoutError` + `SSL handshake failed`, run:
```bash
.venv\Scripts\python.exe scripts/mongo_tls_probe.py
```

Then verify:
1. You are running with repo interpreter `.venv\Scripts\python.exe`.
2. `MONGODB_URI` in root `.env` is a fresh Atlas Driver URI.
3. Atlas Network Access includes your current public IP.
4. Atlas DB user has read/write permissions on `open_day_knowledge`.
5. VPN/proxy/SSL interception is disabled or tested from another network.

Once probe passes, retry backend startup:
```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

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
