# PartD Group Project

This repository currently implements and tests a JSON-first ingestion pipeline for RAG data, with optional upload to MongoDB.

The active workflow is:
1. Load source JSON from `data/`
2. Normalize and chunk it with `IngestionService`
3. Generate embeddings (fake/local or OpenAI)
4. Build `ChunkRecord` objects
5. Optionally upsert to MongoDB (`open_day_knowledge.kb_chuncks`)

## Current project status

Active, implemented modules:
- `schemas/models.py`: Pydantic models (`Document`, `Chunk`, `ChunkTags`, `ChunkRecord`)
- `services/ingestion_service.py`: ingestion pipeline and chunk record creation
- `services/embedding_service.py`: OpenAI embedding wrapper
- `services/mongo_repo.py`: MongoDB adapter with bulk upsert by `chunk_id`
- `test_ingestion_service.py`: CLI test runner for local and Mongo integration tests

In-progress/stub modules (currently empty):
- `app/main.py`
- `app/orchestrator.py`
- `agents/processor_agent.py`
- `services/retriever_service.py`
- `services/llm_services.py`

## Repository layout

Key paths:
- `data/`: input JSON files (currently includes `accommodation_halls.json`)
- `data/chunk_previews/`: generated preview outputs from tests
- `services/`: ingestion, embedding, mongo upload logic
- `schemas/`: shared data models
- `test_ingestion_service.py`: main test entrypoint
- `RAG_chatbot_demo.ipynb`: earlier notebook workflow (Mongo DB name: `open_day_knowledge`)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pymongo python-dotenv
```

`python-dotenv` is optional because the code has a fallback parser for `.env`, but installing it is recommended.

### 2. Configure environment

Create `.env` in project root:

```env
MONGODB_URI="your_mongodb_connection_string"
OPENAI_API_KEY="your_openai_api_key"
```

Notes:
- `MONGODB_URI` is required for Mongo upload tests.
- `OPENAI_API_KEY` is only required if you run with `--embedder openai`.

## How to test and run data integration right now

All commands below should be run from repo root:
`c:\Users\Emman\OneDrive\Documents\GitHub\PartD_GroupProject`

### A. Run ingestion test on all JSON files (local only, no Mongo)

Command:
```bash
python test_ingestion_service.py
```

What it does:
- Scans `data/*.json`
- Skips files ending with `_chunk_records_preview.json`
- Ingests each JSON file
- Prints record count and first chunk preview

Expected output (example):
```text
[accommodation_halls.json]
  records: 44
  first chunk id: 50f7658e8e5df42af88d80423c04ea0200678853
  first chunk preview: [0].name: Butler Court [0].type: hall ...
```

### B. Run ingestion test for one file

Command:
```bash
python test_ingestion_service.py --file accommodation_halls.json
```

What it does:
- Tests only `data/accommodation_halls.json`

Expected output:
- Same format as above, for that single file.

### C. Generate preview JSON files for created chunk records

Command:
```bash
python test_ingestion_service.py --write-previews
```

What it does:
- Runs ingestion locally
- Writes output records to:
  - `data/chunk_previews/<source>_chunk_records_preview.json`

Expected output (example):
```text
[accommodation_halls.json]
  records: 44
  first chunk id: 50f7658e8e5df42af88d80423c04ea0200678853
  first chunk preview: [0].name: Butler Court [0].type: hall ...
  wrote preview: data\chunk_previews\accommodation_halls_chunk_records_preview.json
```

### D. Mongo integration test: upload accommodation chunks to MongoDB

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder fake
```

Default Mongo target:
- Database: `open_day_knowledge` (same as notebook)
- Collection: `kb_chuncks`

What it does:
- Loads `data/accommodation_halls.json`
- Ingests and builds chunk records
- Upserts to Mongo with `chunk_id` as unique match key
- Verifies upload count by `source_id=accommodation_halls`

Expected output (example):
```text
[MONGO UPLOAD TEST]
  database: open_day_knowledge
  collection: kb_chuncks
  local records built: 44
  records returned: 44
  records in Mongo by source_id=accommodation_halls: 44
```

### E. Override Mongo target database/collection

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder fake --mongo-db open_day_knowledge --mongo-collection kb_chuncks
```

Useful when testing in separate environments.

### F. Use OpenAI embeddings instead of fake embeddings

Command:
```bash
python test_ingestion_service.py --test-mongo-upload --embedder openai
```

Requirements:
- `OPENAI_API_KEY` must be set
- internet access to OpenAI API

## What gets stored in Mongo

Uploaded document shape is based on `ChunkRecord` and includes:
- `chunk_id`, `source_id`, `source_type`
- `title`, `url`
- `text`
- `embedding`
- `domain`, `entity_tags`, `section`, `order`
- `metadata`, `version`

Upsert behavior:
- `MongoRepo.upsert_chunks()` uses `ReplaceOne({"chunk_id": rec.chunk_id}, ..., upsert=True)`
- Re-running ingestion updates existing chunk records instead of duplicating by `chunk_id`.

## Troubleshooting

### `MONGODB_URI is not set`

Fix:
- add `MONGODB_URI=...` to `.env`
- or export it in your shell before running commands

### `ModuleNotFoundError` for dependencies

Fix:
```bash
pip install -r requirements.txt
pip install pymongo python-dotenv
```

### Mongo DNS / timeout / connection errors

Likely causes:
- no internet access
- blocked DNS/network policy
- invalid Atlas/network allowlist settings

### Record count mismatch

If `local records built` and Mongo count differ:
- rerun command and check for write errors
- verify `source_id` filter is correct
- check collection name is exactly `kb_chuncks`

## Quick command reference

```bash
# Local ingestion test (all JSON files)
python test_ingestion_service.py

# Local ingestion test (single file)
python test_ingestion_service.py --file accommodation_halls.json

# Local ingestion + preview files
python test_ingestion_service.py --write-previews

# Mongo upload integration test (recommended first pass)
python test_ingestion_service.py --test-mongo-upload --embedder fake

# Mongo upload with OpenAI embeddings
python test_ingestion_service.py --test-mongo-upload --embedder openai
```
