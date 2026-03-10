from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

from app.orchestrator import IngestionOrchestrator, QueryOrchestrator

def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return

    # Fallback parser when python-dotenv is not installed.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_project_env()

def _default_query_embedder_backend() -> str:
    return "openai" if (os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")) else "fake"

def handle_ingestion_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """API-style ingestion handler you can call from routes or other services."""
    orchestrator = IngestionOrchestrator(
        data_dir=payload.get("data_dir", "data"),
        mongo_db=payload.get("mongo_db", "open_day_knowledge"),
        mongo_collection=payload.get("mongo_collection", "kb_chuncks"),
        embedder_backend=payload.get("embedder", "fake"),
        embedding_model=payload.get("embedding_model", "text-embedding-3-small"),
        tagger_mode=payload.get("tagger", "heuristic"),
        llm_model=payload.get("llm_model", "gpt-4o-mini"),
        version=payload.get("version", "ingest-v2"),
        json_group_size=int(payload.get("json_group_size", 30)),
    )
    return orchestrator.ingest(
        file_name=payload.get("file"),
        incremental=bool(payload.get("incremental", True)),
        clear_source_before_reingest=bool(payload.get("clear_source_before_reingest", True)),
    )


ingestion_orchestrator: Optional[IngestionOrchestrator] = None
query_orchestrator: Optional[QueryOrchestrator] = None

if FastAPI is not None:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global ingestion_orchestrator, query_orchestrator
        default_embedder_backend = _default_query_embedder_backend()
        ingestion_orchestrator = IngestionOrchestrator(
            embedder_backend=default_embedder_backend,
        )
        query_orchestrator = QueryOrchestrator(
            embedder_backend=default_embedder_backend,
        )
        yield

    app = FastAPI(
        title="Loughborough RAG API",
        version="1.1.0",
        lifespan=lifespan,
    )

    class IngestFileRequest(BaseModel):
        file_path: str
        incremental: bool = True

    class IngestPayloadRequest(BaseModel):
        data: Dict[str, Any] | List[Any]
        source_id: str
        title: Optional[str] = None
        incremental: bool = False

    class QueryRequest(BaseModel):
        query: str
        top_k: Optional[int] = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest/file")
    def ingest_file(req: IngestFileRequest):
        try:
            if ingestion_orchestrator is None:
                raise RuntimeError("Orchestrator is not ready.")
            return ingestion_orchestrator.ingest_file(
                file_path=req.file_path,
                incremental=req.incremental,
                clear_source_before_reingest=True,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest/all")
    def ingest_all(incremental: bool = True):
        try:
            if ingestion_orchestrator is None:
                raise RuntimeError("Orchestrator is not ready.")
            return ingestion_orchestrator.ingest_all_data_files(
                incremental=incremental,
                clear_source_before_reingest=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest/payload")
    def ingest_payload(req: IngestPayloadRequest):
        try:
            if ingestion_orchestrator is None:
                raise RuntimeError("Orchestrator is not ready.")
            return ingestion_orchestrator.ingest_json_payload(
                data=req.data,
                source_id=req.source_id,
                title=req.title,
                incremental=req.incremental,
                clear_source_before_reingest=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    def run_query(req: QueryRequest):
        try:
            if query_orchestrator is None:
                raise RuntimeError("QueryOrchestrator is not ready.")
            return query_orchestrator.run(
                user_query=req.query,
                top_k_override=req.top_k,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/status")
    def get_status():
        try:
            if query_orchestrator is None:
                raise RuntimeError("QueryOrchestrator is not ready.")
            return query_orchestrator.get_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
else:
    app = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initial/incremental ingestion into MongoDB.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--file", default=None, help="Optional single file name in data-dir.")
    parser.add_argument("--mongo-db", default="open_day_knowledge")
    parser.add_argument("--mongo-collection", default="kb_chuncks")
    parser.add_argument("--embedder", choices=["fake", "openai"], default="fake")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--tagger", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--version", default="ingest-v2")
    parser.add_argument("--json-group-size", type=int, default=30)
    parser.add_argument(
        "--full-reingest",
        action="store_true",
        help="Process all selected sources regardless of unchanged manifest.",
    )
    parser.add_argument(
        "--keep-existing-on-reingest",
        action="store_true",
        help="Do not clear existing source docs before reingesting changed sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "data_dir": args.data_dir,
        "file": args.file,
        "mongo_db": args.mongo_db,
        "mongo_collection": args.mongo_collection,
        "embedder": args.embedder,
        "embedding_model": args.embedding_model,
        "tagger": args.tagger,
        "llm_model": args.llm_model,
        "version": args.version,
        "json_group_size": args.json_group_size,
        "incremental": not args.full_reingest,
        "clear_source_before_reingest": not args.keep_existing_on_reingest,
    }
    summary = handle_ingestion_api(payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
