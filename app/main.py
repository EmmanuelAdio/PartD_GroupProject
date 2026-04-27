from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
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

def _frontend_origins() -> List[str]:
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    raw = os.getenv("FRONTEND_ORIGINS")
    if not raw:
        return default_origins

    origins: List[str] = []
    seen = set()
    for part in raw.split(","):
        value = part.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        origins.append(value)
    return origins or default_origins

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
        version=payload.get("version", "ingest-v3"),
        json_group_size=int(payload.get("json_group_size", 30)),
    )
    return orchestrator.ingest(
        file_name=payload.get("file"),
        incremental=bool(payload.get("incremental", True)),
        clear_source_before_reingest=bool(payload.get("clear_source_before_reingest", True)),
    )


ingestion_orchestrator: Optional[IngestionOrchestrator] = None
query_orchestrator: Optional[QueryOrchestrator] = None


def _shape_query_response(result: Dict[str, Any], *, debug: bool) -> Dict[str, Any]:
    if debug:
        return result

    answerer_run = result.get("answerer_run", {}) if isinstance(result, dict) else {}
    return {
        "query": result.get("user_query"),
        "answer": answerer_run.get("answer"),
        "grounded": answerer_run.get("grounded"),
        "confidence": answerer_run.get("confidence"),
        "citations": answerer_run.get("citations", []),
    }


def _feedback_log_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "feedback_events.jsonl"


def _feedback_retry_succeeded(result: Dict[str, Any]) -> bool:
    decision = result.get("orchestration_decision", {}) if isinstance(result, dict) else {}
    runtime_action = decision.get("runtime_action")
    return runtime_action in {"pass", "ask_clarification"}


def _log_feedback_event(
    *,
    user_query: str,
    last_answer: str,
    resolved: bool,
    reason: Optional[str],
    retry_succeeded: bool,
) -> None:
    try:
        path = _feedback_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": user_query,
            "last_answer": last_answer,
            "resolved": bool(resolved),
            "reason": reason,
            "retry_succeeded": bool(retry_succeeded),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never block the user-facing response path.
        return

def _log_query_pipeline(user_query: str, result: Dict[str, Any]) -> None:
    """Print a one-line pipeline summary to the terminal for demo visibility."""
    try:
        plan = result.get("processor_plan", {})
        retrieval = result.get("retrieval_run", {})
        answerer = result.get("answerer_run", {})
        evaluator = result.get("evaluator_run", {})
        decision = result.get("orchestration_decision", {})
        timing = result.get("timing_ms", {})
        diag = retrieval.get("diagnostics", {})
        conf = float(answerer.get("confidence") or 0)
        print(
            f'[query] "{user_query[:70]}"'
            f' | domain={plan.get("domain") or "none"}'
            f' | chunks={retrieval.get("result_count", 0)}'
            f' | vector={diag.get("vector_mode", "?")}'
            f' | grounded={answerer.get("grounded")}'
            f' | conf={conf:.2f}'
            f' | verdict={evaluator.get("verdict", "?")}'
            f' | action={decision.get("runtime_action", "?")}'
            f' | {timing.get("total", "?")}ms'
        )
    except Exception:
        pass


if FastAPI is not None:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global ingestion_orchestrator, query_orchestrator
        default_embedder_backend = _default_query_embedder_backend()
        openai_ok = bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"))
        mongo_ok = bool(os.getenv("MONGODB_URI"))
        print(
            f"[startup] embedder={default_embedder_backend}"
            f" | openai_key={'set' if openai_ok else 'MISSING'}"
            f" | mongodb_uri={'set' if mongo_ok else 'MISSING'}"
        )
        try:
            ingestion_orchestrator = IngestionOrchestrator(
                embedder_backend=default_embedder_backend,
            )
            query_orchestrator = QueryOrchestrator(
                embedder_backend=default_embedder_backend,
            )
            print("[startup] Orchestrators ready.")
        except Exception as exc:
            print(f"[startup ERROR] Orchestrator init failed: {exc}")
            raise
        yield

    app = FastAPI(
        title="Loughborough RAG API",
        version="1.1.0",
        lifespan=lifespan,
    )
    if CORSMiddleware is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_frontend_origins(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    class IngestFileRequest(BaseModel):
        file_path: str
        incremental: bool = True

    class IngestPayloadRequest(BaseModel):
        data: Union[Dict[str, Any], List[Any]]
        source_id: str
        title: Optional[str] = None
        incremental: bool = False

    class QueryRequest(BaseModel):
        query: str
        top_k: Optional[int] = None
        debug: bool = False

    FeedbackReason = Literal["wrong_topic", "too_vague", "missing_detail", "incorrect"]

    class FeedbackRequest(BaseModel):
        user_query: str
        last_answer: str
        resolved: bool
        reason: Optional[FeedbackReason] = None

    class FeedbackResponse(BaseModel):
        action: Literal["acknowledged", "retried"]
        answer_payload: Optional[Dict[str, Any]] = None

    @app.get("/health")
    def health():
        if query_orchestrator is None:
            return {"status": "starting", "detail": "Orchestrators not yet ready."}
        result: Dict[str, Any] = {"status": "ok"}
        result["openai_configured"] = bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"))
        result["mongo"] = query_orchestrator._get_mongo_status()
        result["index_health"] = query_orchestrator._get_index_health()
        if result["mongo"].get("error"):
            result["status"] = "degraded"
        return result

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
            result = query_orchestrator.run(
                user_query=req.query,
                top_k_override=req.top_k,
            )
            _log_query_pipeline(req.query, result)
            return _shape_query_response(result, debug=req.debug)
        except Exception as e:
            print(f"[query ERROR] {req.query!r}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/feedback", response_model=FeedbackResponse)
    def run_feedback(req: FeedbackRequest):
        try:
            if query_orchestrator is None:
                raise RuntimeError("QueryOrchestrator is not ready.")

            if req.resolved:
                _log_feedback_event(
                    user_query=req.user_query,
                    last_answer=req.last_answer,
                    resolved=req.resolved,
                    reason=req.reason,
                    retry_succeeded=False,
                )
                return FeedbackResponse(action="acknowledged", answer_payload=None)

            result = query_orchestrator.run_with_feedback(
                user_query=req.user_query,
                last_answer=req.last_answer,
                reason=req.reason,
            )
            answer_payload = _shape_query_response(result, debug=False)
            _log_feedback_event(
                user_query=req.user_query,
                last_answer=req.last_answer,
                resolved=req.resolved,
                reason=req.reason,
                retry_succeeded=_feedback_retry_succeeded(result),
            )
            return FeedbackResponse(
                action="retried",
                answer_payload=answer_payload,
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
    parser.add_argument("--version", default="ingest-v3")
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
