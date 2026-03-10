from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.processor_agent import ProcessorAgent
from schemas.models import RetrievalQuery
from services.embedding_service import DeterministicEmbeddingService, EmbeddingService
from services.ingestion_service import IngestionService
from services.llm_services import LLMService
from services.mongo_repo import MongoRepo
from services.retriever_service import RetrieverService


class IngestionOrchestrator:
    """Coordinates source discovery, incremental checks, and ingestion execution."""

    def __init__(
        self,
        data_dir: str = "data",
        mongo_db: str = "open_day_knowledge",
        mongo_collection: str = "kb_chuncks",
        embedder_backend: str = "fake",
        embedding_model: str = "text-embedding-3-small",
        tagger_mode: str = "heuristic",
        llm_model: str = "gpt-4o-mini",
        version: str = "ingest-v2",
        json_group_size: int = 30,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.repo = MongoRepo(db_name=mongo_db, collection_name=mongo_collection)
        self.embedder = self._build_embedder(embedder_backend, embedding_model)
        self.tagger = self._build_tagger(tagger_mode, llm_model)
        self.ingestion = IngestionService(
            repo=self.repo,
            embedder=self.embedder,
            tagger=self.tagger,
            version=version,
            json_group_size=json_group_size,
        )

        embedder_name = self.embedder.get_model_name() if hasattr(self.embedder, "get_model_name") else self.embedder.__class__.__name__
        tagger_name = llm_model if self.tagger is not None else "heuristic"
        self.pipeline_signature = {
            "version": version,
            "embedder_backend": embedder_backend,
            "embedder_name": embedder_name,
            "tagger_mode": tagger_mode,
            "tagger_name": tagger_name,
            "json_group_size": json_group_size,
        }
        self.pipeline_hash = self._stable_hash_json(self.pipeline_signature)

    def ingest(
        self,
        file_name: Optional[str] = None,
        incremental: bool = True,
        clear_source_before_reingest: bool = True,
    ) -> Dict[str, Any]:
        paths = self._resolve_paths(file_name=file_name)
        summary: Dict[str, Any] = {
            "ingested_sources": [],
            "skipped_sources": [],
            "total_new_records": 0,
            "total_sources_seen": len(paths),
            "pipeline_signature": self.pipeline_signature,
            "pipeline_hash": self.pipeline_hash,
        }

        for path in paths:
            source_id = path.stem
            source_hash = self._sha256_file(path)
            existing_count = self.repo.count_source_records(source_id)

            if incremental and self._is_source_unchanged(source_id=source_id, source_hash=source_hash, existing_count=existing_count):
                summary["skipped_sources"].append(
                    {
                        "source_id": source_id,
                        "source_path": str(path),
                        "reason": "unchanged_source_and_pipeline",
                        "records_in_db": existing_count,
                    }
                )
                continue

            if clear_source_before_reingest and existing_count > 0:
                self.repo.delete_source_records(source_id)

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            count, _ = self.ingestion.ingest_json(
                data=data,
                source_id=source_id,
                title=source_id.replace("_", " ").title(),
            )
            records_in_db = self.repo.count_source_records(source_id)

            self.repo.upsert_source_manifest(
                source_id=source_id,
                source_hash=source_hash,
                pipeline_hash=self.pipeline_hash,
                pipeline_signature=self.pipeline_signature,
                source_path=str(path),
                records_in_db=records_in_db,
            )

            summary["ingested_sources"].append(
                {
                    "source_id": source_id,
                    "source_path": str(path),
                    "new_records": count,
                    "records_in_db": records_in_db,
                }
            )
            summary["total_new_records"] += int(count)

        return summary

    def ingest_all_data_files(
        self,
        incremental: bool = True,
        clear_source_before_reingest: bool = True,
    ) -> Dict[str, Any]:
        return self.ingest(
            file_name=None,
            incremental=incremental,
            clear_source_before_reingest=clear_source_before_reingest,
        )

    def ingest_file(
        self,
        file_path: str,
        incremental: bool = True,
        clear_source_before_reingest: bool = True,
    ) -> Dict[str, Any]:
        file_name = Path(file_path).name
        return self.ingest(
            file_name=file_name,
            incremental=incremental,
            clear_source_before_reingest=clear_source_before_reingest,
        )

    def ingest_json_payload(
        self,
        data: Any,
        source_id: str,
        title: Optional[str] = None,
        incremental: bool = False,
        clear_source_before_reingest: bool = True,
    ) -> Dict[str, Any]:
        source_hash = self._stable_hash_json({"payload": data})
        existing_count = self.repo.count_source_records(source_id)
        if incremental and self._is_source_unchanged(source_id, source_hash, existing_count):
            return {
                "source_id": source_id,
                "chunks_upserted": 0,
                "records_in_db": existing_count,
                "skipped": True,
                "reason": "unchanged_payload_and_pipeline",
            }

        if clear_source_before_reingest and existing_count > 0:
            self.repo.delete_source_records(source_id)

        count, _ = self.ingestion.ingest_json(data=data, source_id=source_id, title=title)
        records_in_db = self.repo.count_source_records(source_id)
        self.repo.upsert_source_manifest(
            source_id=source_id,
            source_hash=source_hash,
            pipeline_hash=self.pipeline_hash,
            pipeline_signature=self.pipeline_signature,
            source_path="payload",
            records_in_db=records_in_db,
        )
        return {
            "source_id": source_id,
            "chunks_upserted": count,
            "records_in_db": records_in_db,
            "skipped": False,
        }

    def _resolve_paths(self, file_name: Optional[str]) -> List[Path]:
        if file_name:
            path = self.data_dir / file_name
            if not path.exists():
                raise FileNotFoundError(f"JSON file not found: {path}")
            return [path]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        paths = []
        for path in sorted(self.data_dir.glob("*.json")):
            if path.name.endswith("_chunk_records_preview.json"):
                continue
            paths.append(path)
        return paths

    def _is_source_unchanged(self, source_id: str, source_hash: str, existing_count: int) -> bool:
        manifest = self.repo.get_source_manifest(source_id)
        if not manifest:
            return False
        if manifest.get("source_hash") != source_hash:
            return False
        if manifest.get("pipeline_hash") != self.pipeline_hash:
            return False
        return existing_count > 0

    @staticmethod
    def _build_embedder(embedder_backend: str, embedding_model: str):
        if embedder_backend == "fake":
            return DeterministicEmbeddingService(dim=64)
        if embedder_backend == "openai":
            return EmbeddingService(model=embedding_model)
        raise ValueError("embedder_backend must be 'fake' or 'openai'.")

    @staticmethod
    def _build_tagger(tagger_mode: str, llm_model: str):
        if tagger_mode == "heuristic":
            return None
        if tagger_mode == "llm":
            return LLMService(model=llm_model)
        raise ValueError("tagger_mode must be 'heuristic' or 'llm'.")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _stable_hash_json(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class QueryOrchestrator:
    """Wraps ProcessorAgent + RetrieverService with progressive fallback on zero results.

    Fallback levels:
      1. Full plan (domain filter + planned query_text)
      2. Relax domain — remove the hard domain filter, keep planned query_text
      3. Bare query — raw user input, no filters at all
    """

    def __init__(
        self,
        mongo_db: str = "open_day_knowledge",
        mongo_collection: str = "kb_chuncks",
        embedder_backend: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        processor_model: str = "gpt-4o-mini",
        vector_index_name: str = "kb_vector_index",
        atlas_search_index_name: str = "kb_text_index",
    ) -> None:
        self._mongo_db = mongo_db
        self._mongo_collection = mongo_collection

        self.repo = MongoRepo(db_name=mongo_db, collection_name=mongo_collection)
        self.embedder = self._build_embedder(embedder_backend, embedding_model)
        self.processor = ProcessorAgent(llm_model=processor_model)
        self.retriever = RetrieverService(
            repo=self.repo,
            embedder=self.embedder,
            vector_index_name=vector_index_name,
            atlas_search_index_name=atlas_search_index_name,
        )

    def run(self, user_query: str, top_k_override: Optional[int] = None) -> Dict[str, Any]:
        """Plan, retrieve with fallback, return structured response."""
        mongo_status = self._get_mongo_status()
        index_health = self._get_index_health()

        plan = self.processor.process(user_query)
        if top_k_override is not None:
            plan = plan.model_copy(update={"top_k": int(top_k_override)})

        attempts_log: List[Dict[str, Any]] = []

        # Attempt 1: full plan
        results, diag = self._retrieve(plan)
        attempts_log.append({"attempt": 1, "description": "full_plan", "hits": len(results)})

        # Attempt 2: drop domain filter
        if not results:
            relaxed = plan.model_copy(update={"domain": None, "domains": []})
            results, diag = self._retrieve(relaxed)
            attempts_log.append({"attempt": 2, "description": "relax_domain", "hits": len(results)})

        # Attempt 3: bare raw query, no filters
        if not results:
            bare = RetrievalQuery(query_text=user_query, top_k=plan.top_k)
            results, diag = self._retrieve(bare)
            attempts_log.append({"attempt": 3, "description": "bare_query", "hits": len(results)})

        attempt_used = next((a["attempt"] for a in attempts_log if a["hits"] > 0), None)

        return {
            "user_query": user_query,
            "mongo_status": mongo_status,
            "index_health": index_health,
            "processor_plan": plan.model_dump(),
            "retrieval_run": {
                "attempt_used": attempt_used,
                "attempts_log": attempts_log,
                "result_count": len(results),
                "diagnostics": diag,
                "evidence": [item.model_dump() for item in results],
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Lightweight status check — no LLM or retrieval calls."""
        return {
            "mongo_status": self._get_mongo_status(),
            "index_health": self._get_index_health(),
        }

    def _retrieve(self, plan: RetrievalQuery):
        results = self.retriever.retrieve(plan)
        diag = self.retriever.get_last_query_diagnostics()
        return results, diag

    def _get_mongo_status(self) -> Dict[str, Any]:
        doc_count = self.repo.collection.count_documents({})
        return {
            "database": self._mongo_db,
            "collection": self._mongo_collection,
            "total_chunk_docs": doc_count,
        }

    def _get_index_health(self) -> Dict[str, Any]:
        r = self.retriever.check_index_health()
        return {
            "healthy": r.is_healthy,
            "vector_index_found": r.vector_index_found,
            "vector_index_status": r.vector_index_status,
            "vector_index_dimensions": r.vector_index_dimensions,
            "text_index_found": r.text_index_found,
            "text_index_status": r.text_index_status,
            "errors": r.errors,
        }

    @staticmethod
    def _build_embedder(embedder_backend: str, embedding_model: str):
        if embedder_backend == "fake":
            return DeterministicEmbeddingService(dim=64)
        if embedder_backend == "openai":
            return EmbeddingService(model=embedding_model)
        raise ValueError("embedder_backend must be 'fake' or 'openai'.")


# Backward-compatible alias for existing imports.
Orchestrator = IngestionOrchestrator
