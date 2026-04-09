from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.answerer_agent import AnswererAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.processor_agent import ProcessorAgent
from schemas.models import (
    AnswerCitation,
    AnswerResult,
    EvaluationResult,
    EvidenceItem,
    RetrievalQuery,
)
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
        version: str = "ingest-v3",
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
        answerer_model: str = "gpt-4o-mini",
        vector_index_name: str = "kb_vector_index",
        atlas_search_index_name: str = "kb_text_index",
    ) -> None:
        self._mongo_db = mongo_db
        self._mongo_collection = mongo_collection

        self.repo = MongoRepo(db_name=mongo_db, collection_name=mongo_collection)
        self.embedder = self._build_embedder(embedder_backend, embedding_model)
        self.processor = ProcessorAgent(llm_model=processor_model)
        self.answerer = AnswererAgent(llm_model=answerer_model)
        self.evaluator = EvaluatorAgent(llm_model=answerer_model)
        self.max_revise_retries = 1
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

        retrieval_bundle = self._retrieve_with_fallback(
            user_query=user_query,
            plan=plan,
            attempt_start=1,
            phase="initial",
        )
        final_plan = retrieval_bundle["plan_used"]
        attempts_log = list(retrieval_bundle["attempts_log"])
        final_attempt_used = retrieval_bundle["attempt_used"]

        results, diag, retrieved_result_count = self._prepare_answer_inputs(
            user_query=user_query,
            plan=final_plan,
            evidence=retrieval_bundle["results"],
            diag=retrieval_bundle["diagnostics"],
        )

        draft_answer = self.answerer.answer(
            user_query=user_query,
            evidence_items=results,
            processor_plan=final_plan,
        )
        evaluation = self.evaluator.evaluate(
            user_query=user_query,
            retrieval_query=final_plan,
            evidence=results,
            draft_answer=draft_answer,
        )
        evaluation_history: List[Dict[str, Any]] = [evaluation.model_dump()]

        retries_used = 0
        while self._should_retry_after_evaluation(
            verdict=evaluation.verdict,
            retries_used=retries_used,
            max_retries=self.max_revise_retries,
        ):
            retries_used += 1
            revised_plan = self._apply_evaluator_suggested_filters(
                plan=final_plan,
                suggested_filters=evaluation.suggested_filters,
            )
            retry_bundle = self._retrieve_with_fallback(
                user_query=user_query,
                plan=revised_plan,
                attempt_start=len(attempts_log) + 1,
                phase=f"revise_{retries_used}",
            )
            final_plan = retry_bundle["plan_used"]
            final_attempt_used = retry_bundle["attempt_used"]
            attempts_log.extend(retry_bundle["attempts_log"])

            results, diag, retrieved_result_count = self._prepare_answer_inputs(
                user_query=user_query,
                plan=final_plan,
                evidence=retry_bundle["results"],
                diag=retry_bundle["diagnostics"],
            )
            draft_answer = self.answerer.answer(
                user_query=user_query,
                evidence_items=results,
                processor_plan=final_plan,
            )
            evaluation = self.evaluator.evaluate(
                user_query=user_query,
                retrieval_query=final_plan,
                evidence=results,
                draft_answer=draft_answer,
            )
            evaluation_history.append(evaluation.model_dump())

        effective_verdict = evaluation.verdict
        if effective_verdict == "revise" and retries_used >= self.max_revise_retries:
            effective_verdict = "fallback"

        runtime_action = self._decision_for_verdict(effective_verdict)
        final_answer = draft_answer
        if runtime_action == "ask_clarification":
            final_answer = self._build_clarification_answer(
                evaluation=evaluation,
                evidence=results,
            )
        elif runtime_action == "fallback":
            final_answer = self._build_safe_fallback_answer(evidence=results)

        evaluator_run = evaluation.model_dump()
        evaluator_run["effective_verdict"] = effective_verdict
        evaluator_run["history"] = evaluation_history

        return {
            "user_query": user_query,
            "mongo_status": mongo_status,
            "index_health": index_health,
            "processor_plan": final_plan.model_dump(),
            "answerer_run": final_answer.model_dump(),
            "evaluator_run": evaluator_run,
            "orchestration_decision": {
                "initial_verdict": evaluation_history[0].get("verdict"),
                "final_verdict": evaluation.verdict,
                "effective_verdict": effective_verdict,
                "runtime_action": runtime_action,
                "revise_retries_used": retries_used,
                "max_revise_retries": self.max_revise_retries,
            },
            "retrieval_run": {
                "attempt_used": final_attempt_used,
                "attempts_log": attempts_log,
                "result_count": len(results),
                "retrieved_result_count": retrieved_result_count,
                "diagnostics": diag,
                "evidence": [item.model_dump() for item in results],
            },
        }

    def _retrieve_with_fallback(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        attempt_start: int,
        phase: str,
    ) -> Dict[str, Any]:
        attempts_log: List[Dict[str, Any]] = []
        current_attempt = int(attempt_start)

        results, diag = self._retrieve(plan)
        attempts_log.append(
            {"attempt": current_attempt, "description": f"{phase}:full_plan", "hits": len(results)}
        )
        plan_used = plan
        current_attempt += 1

        if not results:
            relaxed = plan.model_copy(update={"domain": None, "domains": []})
            results, diag = self._retrieve(relaxed)
            attempts_log.append(
                {"attempt": current_attempt, "description": f"{phase}:relax_domain", "hits": len(results)}
            )
            plan_used = relaxed
            current_attempt += 1

        if not results:
            bare = RetrievalQuery(query_text=user_query, top_k=plan.top_k)
            results, diag = self._retrieve(bare)
            attempts_log.append(
                {"attempt": current_attempt, "description": f"{phase}:bare_query", "hits": len(results)}
            )
            plan_used = bare

        attempt_used = next((row["attempt"] for row in attempts_log if row["hits"] > 0), None)
        return {
            "plan_used": plan_used,
            "results": results,
            "diagnostics": diag,
            "attempts_log": attempts_log,
            "attempt_used": attempt_used,
        }

    def _prepare_answer_inputs(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        evidence: List[EvidenceItem],
        diag: Optional[Dict[str, Any]],
    ) -> tuple[List[EvidenceItem], Dict[str, Any], int]:
        retrieved_result_count = len(evidence)
        prepared = self._prepare_answer_evidence(
            user_query=user_query,
            plan=plan,
            evidence=evidence,
        )
        diagnostics = dict(diag or {})
        diagnostics["answer_evidence_expanded"] = len(prepared) > retrieved_result_count
        diagnostics["answer_evidence_count"] = len(prepared)
        return prepared, diagnostics, retrieved_result_count

    @staticmethod
    def _should_retry_after_evaluation(
        *,
        verdict: str,
        retries_used: int,
        max_retries: int,
    ) -> bool:
        return verdict == "revise" and retries_used < max_retries

    @staticmethod
    def _decision_for_verdict(verdict: str) -> str:
        if verdict == "pass":
            return "pass"
        if verdict == "ask_clarification":
            return "ask_clarification"
        if verdict == "fallback":
            return "fallback"
        if verdict == "revise":
            return "fallback"
        return "fallback"

    @staticmethod
    def _apply_evaluator_suggested_filters(
        *,
        plan: RetrievalQuery,
        suggested_filters: Optional[Dict[str, Any]],
    ) -> RetrievalQuery:
        payload = dict(suggested_filters or {})
        merged_domains = QueryOrchestrator._dedupe_values(
            [plan.domain, *(plan.domains or []), *QueryOrchestrator._coerce_str_list(payload.get("domains"))]
        )
        merged_sections = QueryOrchestrator._dedupe_values(
            [plan.section, *(plan.sections or []), *QueryOrchestrator._coerce_str_list(payload.get("sections"))]
        )
        merged_tags = QueryOrchestrator._dedupe_values(
            [*(plan.entity_tags or []), *QueryOrchestrator._coerce_str_list(payload.get("entity_tags"))]
        )

        top_k = int(payload.get("top_k", min(12, plan.top_k + 2)))
        top_k = max(1, min(50, top_k))

        update: Dict[str, Any] = {
            "top_k": top_k,
            "domain": merged_domains[0] if merged_domains else None,
            "domains": merged_domains,
            "section": merged_sections[0] if merged_sections else None,
            "sections": merged_sections,
            "entity_tags": merged_tags,
        }
        return plan.model_copy(update=update)

    @staticmethod
    def _build_clarification_answer(
        *,
        evaluation: EvaluationResult,
        evidence: List[EvidenceItem],
    ) -> AnswerResult:
        question = evaluation.clarification_question or "Could you clarify what specific course, hall, or topic you mean?"
        citations = QueryOrchestrator._first_url_citation(evidence)
        return AnswerResult(
            answer=f"To give you a reliable answer, I need one clarification: {question}",
            grounded=False,
            confidence=0.2,
            citations=citations,
            used_evidence_count=len(citations),
            fallback_used=True,
        )

    @staticmethod
    def _build_safe_fallback_answer(*, evidence: List[EvidenceItem]) -> AnswerResult:
        help_url = QueryOrchestrator._knowledgebase_help_url(evidence)
        citations = QueryOrchestrator._first_url_citation(evidence)
        answer = "I couldn't verify a reliable answer from the available evidence."
        if help_url:
            answer += f" Please check the official source: {help_url}"
        return AnswerResult(
            answer=answer,
            grounded=False,
            confidence=0.0,
            citations=citations,
            used_evidence_count=len(citations),
            fallback_used=True,
        )

    @staticmethod
    def _first_url_citation(evidence: List[EvidenceItem]) -> List[AnswerCitation]:
        for idx, item in enumerate(evidence, start=1):
            url = QueryOrchestrator._clean_text(str(item.url or ""))
            if not url:
                continue
            return [
                AnswerCitation(
                    evidence_id=idx,
                    chunk_id=item.chunk_id,
                    source_id=item.source_id,
                    source_type=item.source_type,
                    title=item.title,
                    section=item.section,
                    url=item.url,
                )
            ]
        return []

    @staticmethod
    def _knowledgebase_help_url(evidence: List[EvidenceItem]) -> str:
        for item in evidence:
            url = QueryOrchestrator._clean_text(str(item.url or ""))
            if url:
                return url
        return "https://www.lboro.ac.uk/"

    @staticmethod
    def _dedupe_values(values: List[Optional[str]]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in values:
            value = QueryOrchestrator._clean_text(str(raw or ""))
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

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

    def _prepare_answer_evidence(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        evidence: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        if not self._should_expand_accommodation_price_evidence(user_query, plan):
            return evidence

        expanded_docs = self.repo.collection.find(
            {
                "source_id": "accommodation_halls",
                "domain": "accommodation",
            },
            {
                "_id": 0,
                "chunk_id": 1,
                "source_id": 1,
                "source_type": 1,
                "title": 1,
                "url": 1,
                "text": 1,
                "domain": 1,
                "entity_tags": 1,
                "section": 1,
                "order": 1,
                "version": 1,
                "metadata": 1,
            },
        ).sort("order", 1)

        merged: List[EvidenceItem] = []
        seen = set()

        for item in evidence:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            merged.append(item)

        for doc in expanded_docs:
            try:
                item = EvidenceItem.model_validate(
                    {
                        **doc,
                        "score": 0.0,
                        "vector_score": None,
                        "text_score": None,
                        "retrieval_channels": [],
                    }
                )
            except Exception:
                continue
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            merged.append(item)

        return merged

    @staticmethod
    def _should_expand_accommodation_price_evidence(user_query: str, plan: RetrievalQuery) -> bool:
        query_lc = (user_query or "").lower()
        plan_domains = {str(v).lower() for v in [plan.domain, *(plan.domains or [])] if v}
        is_accommodation = "accommodation" in plan_domains or "accommodation" in query_lc or "hall" in query_lc
        has_extreme = any(
            token in query_lc
            for token in ("cheapest", "lowest", "least expensive", "minimum", "most expensive", "highest", "maximum", "priciest")
        )
        has_price_signal = any(
            token in query_lc
            for token in ("price", "prices", "cost", "costs", "fee", "fees", "rent", "weekly", "per week")
        )
        return is_accommodation and has_extreme and has_price_signal

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
