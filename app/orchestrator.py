from __future__ import annotations

import hashlib
import json
import time
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

    # Lightweight pre-checks run before retrieval/generation.
    # Each entry is (lowercase_substring, reason_code).
    # Order matters: first match wins.
    _SAFETY_PATTERNS: List[tuple] = [
        # Explicit instruction to fabricate or bypass the knowledge base
        ("make up",                         "instruction_to_guess_or_ignore_evidence"),
        ("just guess",                      "instruction_to_guess_or_ignore_evidence"),
        ("ignore your data",                "instruction_to_guess_or_ignore_evidence"),
        ("ignore the evidence",             "instruction_to_guess_or_ignore_evidence"),
        ("even if the evidence is missing", "instruction_to_guess_or_ignore_evidence"),
        ("even if you don't know",          "instruction_to_guess_or_ignore_evidence"),
        ("even if evidence is missing",     "instruction_to_guess_or_ignore_evidence"),
        # Private or confidential data requests
        ("confidential",                    "private_information_request"),
        ("medical history",                 "private_information_request"),
        ("private data",                    "private_information_request"),
        ("student data",                    "private_information_request"),
        ("personal data",                   "private_information_request"),
        # Future price forecasts (knowledge base only covers current data)
        (" 2030",                           "unsupported_future_claim"),
        (" 2031",                           "unsupported_future_claim"),
        (" 2032",                           "unsupported_future_claim"),
        (" 2033",                           "unsupported_future_claim"),
        (" 2034",                           "unsupported_future_claim"),
        (" 2035",                           "unsupported_future_claim"),
        ("future prices",                   "unsupported_future_claim"),
        ("will prices be",                  "unsupported_future_claim"),
        ("will the price",                  "unsupported_future_claim"),
        ("price forecast",                  "unsupported_future_claim"),
        # Requests for admissions guarantees
        ("can you guarantee",               "guarantee_request"),
        ("guarantee i'll",                  "guarantee_request"),
        ("guarantee i will",                "guarantee_request"),
        ("guaranteed admission",            "guarantee_request"),
        ("guaranteed entry",                "guarantee_request"),
        ("guarantee my place",              "guarantee_request"),
    ]

    _SAFETY_MESSAGES: Dict[str, str] = {
        "instruction_to_guess_or_ignore_evidence": (
            "I can only answer using verified information from the university knowledge base. "
            "I'm unable to make up, guess, or bypass the available evidence. "
            "Please ask a specific question and I'll do my best to help."
        ),
        "private_information_request": (
            "I can't help with private, confidential, or personal information. "
            "I can only use publicly available, verified university information. "
            "For sensitive enquiries, please contact the university directly or speak to a member of staff."
        ),
        "unsupported_future_claim": (
            "I don't have information about future prices or forecasts — my knowledge covers current, "
            "verified university data only. "
            "For the most up-to-date information please check the official Loughborough University website: "
            "https://www.lboro.ac.uk/"
        ),
        "guarantee_request": (
            "I'm not able to make guarantees about admissions or outcomes. "
            "Entry decisions are made by the Admissions team based on your individual application. "
            "For admissions guidance please visit: https://www.lboro.ac.uk/study/undergraduate/apply/"
        ),
    }

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
        self.max_feedback_retries = 1
        self.retriever = RetrieverService(
            repo=self.repo,
            embedder=self.embedder,
            vector_index_name=vector_index_name,
            atlas_search_index_name=atlas_search_index_name,
        )

    def run(self, user_query: str, top_k_override: Optional[int] = None) -> Dict[str, Any]:
        """Plan, retrieve with fallback, return structured response."""
        guard_triggered, guard_reason = self._safety_guard_check(user_query)
        if guard_triggered:
            return self._safety_guard_response(user_query, guard_reason)
        total_started = time.perf_counter()
        plan, processor_time_ms = self._plan_query(
            user_query=user_query,
            top_k_override=top_k_override,
        )
        retriever_time_ms = 0.0
        answerer_time_ms = 0.0
        evaluator_time_ms = 0.0
        mongo_status = self._get_mongo_status()
        index_health = self._get_index_health()
        cycle = self._run_answer_cycle(
            user_query=user_query,
            plan=plan,
            attempt_start=1,
            phase="initial",
        )
        retriever_time_ms += cycle["timing_ms"]["retriever"]
        answerer_time_ms += cycle["timing_ms"]["answerer"]
        evaluator_time_ms += cycle["timing_ms"]["evaluator"]
        final_plan = cycle["plan_used"]
        attempts_log = list(cycle["attempts_log"])
        final_attempt_used = cycle["attempt_used"]
        results = cycle["evidence"]
        diag = cycle["diagnostics"]
        retrieved_result_count = cycle["retrieved_result_count"]
        draft_answer = cycle["draft_answer"]
        evaluation = cycle["evaluation"]
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
            cycle = self._run_answer_cycle(
                user_query=user_query,
                plan=revised_plan,
                attempt_start=len(attempts_log) + 1,
                phase=f"revise_{retries_used}",
            )
            retriever_time_ms += cycle["timing_ms"]["retriever"]
            answerer_time_ms += cycle["timing_ms"]["answerer"]
            evaluator_time_ms += cycle["timing_ms"]["evaluator"]
            final_plan = cycle["plan_used"]
            final_attempt_used = cycle["attempt_used"]
            attempts_log.extend(cycle["attempts_log"])
            results = cycle["evidence"]
            diag = cycle["diagnostics"]
            retrieved_result_count = cycle["retrieved_result_count"]
            draft_answer = cycle["draft_answer"]
            evaluation = cycle["evaluation"]
            evaluation_history.append(evaluation.model_dump())

        final_answer, effective_verdict, runtime_action = self._finalize_runtime_answer(
            evaluation=evaluation,
            draft_answer=draft_answer,
            evidence=results,
            retries_used=retries_used,
            max_retries=self.max_revise_retries,
        )

        evaluator_run = evaluation.model_dump()
        evaluator_run["effective_verdict"] = effective_verdict
        evaluator_run["history"] = evaluation_history
        timing_ms = {
            "processor": round(processor_time_ms, 3),
            "retriever": round(retriever_time_ms, 3),
            "answerer": round(answerer_time_ms, 3),
            "evaluator": round(evaluator_time_ms, 3),
            "total": round((time.perf_counter() - total_started) * 1000.0, 3),
        }

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
            "timing_ms": timing_ms,
        }

    def run_with_feedback(
        self,
        *,
        user_query: str,
        last_answer: str,
        reason: Optional[str] = None,
        top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Re-run the pipeline with broadened retrieval after a negative feedback signal.

        Called when the user indicates the previous answer did not resolve their question.
        Retrieval is broadened (higher top_k, looser domain filter) and the prior answer
        is evaluated against the new evidence before generating a revised response.

        Args:
            user_query: The original user question.
            last_answer: The answer the user marked as unhelpful.
            reason: Optional feedback reason from the frontend
                (``wrong_topic`` | ``too_vague`` | ``missing_detail`` | ``incorrect``).
                Controls retry strategy: ``too_vague`` prefers clarification,
                ``incorrect`` applies a conservative confidence threshold.
            top_k_override: Optional hard cap on the number of chunks retrieved.

        Returns:
            Response dict in the same shape as ``run()``.
        """
        guard_triggered, guard_reason = self._safety_guard_check(user_query)
        if guard_triggered:
            return self._safety_guard_response(user_query, guard_reason)
        total_started = time.perf_counter()
        retriever_time_ms = 0.0
        answerer_time_ms = 0.0
        evaluator_time_ms = 0.0
        mongo_status = self._get_mongo_status()
        index_health = self._get_index_health()

        base_plan, processor_time_ms = self._plan_query(
            user_query=user_query,
            top_k_override=top_k_override,
        )
        broadened_plan = base_plan.model_copy(
            update={
                "top_k": self._feedback_top_k(
                    base_plan.top_k,
                    top_k_override=top_k_override,
                )
            }
        )

        initial_retrieval = self._retrieve_prepared_evidence(
            user_query=user_query,
            plan=broadened_plan,
            attempt_start=1,
            phase="feedback_initial",
        )
        retriever_time_ms += initial_retrieval["timing_ms"]["retriever"]
        evaluation_input = self._coerce_feedback_draft_answer(last_answer)
        initial_feedback_eval, evaluation_time = self._evaluate_answer(
            user_query=user_query,
            plan=initial_retrieval["plan_used"],
            evidence=initial_retrieval["evidence"],
            draft_answer=evaluation_input,
        )
        evaluator_time_ms += evaluation_time
        evaluation_history: List[Dict[str, Any]] = [initial_feedback_eval.model_dump()]

        if self._should_clarify_on_feedback(
            evaluation=initial_feedback_eval,
            reason=reason,
        ):
            final_answer = self._build_clarification_answer(
                evaluation=initial_feedback_eval,
                evidence=initial_retrieval["evidence"],
            )
            effective_verdict = "ask_clarification"
            runtime_action = "ask_clarification"
            final_plan = initial_retrieval["plan_used"]
            attempts_log = list(initial_retrieval["attempts_log"])
            final_attempt_used = initial_retrieval["attempt_used"]
            results = initial_retrieval["evidence"]
            diag = initial_retrieval["diagnostics"]
            retrieved_result_count = initial_retrieval["retrieved_result_count"]
            feedback_retry_used = 0
            final_evaluation = initial_feedback_eval
        else:
            revised_plan = self._build_feedback_retry_plan(
                base_plan=broadened_plan,
                current_plan=initial_retrieval["plan_used"],
                evaluation=initial_feedback_eval,
                reason=reason,
            )
            retry_cycle = self._run_answer_cycle(
                user_query=user_query,
                plan=revised_plan,
                attempt_start=len(initial_retrieval["attempts_log"]) + 1,
                phase="feedback_retry",
            )
            retriever_time_ms += retry_cycle["timing_ms"]["retriever"]
            answerer_time_ms += retry_cycle["timing_ms"]["answerer"]
            evaluator_time_ms += retry_cycle["timing_ms"]["evaluator"]
            evaluation_history.append(retry_cycle["evaluation"].model_dump())

            final_answer, effective_verdict, runtime_action = self._finalize_runtime_answer(
                evaluation=retry_cycle["evaluation"],
                draft_answer=retry_cycle["draft_answer"],
                evidence=retry_cycle["evidence"],
                retries_used=self.max_feedback_retries,
                max_retries=self.max_feedback_retries,
                conservative_fallback=(reason == "incorrect"),
            )
            final_plan = retry_cycle["plan_used"]
            attempts_log = list(initial_retrieval["attempts_log"]) + list(retry_cycle["attempts_log"])
            final_attempt_used = retry_cycle["attempt_used"]
            results = retry_cycle["evidence"]
            diag = retry_cycle["diagnostics"]
            retrieved_result_count = retry_cycle["retrieved_result_count"]
            feedback_retry_used = 1
            final_evaluation = retry_cycle["evaluation"]

        evaluator_run = final_evaluation.model_dump()
        evaluator_run["effective_verdict"] = effective_verdict
        evaluator_run["history"] = evaluation_history
        timing_ms = {
            "processor": round(processor_time_ms, 3),
            "retriever": round(retriever_time_ms, 3),
            "answerer": round(answerer_time_ms, 3),
            "evaluator": round(evaluator_time_ms, 3),
            "total": round((time.perf_counter() - total_started) * 1000.0, 3),
        }

        return {
            "user_query": user_query,
            "mongo_status": mongo_status,
            "index_health": index_health,
            "processor_plan": final_plan.model_dump(),
            "answerer_run": final_answer.model_dump(),
            "evaluator_run": evaluator_run,
            "orchestration_decision": {
                "initial_verdict": evaluation_history[0].get("verdict"),
                "final_verdict": final_evaluation.verdict,
                "effective_verdict": effective_verdict,
                "runtime_action": runtime_action,
                "feedback_retry_used": feedback_retry_used,
                "max_feedback_retries": self.max_feedback_retries,
                "reason": reason,
            },
            "retrieval_run": {
                "attempt_used": final_attempt_used,
                "attempts_log": attempts_log,
                "result_count": len(results),
                "retrieved_result_count": retrieved_result_count,
                "diagnostics": diag,
                "evidence": [item.model_dump() for item in results],
            },
            "timing_ms": timing_ms,
        }

    def _plan_query(
        self,
        *,
        user_query: str,
        top_k_override: Optional[int] = None,
    ) -> tuple[RetrievalQuery, float]:
        processor_started = time.perf_counter()
        plan = self.processor.process(user_query)
        if top_k_override is not None:
            plan = plan.model_copy(update={"top_k": int(top_k_override)})
        processor_time_ms = (time.perf_counter() - processor_started) * 1000.0
        return plan, processor_time_ms

    def _retrieve_prepared_evidence(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        attempt_start: int,
        phase: str,
    ) -> Dict[str, Any]:
        retriever_started = time.perf_counter()
        retrieval_bundle = self._retrieve_with_fallback(
            user_query=user_query,
            plan=plan,
            attempt_start=attempt_start,
            phase=phase,
        )
        plan_used = retrieval_bundle["plan_used"]
        evidence, diag, retrieved_result_count = self._prepare_answer_inputs(
            user_query=user_query,
            plan=plan_used,
            evidence=retrieval_bundle["results"],
            diag=retrieval_bundle["diagnostics"],
        )
        retriever_time_ms = (time.perf_counter() - retriever_started) * 1000.0
        return {
            "plan_used": plan_used,
            "attempts_log": list(retrieval_bundle["attempts_log"]),
            "attempt_used": retrieval_bundle["attempt_used"],
            "evidence": evidence,
            "diagnostics": diag,
            "retrieved_result_count": retrieved_result_count,
            "timing_ms": {
                "retriever": retriever_time_ms,
            },
        }

    def _generate_answer(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        evidence: List[EvidenceItem],
    ) -> tuple[AnswerResult, float]:
        answerer_started = time.perf_counter()
        draft_answer = self.answerer.answer(
            user_query=user_query,
            evidence_items=evidence,
            processor_plan=plan,
        )
        answerer_time_ms = (time.perf_counter() - answerer_started) * 1000.0
        return draft_answer, answerer_time_ms

    def _evaluate_answer(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        evidence: List[EvidenceItem],
        draft_answer: AnswerResult,
    ) -> tuple[EvaluationResult, float]:
        evaluator_started = time.perf_counter()
        evaluation = self.evaluator.evaluate(
            user_query=user_query,
            retrieval_query=plan,
            evidence=evidence,
            draft_answer=draft_answer,
        )
        evaluator_time_ms = (time.perf_counter() - evaluator_started) * 1000.0
        return evaluation, evaluator_time_ms

    def _run_answer_cycle(
        self,
        *,
        user_query: str,
        plan: RetrievalQuery,
        attempt_start: int,
        phase: str,
    ) -> Dict[str, Any]:
        retrieval = self._retrieve_prepared_evidence(
            user_query=user_query,
            plan=plan,
            attempt_start=attempt_start,
            phase=phase,
        )
        draft_answer, answerer_time_ms = self._generate_answer(
            user_query=user_query,
            plan=retrieval["plan_used"],
            evidence=retrieval["evidence"],
        )
        evaluation, evaluator_time_ms = self._evaluate_answer(
            user_query=user_query,
            plan=retrieval["plan_used"],
            evidence=retrieval["evidence"],
            draft_answer=draft_answer,
        )
        return {
            **retrieval,
            "draft_answer": draft_answer,
            "evaluation": evaluation,
            "timing_ms": {
                "retriever": retrieval["timing_ms"]["retriever"],
                "answerer": answerer_time_ms,
                "evaluator": evaluator_time_ms,
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
    def _finalize_runtime_answer(
        *,
        evaluation: EvaluationResult,
        draft_answer: AnswerResult,
        evidence: List[EvidenceItem],
        retries_used: int,
        max_retries: int,
        conservative_fallback: bool = False,
    ) -> tuple[AnswerResult, str, str]:
        effective_verdict = evaluation.verdict
        if effective_verdict == "revise" and retries_used >= max_retries:
            effective_verdict = "fallback"

        if conservative_fallback and effective_verdict != "ask_clarification":
            if not (evaluation.verdict == "pass" and evaluation.grounded and evaluation.relevant and evaluation.safe):
                effective_verdict = "fallback"

        runtime_action = QueryOrchestrator._decision_for_verdict(effective_verdict)
        final_answer = draft_answer
        if runtime_action == "ask_clarification":
            final_answer = QueryOrchestrator._build_clarification_answer(
                evaluation=evaluation,
                evidence=evidence,
            )
        elif runtime_action == "fallback":
            final_answer = QueryOrchestrator._build_safe_fallback_answer(evidence=evidence)
        return final_answer, effective_verdict, runtime_action

    @staticmethod
    def _coerce_feedback_draft_answer(last_answer: str) -> AnswerResult:
        return AnswerResult(
            answer=QueryOrchestrator._clean_text(last_answer),
            grounded=False,
            confidence=0.0,
            citations=[],
            used_evidence_count=0,
            fallback_used=False,
        )

    @staticmethod
    def _should_clarify_on_feedback(
        *,
        evaluation: EvaluationResult,
        reason: Optional[str],
    ) -> bool:
        if evaluation.verdict == "ask_clarification":
            return True
        return reason == "too_vague" and bool(evaluation.clarification_question)

    @staticmethod
    def _feedback_top_k(base_top_k: int, *, top_k_override: Optional[int] = None) -> int:
        broadened = min(12, int(base_top_k) + 2)
        if top_k_override is not None:
            return max(1, min(broadened, int(top_k_override)))
        return max(1, broadened)

    @staticmethod
    def _replace_plan_filters(
        *,
        plan: RetrievalQuery,
        filters: Optional[Dict[str, Any]],
    ) -> RetrievalQuery:
        payload = dict(filters or {})
        domains = QueryOrchestrator._dedupe_values(
            QueryOrchestrator._coerce_str_list(payload.get("domains"))
        )
        sections = QueryOrchestrator._dedupe_values(
            QueryOrchestrator._coerce_str_list(payload.get("sections"))
        )
        tags = QueryOrchestrator._dedupe_values(
            QueryOrchestrator._coerce_str_list(payload.get("entity_tags"))
        )
        update: Dict[str, Any] = {}
        if domains:
            update["domain"] = domains[0]
            update["domains"] = domains
        if sections:
            update["section"] = sections[0]
            update["sections"] = sections
        if tags:
            update["entity_tags"] = tags
        if "top_k" in payload:
            update["top_k"] = max(1, min(50, int(payload["top_k"])))
        return plan.model_copy(update=update) if update else plan

    @staticmethod
    def _build_feedback_retry_plan(
        *,
        base_plan: RetrievalQuery,
        current_plan: RetrievalQuery,
        evaluation: EvaluationResult,
        reason: Optional[str],
    ) -> RetrievalQuery:
        suggested_filters = evaluation.suggested_filters or {}
        if reason == "wrong_topic":
            retry_plan = QueryOrchestrator._replace_plan_filters(
                plan=base_plan,
                filters=suggested_filters,
            )
        else:
            retry_plan = QueryOrchestrator._apply_evaluator_suggested_filters(
                plan=current_plan,
                suggested_filters=suggested_filters,
            )

        if reason == "missing_detail":
            retry_plan = retry_plan.model_copy(
                update={"top_k": min(12, retry_plan.top_k + 2)}
            )
        elif reason == "incorrect":
            retry_plan = retry_plan.model_copy(
                update={"top_k": min(12, max(base_plan.top_k, retry_plan.top_k))}
            )

        return retry_plan

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
    def _safety_guard_check(user_query: str) -> tuple:
        """Return (triggered, reason_code) if the query matches a safety pattern."""
        normalised = (
            user_query.lower()
            .replace("’", "'")  # curly right apostrophe → straight
            .replace("‘", "'")  # curly left apostrophe → straight
        )
        for pattern, reason_code in QueryOrchestrator._SAFETY_PATTERNS:
            if pattern in normalised:
                return True, reason_code
        return False, ""

    @staticmethod
    def _build_safety_guard_answer(reason_code: str) -> AnswerResult:
        message = QueryOrchestrator._SAFETY_MESSAGES.get(
            reason_code,
            "I don't have enough verified information to answer that. "
            "Please check the official university website or ask a member of staff.",
        )
        return AnswerResult(
            answer=message,
            grounded=False,
            confidence=0.0,
            citations=[],
            used_evidence_count=0,
            fallback_used=True,
        )

    def _safety_guard_response(self, user_query: str, guard_reason: str) -> Dict[str, Any]:
        """Full response dict returned when the safety guard fires — same shape as run()."""
        started = time.perf_counter()
        answer = self._build_safety_guard_answer(guard_reason)
        total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return {
            "user_query": user_query,
            "mongo_status": self._get_mongo_status(),
            "index_health": self._get_index_health(),
            "processor_plan": RetrievalQuery(query_text=user_query).model_dump(),
            "answerer_run": answer.model_dump(),
            "evaluator_run": {
                "verdict": "fallback",
                "effective_verdict": "fallback",
                "grounded": False,
                "relevant": False,
                "clear": True,
                "safe": True,
                "issues": [guard_reason],
                "suggested_action": None,
                "suggested_filters": None,
                "clarification_question": None,
                "notes": f"safety_guard_triggered: {guard_reason}",
                "history": [],
            },
            "orchestration_decision": {
                "initial_verdict": "fallback",
                "final_verdict": "fallback",
                "effective_verdict": "fallback",
                "runtime_action": "fallback",
                "revise_retries_used": 0,
                "max_revise_retries": self.max_revise_retries,
                "safety_guard_triggered": True,
                "safety_guard_reason": guard_reason,
            },
            "retrieval_run": {
                "attempt_used": None,
                "attempts_log": [],
                "result_count": 0,
                "retrieved_result_count": 0,
                "diagnostics": {"safety_guard": guard_reason},
                "evidence": [],
            },
            "timing_ms": {
                "processor": 0.0,
                "retriever": 0.0,
                "answerer": 0.0,
                "evaluator": 0.0,
                "total": total_ms,
            },
        }

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

        try:
            expanded_docs = list(
                self.repo.collection.find(
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
            )
        except Exception:
            expanded_docs = []

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
        has_accommodation_subject = any(
            token in query_lc
            for token in ("accommodation", "hall", "halls", "room", "rooms", "on-campus", "on campus")
        )
        return is_accommodation and has_extreme and (has_price_signal or has_accommodation_subject)

    def _get_mongo_status(self) -> Dict[str, Any]:
        try:
            doc_count = self.repo.collection.count_documents({})
            return {
                "database": self._mongo_db,
                "collection": self._mongo_collection,
                "total_chunk_docs": doc_count,
            }
        except Exception as exc:
            return {
                "database": self._mongo_db,
                "collection": self._mongo_collection,
                "total_chunk_docs": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _get_index_health(self) -> Dict[str, Any]:
        try:
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
        except Exception as exc:
            return {
                "healthy": False,
                "error": f"{type(exc).__name__}: {exc}",
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
