from __future__ import annotations

from types import MethodType

from app.orchestrator import QueryOrchestrator
from schemas.models import AnswerResult, EvaluationResult, EvidenceItem, RetrievalQuery


def test_query_orchestrator_expands_accommodation_price_comparisons() -> None:
    plan = RetrievalQuery(
        query_text="Which on-campus accommodation has the lowest weekly price?",
        top_k=8,
        domain="accommodation",
        domains=["accommodation"],
    )

    assert QueryOrchestrator._should_expand_accommodation_price_evidence(
        "Which on-campus accommodation has the lowest weekly price?",
        plan,
    )


def test_query_orchestrator_expands_cheapest_accommodation_without_explicit_price_words() -> None:
    plan = RetrievalQuery(
        query_text="What is the cheapest accommodation?",
        top_k=8,
        domain="accommodation",
        domains=["accommodation"],
    )

    assert QueryOrchestrator._should_expand_accommodation_price_evidence(
        "What is the cheapest accommodation?",
        plan,
    )


def test_query_orchestrator_does_not_expand_simple_price_lookup() -> None:
    plan = RetrievalQuery(
        query_text="How much does Butler Court cost per week?",
        top_k=8,
        domain="accommodation",
        domains=["accommodation"],
    )

    assert not QueryOrchestrator._should_expand_accommodation_price_evidence(
        "How much does Butler Court cost per week?",
        plan,
    )


def test_query_orchestrator_decision_mapping_for_verdicts() -> None:
    assert QueryOrchestrator._decision_for_verdict("pass") == "pass"
    assert QueryOrchestrator._decision_for_verdict("ask_clarification") == "ask_clarification"
    assert QueryOrchestrator._decision_for_verdict("fallback") == "fallback"
    assert QueryOrchestrator._decision_for_verdict("revise") == "fallback"


def test_query_orchestrator_revise_retry_budget_is_capped() -> None:
    assert QueryOrchestrator._should_retry_after_evaluation(
        verdict="revise",
        retries_used=0,
        max_retries=1,
    )
    assert not QueryOrchestrator._should_retry_after_evaluation(
        verdict="revise",
        retries_used=1,
        max_retries=1,
    )
    assert not QueryOrchestrator._should_retry_after_evaluation(
        verdict="pass",
        retries_used=0,
        max_retries=1,
    )


def test_query_orchestrator_merges_suggested_filters_into_plan() -> None:
    base = RetrievalQuery(
        query_text="How much does Butler Court cost?",
        top_k=6,
        domain="accommodation",
        domains=["accommodation"],
        section="overview",
        sections=["overview"],
        entity_tags=["Butler Court"],
    )

    revised = QueryOrchestrator._apply_evaluator_suggested_filters(
        plan=base,
        suggested_filters={
            "top_k": 10,
            "domains": ["finance"],
            "sections": ["prices"],
            "entity_tags": ["Weekly Cost"],
        },
    )

    assert revised.top_k == 10
    assert "accommodation" in revised.domains
    assert "finance" in revised.domains
    assert "overview" in revised.sections
    assert "prices" in revised.sections
    assert "Butler Court" in revised.entity_tags
    assert "Weekly Cost" in revised.entity_tags


def test_query_orchestrator_run_includes_timing_ms() -> None:
    orchestrator = QueryOrchestrator.__new__(QueryOrchestrator)
    orchestrator._mongo_db = "open_day_knowledge"
    orchestrator._mongo_collection = "kb_chuncks"
    orchestrator.max_revise_retries = 1

    class _StubProcessor:
        def process(self, user_query: str) -> RetrievalQuery:
            return RetrievalQuery(
                query_text=user_query,
                top_k=4,
                domain="accommodation",
                domains=["accommodation"],
                entity_tags=["Butler Court"],
            )

    class _StubAnswerer:
        def answer(self, *, user_query: str, evidence_items, processor_plan: RetrievalQuery) -> AnswerResult:
            _ = (user_query, evidence_items, processor_plan)
            return AnswerResult(
                answer="Butler Court costs GBP 126.68 per week.",
                grounded=True,
                confidence=0.95,
                citations=[],
                used_evidence_count=0,
                fallback_used=False,
            )

    class _StubEvaluator:
        def evaluate(self, *, user_query: str, retrieval_query: RetrievalQuery, evidence, draft_answer: AnswerResult) -> EvaluationResult:
            _ = (user_query, retrieval_query, evidence, draft_answer)
            return EvaluationResult(
                verdict="pass",
                grounded=True,
                relevant=True,
                clear=True,
                safe=True,
                issues=[],
                suggested_action="accept_answer",
                suggested_filters=None,
                clarification_question=None,
                notes="ok",
            )

    evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            url="https://www.lboro.ac.uk/services/accommodation/our-halls/butler-court/",
            text="Butler Court standard rooms cost GBP 126.68 per week.",
            domain="accommodation",
            entity_tags=["Butler Court"],
            section="prices",
            order=0,
            version="ingest-v3",
            score=0.91,
            retrieval_channels=["vector", "text"],
            metadata={},
        )
    ]

    orchestrator.processor = _StubProcessor()
    orchestrator.answerer = _StubAnswerer()
    orchestrator.evaluator = _StubEvaluator()

    def _get_mongo_status(self):
        return {"database": self._mongo_db, "collection": self._mongo_collection, "total_chunk_docs": 1}

    def _get_index_health(self):
        return {
            "healthy": True,
            "vector_index_found": True,
            "vector_index_status": "ready",
            "vector_index_dimensions": 64,
            "text_index_found": True,
            "text_index_status": "ready",
            "errors": [],
        }

    def _retrieve_with_fallback(self, *, user_query: str, plan: RetrievalQuery, attempt_start: int, phase: str):
        _ = (self, user_query, attempt_start, phase)
        return {
            "plan_used": plan,
            "results": evidence,
            "diagnostics": {"vector_mode": "fallback_cosine"},
            "attempts_log": [{"attempt": 1, "description": "initial:full_plan", "hits": 1}],
            "attempt_used": 1,
        }

    def _prepare_answer_inputs(self, *, user_query: str, plan: RetrievalQuery, evidence, diag):
        _ = (self, user_query, plan, evidence)
        return list(evidence), dict(diag or {}), len(evidence)

    orchestrator._get_mongo_status = MethodType(_get_mongo_status, orchestrator)
    orchestrator._get_index_health = MethodType(_get_index_health, orchestrator)
    orchestrator._retrieve_with_fallback = MethodType(_retrieve_with_fallback, orchestrator)
    orchestrator._prepare_answer_inputs = MethodType(_prepare_answer_inputs, orchestrator)

    payload = orchestrator.run("How much is Butler Court?")

    assert "timing_ms" in payload
    timing = payload["timing_ms"]
    for key in ("processor", "retriever", "answerer", "evaluator", "total"):
        assert key in timing
        assert isinstance(timing[key], (int, float))
        assert timing[key] >= 0
