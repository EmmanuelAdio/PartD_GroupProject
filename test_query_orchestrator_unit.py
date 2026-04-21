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


def test_query_orchestrator_feedback_top_k_broadens_modestly() -> None:
    assert QueryOrchestrator._feedback_top_k(8) == 10
    assert QueryOrchestrator._feedback_top_k(11) == 12
    assert QueryOrchestrator._feedback_top_k(8, top_k_override=9) == 9


def test_query_orchestrator_feedback_wrong_topic_replaces_conflicting_filters() -> None:
    base = RetrievalQuery(
        query_text="Tell me about accommodation fees",
        top_k=10,
        domain="accommodation",
        domains=["accommodation"],
        section="overview",
        sections=["overview"],
    )

    revised = QueryOrchestrator._build_feedback_retry_plan(
        base_plan=base,
        current_plan=base,
        evaluation=EvaluationResult(
            verdict="revise",
            grounded=False,
            relevant=False,
            clear=True,
            safe=True,
            issues=["wrong_topic"],
            suggested_action="retry_with_adjusted_filters",
            suggested_filters={
                "domains": ["finance"],
                "sections": ["fees"],
                "entity_tags": ["Scholarships"],
                "top_k": 12,
            },
            clarification_question=None,
            notes="off-topic",
        ),
        reason="wrong_topic",
    )

    assert revised.domains == ["finance"]
    assert revised.domain == "finance"
    assert revised.sections == ["fees"]
    assert revised.section == "fees"
    assert revised.entity_tags == ["Scholarships"]
    assert revised.top_k == 12


def test_query_orchestrator_feedback_missing_detail_broadens_retry_plan() -> None:
    base = RetrievalQuery(
        query_text="Tell me about accommodation fees",
        top_k=8,
        domain="accommodation",
        domains=["accommodation"],
    )

    revised = QueryOrchestrator._build_feedback_retry_plan(
        base_plan=base,
        current_plan=base,
        evaluation=EvaluationResult(
            verdict="revise",
            grounded=True,
            relevant=True,
            clear=False,
            safe=True,
            issues=["answer_too_vague_or_short"],
            suggested_action="retry_with_adjusted_filters",
            suggested_filters=None,
            clarification_question=None,
            notes="needs more detail",
        ),
        reason="missing_detail",
    )

    assert revised.top_k == 12


def test_query_orchestrator_feedback_short_circuits_to_clarification() -> None:
    orchestrator = QueryOrchestrator.__new__(QueryOrchestrator)
    orchestrator._mongo_db = "open_day_knowledge"
    orchestrator._mongo_collection = "kb_chuncks"
    orchestrator.max_feedback_retries = 1

    evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            url="https://www.lboro.ac.uk/services/accommodation/our-halls/butler-court/",
            text="Butler Court costs GBP 126.68 per week.",
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

    class _StubProcessor:
        def process(self, user_query: str) -> RetrievalQuery:
            return RetrievalQuery(query_text=user_query, top_k=6)

    class _StubAnswerer:
        def __init__(self) -> None:
            self.calls = 0

        def answer(self, *, user_query: str, evidence_items, processor_plan: RetrievalQuery) -> AnswerResult:
            self.calls += 1
            _ = (user_query, evidence_items, processor_plan)
            return AnswerResult(
                answer="retry answer",
                grounded=True,
                confidence=0.5,
                citations=[],
                used_evidence_count=0,
                fallback_used=False,
            )

    class _StubEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, *, user_query: str, retrieval_query: RetrievalQuery, evidence, draft_answer: AnswerResult) -> EvaluationResult:
            self.calls += 1
            _ = (user_query, retrieval_query, evidence, draft_answer)
            return EvaluationResult(
                verdict="ask_clarification",
                grounded=True,
                relevant=False,
                clear=False,
                safe=True,
                issues=["ambiguous_user_query"],
                suggested_action="ask_user_clarification",
                suggested_filters=None,
                clarification_question="Which accommodation hall do you mean?",
                notes="ambiguous",
            )

    retrieval_calls = []

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
        _ = (self, user_query, attempt_start)
        retrieval_calls.append((phase, plan.top_k))
        return {
            "plan_used": plan,
            "results": evidence,
            "diagnostics": {"phase": phase},
            "attempts_log": [{"attempt": attempt_start, "description": f"{phase}:full_plan", "hits": 1}],
            "attempt_used": attempt_start,
        }

    def _prepare_answer_inputs(self, *, user_query: str, plan: RetrievalQuery, evidence, diag):
        _ = (self, user_query, plan)
        return list(evidence), dict(diag or {}), len(evidence)

    orchestrator.processor = _StubProcessor()
    orchestrator.answerer = _StubAnswerer()
    orchestrator.evaluator = _StubEvaluator()
    orchestrator._get_mongo_status = MethodType(_get_mongo_status, orchestrator)
    orchestrator._get_index_health = MethodType(_get_index_health, orchestrator)
    orchestrator._retrieve_with_fallback = MethodType(_retrieve_with_fallback, orchestrator)
    orchestrator._prepare_answer_inputs = MethodType(_prepare_answer_inputs, orchestrator)

    payload = orchestrator.run_with_feedback(
        user_query="How much is it?",
        last_answer="It costs GBP 126.68 per week.",
        reason="too_vague",
    )

    assert orchestrator.answerer.calls == 0
    assert orchestrator.evaluator.calls == 1
    assert retrieval_calls == [("feedback_initial", 8)]
    assert payload["orchestration_decision"]["runtime_action"] == "ask_clarification"
    assert "clarification" in payload["answerer_run"]["answer"].lower()


def test_query_orchestrator_feedback_uses_single_retry_and_adjusted_filters() -> None:
    orchestrator = QueryOrchestrator.__new__(QueryOrchestrator)
    orchestrator._mongo_db = "open_day_knowledge"
    orchestrator._mongo_collection = "kb_chuncks"
    orchestrator.max_feedback_retries = 1

    evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            url="https://www.lboro.ac.uk/services/accommodation/our-halls/butler-court/",
            text="Butler Court costs GBP 126.68 per week.",
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

    class _StubProcessor:
        def process(self, user_query: str) -> RetrievalQuery:
            return RetrievalQuery(
                query_text=user_query,
                top_k=6,
                domain="accommodation",
                domains=["accommodation"],
            )

    class _StubAnswerer:
        def __init__(self) -> None:
            self.calls = 0

        def answer(self, *, user_query: str, evidence_items, processor_plan: RetrievalQuery) -> AnswerResult:
            self.calls += 1
            _ = (user_query, evidence_items, processor_plan)
            return AnswerResult(
                answer="Butler Court costs GBP 126.68 per week.",
                grounded=True,
                confidence=0.92,
                citations=[],
                used_evidence_count=1,
                fallback_used=False,
            )

    class _StubEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, *, user_query: str, retrieval_query: RetrievalQuery, evidence, draft_answer: AnswerResult) -> EvaluationResult:
            self.calls += 1
            _ = (user_query, retrieval_query, evidence, draft_answer)
            if self.calls == 1:
                return EvaluationResult(
                    verdict="revise",
                    grounded=False,
                    relevant=False,
                    clear=False,
                    safe=True,
                    issues=["wrong_topic"],
                    suggested_action="retry_with_adjusted_filters",
                    suggested_filters={
                        "domains": ["finance"],
                        "sections": ["fees"],
                        "top_k": 11,
                    },
                    clarification_question=None,
                    notes="use fees evidence",
                )
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

    retrieval_plans = []

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
        _ = (self, user_query, attempt_start)
        retrieval_plans.append((phase, plan.top_k, list(plan.domains), list(plan.sections)))
        return {
            "plan_used": plan,
            "results": evidence,
            "diagnostics": {"phase": phase},
            "attempts_log": [{"attempt": attempt_start, "description": f"{phase}:full_plan", "hits": 1}],
            "attempt_used": attempt_start,
        }

    def _prepare_answer_inputs(self, *, user_query: str, plan: RetrievalQuery, evidence, diag):
        _ = (self, user_query, plan)
        return list(evidence), dict(diag or {}), len(evidence)

    orchestrator.processor = _StubProcessor()
    orchestrator.answerer = _StubAnswerer()
    orchestrator.evaluator = _StubEvaluator()
    orchestrator._get_mongo_status = MethodType(_get_mongo_status, orchestrator)
    orchestrator._get_index_health = MethodType(_get_index_health, orchestrator)
    orchestrator._retrieve_with_fallback = MethodType(_retrieve_with_fallback, orchestrator)
    orchestrator._prepare_answer_inputs = MethodType(_prepare_answer_inputs, orchestrator)

    payload = orchestrator.run_with_feedback(
        user_query="Tell me about accommodation fees",
        last_answer="Accommodation costs vary.",
        reason="wrong_topic",
    )

    assert orchestrator.answerer.calls == 1
    assert orchestrator.evaluator.calls == 2
    assert retrieval_plans == [
        ("feedback_initial", 8, ["accommodation"], []),
        ("feedback_retry", 11, ["finance"], ["fees"]),
    ]
    assert payload["orchestration_decision"]["feedback_retry_used"] == 1
    assert payload["orchestration_decision"]["runtime_action"] == "pass"
    assert payload["answerer_run"]["answer"] == "Butler Court costs GBP 126.68 per week."


def test_query_orchestrator_feedback_incorrect_uses_conservative_fallback() -> None:
    orchestrator = QueryOrchestrator.__new__(QueryOrchestrator)
    orchestrator._mongo_db = "open_day_knowledge"
    orchestrator._mongo_collection = "kb_chuncks"
    orchestrator.max_feedback_retries = 1

    evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            source_id="all_ug_courses",
            source_type="json",
            title="Computer Science Course",
            url="https://www.lboro.ac.uk/study/undergraduate/courses/computer-science/",
            text="Computer Science entry requirements: AAB including Mathematics.",
            domain="courses",
            entity_tags=["Computer Science"],
            section="entry_requirements",
            order=0,
            version="ingest-v3",
            score=0.88,
            retrieval_channels=["vector", "text"],
            metadata={},
        )
    ]

    class _StubProcessor:
        def process(self, user_query: str) -> RetrievalQuery:
            return RetrievalQuery(query_text=user_query, top_k=5, domain="courses", domains=["courses"])

    class _StubAnswerer:
        def answer(self, *, user_query: str, evidence_items, processor_plan: RetrievalQuery) -> AnswerResult:
            _ = (user_query, evidence_items, processor_plan)
            return AnswerResult(
                answer="The requirement is ABB.",
                grounded=True,
                confidence=0.9,
                citations=[],
                used_evidence_count=0,
                fallback_used=False,
            )

    class _StubEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, *, user_query: str, retrieval_query: RetrievalQuery, evidence, draft_answer: AnswerResult) -> EvaluationResult:
            self.calls += 1
            _ = (user_query, retrieval_query, evidence, draft_answer)
            if self.calls == 1:
                return EvaluationResult(
                    verdict="revise",
                    grounded=False,
                    relevant=True,
                    clear=True,
                    safe=False,
                    issues=["unsupported_numeric_claims"],
                    suggested_action="retry_with_adjusted_filters",
                    suggested_filters={"sections": ["entry_requirements"], "top_k": 7},
                    clarification_question=None,
                    notes="recheck grounding",
                )
            return EvaluationResult(
                verdict="revise",
                grounded=False,
                relevant=True,
                clear=True,
                safe=False,
                issues=["unsupported_numeric_claims"],
                suggested_action="use_safe_fallback",
                suggested_filters=None,
                clarification_question=None,
                notes="still unsafe",
            )

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
            "diagnostics": {"phase": phase},
            "attempts_log": [{"attempt": attempt_start, "description": f"{phase}:full_plan", "hits": 1}],
            "attempt_used": attempt_start,
        }

    def _prepare_answer_inputs(self, *, user_query: str, plan: RetrievalQuery, evidence, diag):
        _ = (self, user_query, plan)
        return list(evidence), dict(diag or {}), len(evidence)

    orchestrator.processor = _StubProcessor()
    orchestrator.answerer = _StubAnswerer()
    orchestrator.evaluator = _StubEvaluator()
    orchestrator._get_mongo_status = MethodType(_get_mongo_status, orchestrator)
    orchestrator._get_index_health = MethodType(_get_index_health, orchestrator)
    orchestrator._retrieve_with_fallback = MethodType(_retrieve_with_fallback, orchestrator)
    orchestrator._prepare_answer_inputs = MethodType(_prepare_answer_inputs, orchestrator)

    payload = orchestrator.run_with_feedback(
        user_query="What are the entry requirements for Computer Science?",
        last_answer="The requirement is ABB.",
        reason="incorrect",
    )

    assert payload["orchestration_decision"]["feedback_retry_used"] == 1
    assert payload["orchestration_decision"]["runtime_action"] == "fallback"
    assert payload["answerer_run"]["fallback_used"] is True
