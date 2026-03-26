from __future__ import annotations

from app.orchestrator import QueryOrchestrator
from schemas.models import RetrievalQuery


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
