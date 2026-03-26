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
