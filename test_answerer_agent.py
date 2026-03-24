from __future__ import annotations

from typing import Any, Dict, List

from agents.answerer_agent import AnswererAgent
from schemas.models import EvidenceItem, RetrievalQuery


class _StubLLMService:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def generate_json(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        return dict(self.payload)


def _sample_evidence() -> List[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id="chunk-1",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text=(
                "room_types[0].prices[0].per_week_gbp: 126.68\n"
                "room_types[0].prices[0].total_contract_gbp: 5302.27\n"
                "name: Butler Court"
            ),
            domain="accommodation",
            entity_tags=["Butler Court"],
            section="json_fields",
            order=0,
            version="ingest-v2",
            score=0.91,
            retrieval_channels=["vector", "text"],
            metadata={},
        ),
        EvidenceItem(
            chunk_id="chunk-2",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text="address: Butler Court Loughborough University LE11 3TS",
            domain="accommodation",
            entity_tags=["Butler Court"],
            section="json_fields",
            order=1,
            version="ingest-v2",
            score=0.74,
            retrieval_channels=["text"],
            metadata={},
        ),
    ]


def test_answerer_returns_no_evidence_response_when_empty() -> None:
    agent = AnswererAgent(llm_service=None)

    result = agent.answer(
        user_query="How much is Butler Court accommodation?",
        evidence_items=[],
    )

    assert result.grounded is False
    assert result.used_evidence_count == 0
    assert result.citations == []
    assert "couldn't find enough relevant information" in result.answer.lower()


def test_answerer_fallback_extracts_relevant_lines() -> None:
    agent = AnswererAgent(llm_service=None)

    result = agent.answer(
        user_query="How much is Butler Court accommodation per week?",
        evidence_items=_sample_evidence(),
        processor_plan=RetrievalQuery(
            query_text="Butler Court accommodation price per week",
            top_k=5,
            domains=["accommodation"],
            entity_tags=["Butler Court"],
        ),
    )

    assert result.fallback_used is True
    assert result.grounded is True
    assert result.used_evidence_count >= 1
    assert any(citation.chunk_id == "chunk-1" for citation in result.citations)
    assert "GBP 126.68 per week" in result.answer


def test_answerer_llm_output_maps_valid_citations_and_ignores_invalid_ids() -> None:
    agent = AnswererAgent(
        llm_service=_StubLLMService(
            {
                "answer": "Butler Court costs £126.68 per week based on the retrieved accommodation pricing data.",
                "grounded": True,
                "confidence": 0.88,
                "citation_ids": [1, 1, 99],
            }
        )
    )

    result = agent.answer(
        user_query="How much is Butler Court accommodation per week?",
        evidence_items=_sample_evidence(),
    )

    assert result.fallback_used is False
    assert result.grounded is True
    assert result.confidence == 0.88
    assert result.used_evidence_count == 1
    assert len(result.citations) == 1
    assert result.citations[0].evidence_id == 1
    assert result.citations[0].chunk_id == "chunk-1"


def test_answerer_prefers_grounded_fallback_when_llm_misses_obvious_price_evidence() -> None:
    agent = AnswererAgent(
        llm_service=_StubLLMService(
            {
                "answer": "The evidence does not provide specific pricing information for Butler Court accommodation.",
                "grounded": False,
                "confidence": 0.0,
                "citation_ids": [],
            }
        )
    )

    result = agent.answer(
        user_query="How much is Butler Court accommodation per week?",
        evidence_items=_sample_evidence(),
    )

    assert result.fallback_used is True
    assert result.grounded is True
    assert result.used_evidence_count == 1
    assert "126.68" in result.answer
    assert "Butler Court" in result.answer


def test_answerer_polishes_llm_price_formatting() -> None:
    agent = AnswererAgent(
        llm_service=_StubLLMService(
            {
                "answer": "The accommodation at Butler Court costs £126.68 per week, totaling £5302.27 for the contract period.",
                "grounded": True,
                "confidence": 1.0,
                "citation_ids": [1],
            }
        )
    )

    result = agent.answer(
        user_query="How much is Butler Court accommodation?",
        evidence_items=_sample_evidence(),
    )

    assert "GBP 126.68 per week" in result.answer
    assert "GBP 5,302.27" in result.answer
    assert "total contract cost" in result.answer.lower()
