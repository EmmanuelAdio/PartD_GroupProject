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
            url="https://www.lboro.ac.uk/services/accommodation/halls/",
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


def _sample_price_comparison_evidence() -> List[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id="chunk-butler",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text=(
                "[0].name: Butler Court\n"
                "[0].room_types[0].name: Standard\n"
                "[0].room_types[0].prices[0].year: 2025/26\n"
                "[0].room_types[0].prices[0].per_week_gbp: 126.68\n"
                "[0].room_types[0].prices[0].total_contract_gbp: 5302.27"
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
            chunk_id="chunk-holt",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text=(
                "[14].name: The Holt\n"
                "[14].room_types[0].name: One bedroom, 4ft bed, in 2 bedroom bungalow\n"
                "[14].room_types[0].prices[0].year: 2025/26\n"
                "[14].room_types[0].prices[0].per_week_gbp: 165.41\n"
                "[14].room_types[0].prices[0].total_contract_gbp: 6900.14"
            ),
            domain="accommodation",
            entity_tags=["The Holt"],
            section="json_fields",
            order=1,
            version="ingest-v2",
            score=0.82,
            retrieval_channels=["vector", "text"],
            metadata={},
        ),
    ]


def _sample_split_price_extreme_evidence() -> List[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id="chunk-faraday-name",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text="[6].name: Faraday",
            domain="accommodation",
            entity_tags=["Faraday"],
            section="json_fields",
            order=0,
            version="ingest-v2",
            score=0.0,
            retrieval_channels=[],
            metadata={},
        ),
        EvidenceItem(
            chunk_id="chunk-faraday-price",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text=(
                "[6].room_types[4].name: En-suite, 4ft bed\n"
                "[6].room_types[4].prices[0].year: 2025/26\n"
                "[6].room_types[4].prices[0].per_week_gbp: 239.53\n"
                "[6].room_types[4].prices[0].total_contract_gbp: 10000.00"
            ),
            domain="accommodation",
            entity_tags=["En-suite, 4ft bed"],
            section="json_fields",
            order=1,
            version="ingest-v2",
            score=0.0,
            retrieval_channels=[],
            metadata={},
        ),
        EvidenceItem(
            chunk_id="chunk-royce-name",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text="[11].name: Royce",
            domain="accommodation",
            entity_tags=["Royce"],
            section="json_fields",
            order=2,
            version="ingest-v2",
            score=0.0,
            retrieval_channels=[],
            metadata={},
        ),
        EvidenceItem(
            chunk_id="chunk-royce-price",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text=(
                "[11].room_types[3].name: En-suite, 4ft bed\n"
                "[11].room_types[3].prices[0].year: 2025/26\n"
                "[11].room_types[3].prices[0].per_week_gbp: 239.53\n"
                "[11].room_types[3].prices[0].total_contract_gbp: 10000.00"
            ),
            domain="accommodation",
            entity_tags=["En-suite, 4ft bed"],
            section="json_fields",
            order=3,
            version="ingest-v2",
            score=0.0,
            retrieval_channels=[],
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
    assert result.answer.startswith("I do not know the answer.")
    assert "https://www.lboro.ac.uk/" in result.answer


def test_unknown_answer_prefers_evidence_url_when_available() -> None:
    agent = AnswererAgent(llm_service=None)
    result = agent._unknown_answer_result(  # pylint: disable=protected-access
        evidence=_sample_evidence(),
        fallback_used=True,
    )

    assert result.grounded is False
    assert result.answer.startswith("I do not know the answer.")
    assert "https://www.lboro.ac.uk/services/accommodation/halls/" in result.answer
    assert result.used_evidence_count == 1
    assert len(result.citations) == 1
    assert result.citations[0].url == "https://www.lboro.ac.uk/services/accommodation/halls/"


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


def test_answerer_fallback_returns_unknown_for_low_relevance_personal_question() -> None:
    agent = AnswererAgent(llm_service=None)

    result = agent.answer(
        user_query="What is my middle name?",
        evidence_items=_sample_evidence(),
    )

    assert result.fallback_used is True
    assert result.grounded is False
    assert result.answer.startswith("I do not know the answer.")
    assert "Based on the retrieved information" not in result.answer
    assert "https://www.lboro.ac.uk/services/accommodation/halls/" in result.answer


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


def test_answerer_fallback_handles_cheapest_price_questions() -> None:
    agent = AnswererAgent(llm_service=None)

    result = agent.answer(
        user_query="Which on-campus accommodation has the lowest weekly price?",
        evidence_items=_sample_price_comparison_evidence(),
    )

    assert result.fallback_used is True
    assert result.grounded is True
    assert "lowest weekly price" in result.answer.lower()
    assert "Butler Court" in result.answer
    assert "GBP 126.68" in result.answer
    assert result.used_evidence_count == 1


def test_answerer_uses_deterministic_extreme_price_logic_even_with_llm_available() -> None:
    agent = AnswererAgent(
        llm_service=_StubLLMService(
            {
                "answer": "The cheapest accommodation is The Holt at GBP 165.41 per week.",
                "grounded": True,
                "confidence": 1.0,
                "citation_ids": [2],
            }
        )
    )

    result = agent.answer(
        user_query="Which on-campus accommodation has the lowest weekly price?",
        evidence_items=_sample_price_comparison_evidence(),
    )

    assert result.fallback_used is True
    assert result.grounded is True
    assert "Butler Court" in result.answer
    assert "GBP 126.68" in result.answer


def test_answerer_extreme_price_logic_recovers_names_across_split_chunks_and_handles_ties() -> None:
    agent = AnswererAgent(llm_service=None)

    result = agent.answer(
        user_query="Which on-campus accommodation has the highest weekly price?",
        evidence_items=_sample_split_price_extreme_evidence(),
    )

    assert result.fallback_used is True
    assert result.grounded is True
    assert "GBP 239.53" in result.answer
    assert "Faraday" in result.answer
    assert "Royce" in result.answer
    assert result.used_evidence_count == 2
