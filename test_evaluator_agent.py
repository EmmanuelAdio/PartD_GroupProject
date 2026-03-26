from __future__ import annotations

from typing import Any, Dict, List

from agents.evaluator_agent import EvaluatorAgent
from schemas.models import AnswerCitation, AnswerResult, EvidenceItem, RetrievalQuery


class _TrackingLLMService:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def generate_json(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        self.calls += 1
        return dict(self.payload)


def _strong_evidence() -> List[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id="chunk-1",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            url="https://www.lboro.ac.uk/services/accommodation/halls/",
            text="name: Butler Court\nper_week_gbp: 126.68\ntotal_contract_gbp: 5302.27",
            domain="accommodation",
            entity_tags=["Butler Court"],
            section="prices",
            order=0,
            version="ingest-v2",
            score=0.92,
            vector_score=0.88,
            text_score=0.81,
            retrieval_channels=["vector", "text"],
            metadata={},
        ),
        EvidenceItem(
            chunk_id="chunk-2",
            source_id="accommodation_halls",
            source_type="json",
            title="Accommodation Halls",
            text="Butler Court is on campus and offers catered and self-catered options.",
            domain="accommodation",
            entity_tags=["Butler Court"],
            section="overview",
            order=1,
            version="ingest-v2",
            score=0.74,
            retrieval_channels=["text"],
            metadata={},
        ),
    ]


def _weak_evidence() -> List[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id="chunk-weak",
            source_id="general_source",
            source_type="txt",
            title="General Notes",
            text="Some accommodation information may exist.",
            domain="general",
            entity_tags=[],
            section="text",
            order=0,
            version="ingest-v2",
            score=0.10,
            retrieval_channels=["text"],
            metadata={},
        )
    ]


def test_evaluator_no_evidence_returns_fallback() -> None:
    agent = EvaluatorAgent(llm_service=None)
    result = agent.evaluate(
        user_query="How much is Butler Court accommodation?",
        retrieval_query=None,
        evidence=[],
        draft_answer=AnswerResult(
            answer="Butler Court is GBP 126.68 per week.",
            grounded=True,
            confidence=0.95,
            citations=[],
            used_evidence_count=0,
            fallback_used=False,
        ),
    )

    assert result.verdict == "fallback"
    assert result.grounded is False
    assert "no_evidence_returned" in result.issues


def test_evaluator_flags_overconfident_answer_on_weak_evidence() -> None:
    agent = EvaluatorAgent(llm_service=None)
    result = agent.evaluate(
        user_query="What are accommodation fees?",
        retrieval_query=RetrievalQuery(query_text="accommodation fees", top_k=6),
        evidence=_weak_evidence(),
        draft_answer=AnswerResult(
            answer="The fee is definitely GBP 9999 per week.",
            grounded=True,
            confidence=0.99,
            citations=[],
            used_evidence_count=0,
            fallback_used=False,
        ),
    )

    assert result.safe is False
    assert result.verdict in {"fallback", "revise"}


def test_evaluator_passes_strong_grounded_answer() -> None:
    agent = EvaluatorAgent(llm_service=None)
    evidence = _strong_evidence()
    result = agent.evaluate(
        user_query="How much does Butler Court cost per week?",
        retrieval_query=RetrievalQuery(
            query_text="Butler Court weekly price",
            top_k=6,
            domain="accommodation",
            domains=["accommodation"],
            entity_tags=["Butler Court"],
            section="prices",
            sections=["prices"],
        ),
        evidence=evidence,
        draft_answer=AnswerResult(
            answer="Butler Court is listed at GBP 126.68 per week.",
            grounded=True,
            confidence=0.88,
            citations=[
                AnswerCitation(
                    evidence_id=1,
                    chunk_id=evidence[0].chunk_id,
                    source_id=evidence[0].source_id,
                    source_type=evidence[0].source_type,
                    title=evidence[0].title,
                    section=evidence[0].section,
                    url=evidence[0].url,
                )
            ],
            used_evidence_count=1,
            fallback_used=False,
        ),
    )

    assert result.verdict == "pass"
    assert result.grounded is True
    assert result.relevant is True
    assert result.clear is True
    assert result.safe is True


def test_evaluator_marks_off_topic_answer_for_revision() -> None:
    agent = EvaluatorAgent(llm_service=None)
    evidence = [
        EvidenceItem(
            chunk_id="chunk-cs",
            source_id="all_ug_courses",
            source_type="json",
            title="Computer Science Course",
            text="Computer Science entry requirements: AAB including Mathematics.",
            domain="courses",
            entity_tags=["Computer Science"],
            section="entry_requirements",
            order=0,
            version="ingest-v2",
            score=0.88,
            retrieval_channels=["vector", "text"],
            metadata={},
        ),
        EvidenceItem(
            chunk_id="chunk-cs-2",
            source_id="all_ug_courses",
            source_type="json",
            title="Computer Science Course",
            text="Typical offer: AAB and GCSE English requirements.",
            domain="courses",
            entity_tags=["Computer Science"],
            section="entry_requirements",
            order=1,
            version="ingest-v2",
            score=0.74,
            retrieval_channels=["text"],
            metadata={},
        ),
    ]

    result = agent.evaluate(
        user_query="What are the entry requirements for Computer Science?",
        retrieval_query=RetrievalQuery(
            query_text="Computer Science entry requirements",
            top_k=6,
            domain="courses",
            domains=["courses"],
            entity_tags=["Computer Science"],
            section="entry_requirements",
            sections=["entry_requirements"],
        ),
        evidence=evidence,
        draft_answer=AnswerResult(
            answer="Butler Court is on campus and has multiple room options.",
            grounded=True,
            confidence=0.8,
            citations=[],
            used_evidence_count=0,
            fallback_used=False,
        ),
    )

    assert result.relevant is False
    assert result.verdict in {"revise", "ask_clarification"}


def test_evaluator_asks_clarification_for_ambiguous_query() -> None:
    agent = EvaluatorAgent(llm_service=None)
    result = agent.evaluate(
        user_query="How much is it?",
        retrieval_query=RetrievalQuery(query_text="How much is it?", top_k=6),
        evidence=_strong_evidence(),
        draft_answer=AnswerResult(
            answer="It costs GBP 126.68 per week.",
            grounded=True,
            confidence=0.9,
            citations=[],
            used_evidence_count=0,
            fallback_used=False,
        ),
    )

    assert result.verdict == "ask_clarification"
    assert bool(result.clarification_question)


def test_evaluator_calls_llm_judge_only_when_needed() -> None:
    llm = _TrackingLLMService(
        payload={
            "grounded": True,
            "relevant": True,
            "clear": True,
            "safe": True,
            "issues": [],
            "suggested_verdict": "revise",
            "notes": "Borderline retrieval confidence.",
        }
    )
    agent = EvaluatorAgent(llm_service=llm)

    _ = agent.evaluate(
        user_query="How much does Butler Court cost?",
        retrieval_query=RetrievalQuery(
            query_text="Butler Court cost",
            top_k=6,
            domain="accommodation",
            domains=["accommodation"],
            entity_tags=["Butler Court"],
        ),
        evidence=_weak_evidence(),
        draft_answer=AnswerResult(
            answer="Butler Court costs GBP 120 per week.",
            grounded=True,
            confidence=0.9,
            citations=[],
            used_evidence_count=0,
            fallback_used=False,
        ),
    )
    assert llm.calls == 1

    _ = agent.evaluate(
        user_query="How much does Butler Court cost?",
        retrieval_query=RetrievalQuery(
            query_text="Butler Court cost",
            top_k=6,
            domain="accommodation",
            domains=["accommodation"],
        ),
        evidence=[],
        draft_answer=AnswerResult(
            answer="I do not know the answer.",
            grounded=False,
            confidence=0.0,
            citations=[],
            used_evidence_count=0,
            fallback_used=True,
        ),
    )
    assert llm.calls == 1
