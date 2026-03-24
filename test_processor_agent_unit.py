from __future__ import annotations

from typing import Any, Dict

from agents.processor_agent import ProcessorAgent


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


def test_processor_drops_hallucinated_version_filter() -> None:
    agent = ProcessorAgent(
        llm_service=_StubLLMService(
            {
                "query_text": "How much is Butler Court accommodation?",
                "top_k": 8,
                "domain": "accommodation",
                "domains": ["accommodation"],
                "entity_tags": ["Butler Court"],
                "section": "prices",
                "sections": ["prices"],
                "version": "1.0",
            }
        )
    )

    plan = agent.process("How much is Butler Court accommodation?")

    assert plan.domain == "accommodation"
    assert plan.section == "prices"
    assert plan.version is None


def test_processor_keeps_version_when_user_explicitly_requests_it() -> None:
    agent = ProcessorAgent(
        llm_service=_StubLLMService(
            {
                "query_text": "Show accommodation data from ingest-v2",
                "top_k": 5,
                "domains": ["accommodation"],
                "version": "ingest-v2",
            }
        )
    )

    plan = agent.process("Show accommodation data from ingest-v2")

    assert plan.version == "ingest-v2"
