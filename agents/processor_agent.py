from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

try:
    from schemas.models import RetrievalQuery
except ImportError:  # pragma: no cover
    from ..schemas.models import RetrievalQuery

try:
    from services.llm_services import LLMService
except ImportError:  # pragma: no cover
    from ..services.llm_services import LLMService


class ProcessorAgent:
    """LLM-first query planner that converts user text into `RetrievalQuery`.

    Design contract:
    - This class does not retrieve documents or talk to MongoDB.
    - This class does not generate final user answers.
    - This class only returns a structured retrieval request for `RetrieverService`.
    """

    DEFAULT_DOMAINS = [
        "accommodation",
        "courses",
        "sports",
        "admissions",
        "finance",
        "general",
        "unknown",
    ]
    DEFAULT_SECTIONS = [
        "prices",
        "entry_requirements",
        "modules",
        "fees",
        "funding",
        "facilities",
        "contact",
        "overview",
        "qna",
        "json_fields",
        "text",
    ]

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        llm_model: str = "gpt-4o-mini",
        allowed_domains: Optional[List[str]] = None,
        allowed_sections: Optional[List[str]] = None,
        default_top_k: int = 8,
        max_entity_tags: int = 12,
    ) -> None:
        """Create a processor agent.

        Args:
            llm_service: Optional injected LLM service. If omitted, this class
                creates `LLMService(model=llm_model)`.
            llm_model: Model used when `llm_service` is not injected.
            allowed_domains: Optional retrieval domain values.
            allowed_sections: Optional retrieval section values.
            default_top_k: Default retrieval size for planned queries.
            max_entity_tags: Max number of entity tags retained post-parse.
        """
        self.llm = llm_service or LLMService(model=llm_model)
        self.default_top_k = int(default_top_k)
        self.max_entity_tags = int(max_entity_tags)

        env_domains = self._split_csv_env("PROCESSOR_ALLOWED_DOMAINS")
        env_sections = self._split_csv_env("PROCESSOR_ALLOWED_SECTIONS")
        self.allowed_domains = allowed_domains or env_domains or list(self.DEFAULT_DOMAINS)
        self.allowed_sections = allowed_sections or env_sections or list(self.DEFAULT_SECTIONS)

    def process(self, user_query: str) -> RetrievalQuery:
        """Convert a raw user query into a safe `RetrievalQuery` plan."""
        normalized_query = self._normalize_query(user_query)
        if not normalized_query:
            raise ValueError("user_query must not be empty.")

        schema_context = self.get_schema_context()
        system_prompt = self._build_system_prompt(schema_context)
        user_prompt = self._build_user_prompt(user_query=user_query, normalized_query=normalized_query)

        llm_payload = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=450,
        )
        planned = self._parse_llm_payload(llm_payload=llm_payload, normalized_query=normalized_query)
        return self._post_process_retrieval_query(planned=planned, user_query=user_query, schema_context=schema_context)

    def build_retrieval_query(self, user_query: str) -> RetrievalQuery:
        """Alias for `process()` to keep orchestration call sites readable."""
        return self.process(user_query)

    def get_schema_context(self) -> Dict[str, Any]:
        """Build retrieval schema context injected into the LLM prompt.

        Keeping values in this method avoids hardcoding schema values directly
        inside prompt text and makes orchestration/config wiring straightforward.
        """
        model_schema = RetrievalQuery.model_json_schema()
        top_k_schema = model_schema.get("properties", {}).get("top_k", {})
        return {
            "allowed_domains": self.allowed_domains,
            "allowed_sections": self.allowed_sections,
            "default_top_k": self.default_top_k,
            "top_k_min": int(top_k_schema.get("minimum", 1)),
            "top_k_max": int(top_k_schema.get("maximum", 50)),
            "max_entity_tags": self.max_entity_tags,
            "schema_fields": [
                "query_text",
                "top_k",
                "domain",
                "domains",
                "entity_tags",
                "section",
                "sections",
                "source_ids",
                "version",
                "vector_k",
                "text_k",
            ],
        }

    def _build_system_prompt(self, schema_context: Dict[str, Any]) -> str:
        """Build strict planner instructions for JSON-only LLM output."""
        return (
            "You are the Processor Agent for a university RAG system.\n"
            "Your job is retrieval planning only.\n"
            "You are not answering the user's question.\n"
            "You are creating a retrieval plan for MongoDB hybrid search.\n"
            "Preserve exact entity names from the user's query when possible.\n"
            "Rewrite query_text to improve retrieval quality.\n"
            "Use known domain and section values when justified by the query.\n"
            "Do not invent unsupported filters if the query does not justify them.\n"
            "Return only valid JSON matching the RetrievalQuery schema.\n\n"
            f"Retrieval schema context:\n{json.dumps(schema_context, ensure_ascii=False)}"
        )

    @staticmethod
    def _build_user_prompt(user_query: str, normalized_query: str) -> str:
        """Build the user prompt with raw and normalized input text."""
        return (
            "Plan a RetrievalQuery object for the user input.\n"
            "Output JSON object only.\n\n"
            f"Raw user query:\n{user_query}\n\n"
            f"Normalized user query:\n{normalized_query}"
        )

    def _parse_llm_payload(self, llm_payload: Dict[str, Any], normalized_query: str) -> RetrievalQuery:
        """Parse LLM JSON output into `RetrievalQuery` with safe fallback."""
        payload = dict(llm_payload or {})
        payload.setdefault("query_text", normalized_query)
        payload.setdefault("top_k", self.default_top_k)

        try:
            return RetrievalQuery.model_validate(payload)
        except ValidationError:
            # Fallback keeps the planning stage robust even if LLM output is malformed.
            return RetrievalQuery(query_text=normalized_query, top_k=self.default_top_k)

    def _post_process_retrieval_query(
        self,
        *,
        planned: RetrievalQuery,
        user_query: str,
        schema_context: Dict[str, Any],
    ) -> RetrievalQuery:
        """Apply small safety/consistency cleanups for retrieval execution."""
        allowed_domains = {d.lower() for d in schema_context["allowed_domains"]}
        allowed_sections = {s.lower() for s in schema_context["allowed_sections"]}

        query_text = self._normalize_query(planned.query_text) or self._normalize_query(user_query)

        domains = self._dedupe([planned.domain, *planned.domains], lowercase=True)
        domains = [d for d in domains if d in allowed_domains]

        sections = self._dedupe([planned.section, *planned.sections], lowercase=False)
        sections = [s for s in sections if s.lower() in allowed_sections]

        entity_tags = self._dedupe(planned.entity_tags, lowercase=False)
        entity_tags = self._preserve_entity_casing_from_query(entity_tags, user_query)
        entity_tags = entity_tags[: int(schema_context["max_entity_tags"])]

        source_ids = self._dedupe(planned.source_ids, lowercase=False)

        top_k = self._clamp_int(
            planned.top_k,
            low=int(schema_context["top_k_min"]),
            high=int(schema_context["top_k_max"]),
        )

        update: Dict[str, Any] = {
            "query_text": query_text,
            "top_k": top_k,
            "domain": domains[0] if domains else None,
            "domains": domains,
            "section": sections[0] if sections else None,
            "sections": sections,
            "entity_tags": entity_tags,
            "source_ids": source_ids,
        }

        if planned.vector_k is not None:
            update["vector_k"] = self._clamp_int(planned.vector_k, low=1, high=250)
        if planned.text_k is not None:
            update["text_k"] = self._clamp_int(planned.text_k, low=1, high=250)

        return planned.model_copy(update=update)

    @staticmethod
    def _normalize_query(text: str) -> str:
        """Normalize user text for stable planning and retrieval matching."""
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _split_csv_env(name: str) -> List[str]:
        """Read a comma-separated environment variable into a clean list."""
        raw = os.getenv(name, "")
        values = [part.strip() for part in raw.split(",")]
        return [v for v in values if v]

    @staticmethod
    def _dedupe(values: List[Any], *, lowercase: bool) -> List[str]:
        """Deduplicate and clean a string list while preserving first order."""
        out: List[str] = []
        seen = set()
        for value in values:
            item = str(value or "").strip()
            if not item:
                continue
            key = item.lower() if lowercase else item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item.lower() if lowercase else item)
        return out

    @staticmethod
    def _clamp_int(value: int, *, low: int, high: int) -> int:
        """Clamp integer values into a safe closed interval."""
        return max(low, min(high, int(value)))

    @staticmethod
    def _preserve_entity_casing_from_query(entity_tags: List[str], user_query: str) -> List[str]:
        """Prefer exact casing from the original query when substring matches exist."""
        out: List[str] = []
        seen = set()
        for tag in entity_tags:
            candidate = tag
            match = re.search(re.escape(tag), user_query, flags=re.IGNORECASE)
            if match:
                candidate = match.group(0)
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out
