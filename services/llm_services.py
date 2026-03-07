from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from schemas.models import ChunkTags
except ImportError:  # pragma: no cover
    from ..schemas.models import ChunkTags


class LLMService:
    """Simple wrapper for text generation and JSON-structured outputs.

    Use this for:
    - chunk/domain tagging during ingestion
    - query classification in the Processor agent later
    - answer generation only if you want a shared LLM wrapper
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is not installed. Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your environment before using LLMService."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 400,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def generate_json(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._safe_json_loads(raw)

    def tag_chunk(
        self,
        text: str,
        hint_domain: Optional[str] = None,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ChunkTags:
        snippet = text[:5000]
        system_prompt = (
            "You are a precise data tagging assistant for a university RAG ingestion pipeline. "
            "Return only valid JSON with keys: domain, entity_tags, key_fields, confidence. "
            "domain must be one of: accommodation, courses, sports, admissions, finance, general, unknown. "
            "entity_tags must be an array of short strings. "
            "key_fields must be an object of useful retrieval hints extracted from the chunk. "
            "confidence must be a float between 0 and 1."
        )
        user_prompt = (
            f"Source ID: {source_id or 'unknown'}\n"
            f"Source type: {source_type or 'unknown'}\n"
            f"Hint domain: {hint_domain or 'unknown'}\n\n"
            f"Chunk:\n{snippet}"
        )

        raw = self.generate_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=350,
        )

        domain = str(raw.get("domain") or hint_domain or "unknown").strip().lower()
        if domain not in {"accommodation", "courses", "sports", "admissions", "finance", "general", "unknown"}:
            domain = hint_domain or "unknown"

        entity_tags = self._normalize_entities(raw.get("entity_tags"))
        key_fields = raw.get("key_fields") if isinstance(raw.get("key_fields"), dict) else {}
        confidence = self._coerce_confidence(raw.get("confidence"), default=0.6)

        return ChunkTags(
            domain=domain,
            entity_tags=entity_tags,
            key_fields=key_fields,
            confidence=confidence,
        )

    @staticmethod
    def _safe_json_loads(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {}
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _normalize_entities(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []

        out: List[str] = []
        seen = set()
        for item in raw:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= 12:
                break
        return out

    @staticmethod
    def _coerce_confidence(value: Any, default: float) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = default
        return max(0.0, min(1.0, num))
