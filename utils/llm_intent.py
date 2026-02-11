import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class LLMIntentResult:
    intent: Optional[str]
    confidence: float
    raw_response: Optional[str] = None


class BaseIntentClassifier:
    def classify_intent(self, text: str, allowed_intents: Iterable[str]) -> LLMIntentResult:
        raise NotImplementedError


class NullIntentClassifier(BaseIntentClassifier):
    def classify_intent(self, text: str, allowed_intents: Iterable[str]) -> LLMIntentResult:
        return LLMIntentResult(intent=None, confidence=0.0, raw_response=None)


class OpenAICompatibleIntentClassifier(BaseIntentClassifier):
    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 20,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def classify_intent(self, text: str, allowed_intents: Iterable[str]) -> LLMIntentResult:
        intents = sorted(set(i for i in allowed_intents if i))
        if not intents:
            return LLMIntentResult(intent=None, confidence=0.0, raw_response=None)

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 50,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. Return ONLY valid JSON with keys "
                        "'intent' and 'confidence'. 'intent' must be one of the provided "
                        "labels or null. 'confidence' must be a float between 0 and 1."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Text: " + text + "\n" +
                        "Allowed intents: " + ", ".join(intents)
                    ),
                },
            ],
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = Request(self.endpoint_url, data=body, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError):
            return LLMIntentResult(intent=None, confidence=0.0, raw_response=None)

        content = _extract_openai_message_content(response_body)
        if not content:
            return LLMIntentResult(intent=None, confidence=0.0, raw_response=response_body)

        parsed = _safe_parse_json(content)
        if parsed:
            intent = parsed.get("intent")
            confidence = _safe_float(parsed.get("confidence"), default=0.0)
            if intent in intents:
                return LLMIntentResult(intent=intent, confidence=confidence, raw_response=content)
            if intent in (None, "null"):
                return LLMIntentResult(intent=None, confidence=confidence, raw_response=content)

        # Fallback: pick the first allowed intent mentioned in the text.
        lowered = content.lower()
        for intent in intents:
            if intent.lower() in lowered:
                return LLMIntentResult(intent=intent, confidence=0.5, raw_response=content)

        return LLMIntentResult(intent=None, confidence=0.0, raw_response=content)


def build_intent_classifier_from_env() -> BaseIntentClassifier:
    provider = os.environ.get("LLM_PROVIDER", "none").strip().lower()
    if provider in {"", "none", "off", "disabled"}:
        return NullIntentClassifier()

    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        endpoint_url = os.environ.get(
            "LLM_INTENT_ENDPOINT",
            "https://api.openai.com/v1/chat/completions",
        )
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()
        if not api_key:
            return NullIntentClassifier()
        return OpenAICompatibleIntentClassifier(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model=model,
        )

    return NullIntentClassifier()


def _extract_openai_message_content(response_body: str) -> Optional[str]:
    parsed = _safe_parse_json(response_body)
    if not parsed:
        return None

    choices = parsed.get("choices")
    if not choices:
        return None

    message = choices[0].get("message") or {}
    return message.get("content")


def _safe_parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
