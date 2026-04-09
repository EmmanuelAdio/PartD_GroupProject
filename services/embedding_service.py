from __future__ import annotations

import hashlib
import os
from typing import List, Sequence


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

if load_dotenv is not None:
    load_dotenv()

class EmbeddingService:
    """Thin wrapper around the OpenAI embeddings API.

    Designed to be injected into the ingestion service and retriever later.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 64,
        max_batch_tokens: int = 250_000,
        chars_per_token: int = 3,
        max_input_tokens: int = 8_000,
        input_token_safety_margin: int = 128,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is not installed. Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY (or OPEN_API_KEY) is not set. Add it to your environment before using EmbeddingService."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_batch_tokens = int(
            os.getenv("OPENAI_EMBED_MAX_BATCH_TOKENS", str(max_batch_tokens))
        )
        self.chars_per_token = max(
            1,
            int(os.getenv("OPENAI_EMBED_CHARS_PER_TOKEN", str(chars_per_token))),
        )
        self.max_input_tokens = max(
            256,
            int(os.getenv("OPENAI_EMBED_MAX_INPUT_TOKENS", str(max_input_tokens))),
        )
        self.input_token_safety_margin = max(
            0,
            int(
                os.getenv(
                    "OPENAI_EMBED_INPUT_TOKEN_SAFETY_MARGIN",
                    str(input_token_safety_margin),
                )
            ),
        )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        text = self._clean_text(text)
        if not text:
            raise ValueError("Cannot embed an empty string.")

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple text strings in token-aware batches."""
        cleaned = []
        for t in texts:
            value = self._clean_text(t)
            if value:
                cleaned.append(value)
        if not cleaned:
            return []

        all_embeddings: List[List[float]] = []
        for batch in self._iter_token_aware_batches(cleaned):
            vectors = self._embed_batch_with_fallback(batch)
            all_embeddings.extend(vectors)

        return all_embeddings

    def get_model_name(self) -> str:
        return self.model

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _iter_token_aware_batches(self, cleaned: Sequence[str]) -> List[List[str]]:
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0

        for text in cleaned:
            token_estimate = self._estimate_tokens(text)
            if not current:
                current = [text]
                current_tokens = token_estimate
                continue

            would_exceed_count = len(current) >= self.batch_size
            would_exceed_tokens = (current_tokens + token_estimate) > self.max_batch_tokens
            if would_exceed_count or would_exceed_tokens:
                batches.append(current)
                current = [text]
                current_tokens = token_estimate
            else:
                current.append(text)
                current_tokens += token_estimate

        if current:
            batches.append(current)
        return batches

    def _embed_batch_with_fallback(self, batch: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            if len(batch) == 1:
                if self._is_single_input_too_long_error(exc):
                    parts = self._split_single_input(batch[0])
                    if len(parts) <= 1:
                        half = max(1, len(batch[0]) // 2)
                        parts = self._split_by_chars(batch[0], half)
                        if len(parts) <= 1:
                            raise
                    part_vectors = [
                        self._embed_batch_with_fallback([part])[0]
                        for part in parts
                    ]
                    return [self._mean_vector(part_vectors)]
                raise

            if not self._is_max_tokens_error(exc):
                raise

            mid = len(batch) // 2
            left = self._embed_batch_with_fallback(batch[:mid])
            right = self._embed_batch_with_fallback(batch[mid:])
            return left + right

    def _estimate_tokens(self, text: str) -> int:
        # Conservative heuristic for batch packing without adding tokenizer deps.
        return max(1, (len(text) // self.chars_per_token) + 8)

    @staticmethod
    def _is_max_tokens_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "max_tokens_per_request" in message
            or "maximum input length is" in message
            or ("max" in message and "tokens" in message and "per request" in message)
        )

    @staticmethod
    def _is_single_input_too_long_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "maximum input length is" in message
            or ("input[" in message and "tokens" in message and "maximum" in message)
        )

    def _split_single_input(self, text: str) -> List[str]:
        budget_tokens = max(64, self.max_input_tokens - self.input_token_safety_margin)
        max_chars = max(64, budget_tokens * self.chars_per_token)
        if len(text) <= max_chars:
            return [text]

        lines = text.splitlines()
        if len(lines) <= 1:
            return self._split_by_chars(text, max_chars)

        parts: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in lines:
            candidate_len = current_len + len(line) + (1 if current else 0)
            if current and candidate_len > max_chars:
                parts.append("\n".join(current))
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len = candidate_len

        if current:
            parts.append("\n".join(current))

        normalized = [self._clean_text(part) for part in parts if self._clean_text(part)]
        if any(len(part) > max_chars for part in normalized):
            resplit: List[str] = []
            for part in normalized:
                if len(part) <= max_chars:
                    resplit.append(part)
                else:
                    resplit.extend(self._split_by_chars(part, max_chars))
            normalized = resplit
        return normalized or [text]

    @staticmethod
    def _split_by_chars(text: str, max_chars: int) -> List[str]:
        parts: List[str] = []
        for i in range(0, len(text), max_chars):
            piece = text[i : i + max_chars].strip()
            if piece:
                parts.append(piece)
        return parts or [text]

    @staticmethod
    def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
        if not vectors:
            return []
        size = len(vectors[0])
        out = [0.0] * size
        for vector in vectors:
            if len(vector) != size:
                raise ValueError("Embedding segment size mismatch while averaging vectors.")
            for i, value in enumerate(vector):
                out[i] += float(value)
        count = float(len(vectors))
        return [value / count for value in out]


class DeterministicEmbeddingService:
    """Deterministic local embedder for integration tests without OpenAI."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        text = self._clean_text(text)
        if not text:
            raise ValueError("Cannot embed an empty string.")
        return self._vectorize(text)

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            cleaned = self._clean_text(text)
            if cleaned:
                vectors.append(self._vectorize(cleaned))
        return vectors

    def get_model_name(self) -> str:
        return f"deterministic-hash-{self.dim}"

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _vectorize(self, text: str) -> List[float]:
        buffer = bytearray()
        counter = 0
        while len(buffer) < self.dim:
            digest = hashlib.sha256(f"{text}|{counter}".encode("utf-8")).digest()
            buffer.extend(digest)
            counter += 1

        values = list(buffer[: self.dim])
        return [((v / 255.0) * 2.0) - 1.0 for v in values]
