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
        """Embed multiple text strings in batches."""
        cleaned = []
        for t in texts:
            value = self._clean_text(t)
            if value:
                cleaned.append(value)
        if not cleaned:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(cleaned), self.batch_size):
            batch = cleaned[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

    def get_model_name(self) -> str:
        return self.model

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()


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
