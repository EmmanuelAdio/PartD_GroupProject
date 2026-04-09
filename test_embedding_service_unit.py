from __future__ import annotations

from types import SimpleNamespace

import services.embedding_service as embedding_module
from services.embedding_service import EmbeddingService


class _FakeEmbeddingsClient:
    def __init__(
        self,
        *,
        max_items_before_fail: int | None = None,
        max_chars_per_item_before_fail: int | None = None,
    ) -> None:
        self.calls: list[int] = []
        self.call_payloads: list[list[str]] = []
        self.success_payloads: list[list[str]] = []
        self.max_items_before_fail = max_items_before_fail
        self.max_chars_per_item_before_fail = max_chars_per_item_before_fail

    def create(self, *, model: str, input):  # noqa: A002 - external API parity
        if isinstance(input, str):
            rows = [input]
        else:
            rows = list(input)
        self.calls.append(len(rows))
        self.call_payloads.append(rows)

        if self.max_items_before_fail is not None and len(rows) > self.max_items_before_fail:
            raise RuntimeError(
                "Requested too many tokens, max_tokens_per_request exceeded"
            )
        if self.max_chars_per_item_before_fail is not None:
            for row in rows:
                if len(row) > self.max_chars_per_item_before_fail:
                    raise RuntimeError("Invalid 'input[0]': maximum input length is 8192 tokens.")

        self.success_payloads.append(rows)

        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(row))]) for row in rows]
        )


class _FakeOpenAI:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_items_before_fail: int | None = None,
        max_chars_per_item_before_fail: int | None = None,
    ) -> None:
        _ = api_key
        self.embeddings = _FakeEmbeddingsClient(
            max_items_before_fail=max_items_before_fail,
            max_chars_per_item_before_fail=max_chars_per_item_before_fail,
        )


def test_embed_many_uses_token_aware_batching(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = _FakeOpenAI()
    monkeypatch.setattr(embedding_module, "OpenAI", lambda api_key=None: client)

    service = EmbeddingService(
        batch_size=10,
        max_batch_tokens=20,
        chars_per_token=1,
    )
    vectors = service.embed_many(["a" * 8, "b" * 7, "c" * 7, "d" * 8])

    assert len(vectors) == 4
    # token estimates become 16, 15, 15, 16 => one item per request with threshold 20
    assert client.embeddings.calls == [1, 1, 1, 1]


def test_embed_many_splits_batch_when_openai_reports_max_tokens(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = _FakeOpenAI(max_items_before_fail=2)
    monkeypatch.setattr(embedding_module, "OpenAI", lambda api_key=None: client)

    service = EmbeddingService(
        batch_size=8,
        max_batch_tokens=999999,
        chars_per_token=4,
    )

    vectors = service.embed_many(["one", "two", "three", "four"])

    assert len(vectors) == 4
    # First attempt with 4 fails, then recursive 2 + 2 succeeds.
    assert client.embeddings.calls == [4, 2, 2]


def test_embed_many_splits_oversized_single_input_and_averages(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = _FakeOpenAI(max_chars_per_item_before_fail=20)
    monkeypatch.setattr(embedding_module, "OpenAI", lambda api_key=None: client)

    service = EmbeddingService(
        batch_size=8,
        max_batch_tokens=999999,
        chars_per_token=1,
        max_input_tokens=20,
        input_token_safety_margin=0,
    )

    long_text = ("A" * 15) + "\n" + ("B" * 15) + "\n" + ("C" * 15)
    vectors = service.embed_many([long_text, "short text"])

    assert len(vectors) == 2
    # First call tries long + short, then long is retried alone and split.
    assert client.embeddings.calls[0] == 2
    assert any(size == 1 for size in client.embeddings.calls[1:])
    # Ensure no successful per-item call exceeds our fake max length.
    for payload in client.embeddings.success_payloads:
        for row in payload:
            assert len(row) <= 20
