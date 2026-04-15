from __future__ import annotations

from scripts.benchmark_questions import BENCHMARK_QUESTIONS
from scripts.eval_common import extract_context_texts, select_benchmark_questions


def test_extract_context_texts_reads_direct_text_field() -> None:
    response = {
        "retrieval_run": {
            "evidence": [
                {"text": "Direct context"},
                {"text": "Another context"},
            ]
        }
    }

    assert extract_context_texts(response) == ["Direct context", "Another context"]


def test_extract_context_texts_handles_nested_payload_shapes() -> None:
    response = {
        "retrieval_run": {
            "evidence": [
                {"item": {"text": "Nested item text"}},
                {"document": {"page_content": "Nested page content"}},
                {"chunk": {"payload": {"content": "Deeply nested content"}}},
            ]
        }
    }

    assert extract_context_texts(response) == [
        "Nested item text",
        "Nested page content",
        "Deeply nested content",
    ]


def test_select_benchmark_questions_filters_by_category_and_ids() -> None:
    selected = select_benchmark_questions(
        BENCHMARK_QUESTIONS,
        category="accommodation",
        ids=["accommodation_01", "undergraduate_courses_01"],
        limit=5,
    )

    assert len(selected) == 1
    assert selected[0]["id"] == "accommodation_01"
