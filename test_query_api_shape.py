from __future__ import annotations

from app.main import _shape_query_response


def _sample_query_result():
    return {
        "user_query": "How much is Butler Court accommodation?",
        "answerer_run": {
            "answer": "Butler Court costs GBP 126.68 per week, with a total contract cost of GBP 5,302.27.",
            "grounded": True,
            "confidence": 0.98,
            "citations": [
                {
                    "evidence_id": 1,
                    "chunk_id": "chunk-1",
                    "source_id": "accommodation_halls",
                }
            ],
        },
        "retrieval_run": {
            "evidence": [{"chunk_id": "chunk-1"}],
        },
    }


def test_shape_query_response_returns_minimal_payload_by_default() -> None:
    payload = _shape_query_response(_sample_query_result(), debug=False)

    assert payload == {
        "query": "How much is Butler Court accommodation?",
        "answer": "Butler Court costs GBP 126.68 per week, with a total contract cost of GBP 5,302.27.",
        "grounded": True,
        "confidence": 0.98,
        "citations": [
            {
                "evidence_id": 1,
                "chunk_id": "chunk-1",
                "source_id": "accommodation_halls",
            }
        ],
    }


def test_shape_query_response_keeps_debug_payload_when_requested() -> None:
    result = _sample_query_result()
    assert _shape_query_response(result, debug=True) == result
