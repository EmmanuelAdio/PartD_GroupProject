from __future__ import annotations

import app.main as main_module


def _sample_feedback_result() -> dict:
    return {
        "user_query": "How much is Butler Court accommodation?",
        "answerer_run": {
            "answer": "Butler Court costs GBP 126.68 per week.",
            "grounded": True,
            "confidence": 0.94,
            "citations": [
                {
                    "evidence_id": 1,
                    "chunk_id": "chunk-1",
                    "source_id": "accommodation_halls",
                }
            ],
        },
        "orchestration_decision": {
            "runtime_action": "pass",
        },
    }


def test_feedback_endpoint_acknowledges_yes_without_retry(monkeypatch) -> None:
    calls = {"count": 0, "logged": None}

    class _StubOrchestrator:
        def run_with_feedback(self, **kwargs):
            calls["count"] += 1
            return kwargs

    def _capture_log(**kwargs):
        calls["logged"] = kwargs

    monkeypatch.setattr(main_module, "query_orchestrator", _StubOrchestrator())
    monkeypatch.setattr(main_module, "_log_feedback_event", _capture_log)

    response = main_module.run_feedback(
        main_module.FeedbackRequest(
            user_query="How much is Butler Court accommodation?",
            last_answer="Butler Court costs GBP 126.68 per week.",
            resolved=True,
        )
    )

    assert response.model_dump() == {
        "action": "acknowledged",
        "answer_payload": None,
    }
    assert calls["count"] == 0
    assert calls["logged"]["resolved"] is True
    assert calls["logged"]["retry_succeeded"] is False


def test_feedback_endpoint_retries_no_path_once(monkeypatch) -> None:
    calls = {"count": 0, "logged": None}

    class _StubOrchestrator:
        def run_with_feedback(self, **kwargs):
            calls["count"] += 1
            assert kwargs["user_query"] == "Tell me about accommodation fees"
            assert kwargs["last_answer"] == "Accommodation costs vary."
            assert kwargs["reason"] == "missing_detail"
            return _sample_feedback_result()

    def _capture_log(**kwargs):
        calls["logged"] = kwargs

    monkeypatch.setattr(main_module, "query_orchestrator", _StubOrchestrator())
    monkeypatch.setattr(main_module, "_log_feedback_event", _capture_log)

    response = main_module.run_feedback(
        main_module.FeedbackRequest(
            user_query="Tell me about accommodation fees",
            last_answer="Accommodation costs vary.",
            resolved=False,
            reason="missing_detail",
        )
    )

    assert calls["count"] == 1
    assert response.model_dump() == {
        "action": "retried",
        "answer_payload": {
            "query": "How much is Butler Court accommodation?",
            "answer": "Butler Court costs GBP 126.68 per week.",
            "grounded": True,
            "confidence": 0.94,
            "citations": [
                {
                    "evidence_id": 1,
                    "chunk_id": "chunk-1",
                    "source_id": "accommodation_halls",
                }
            ],
        },
    }
    assert calls["logged"]["resolved"] is False
    assert calls["logged"]["retry_succeeded"] is True
