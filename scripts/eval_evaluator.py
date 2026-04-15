from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.orchestrator import QueryOrchestrator
from scripts.benchmark_questions import BENCHMARK_QUESTIONS
from scripts.eval_common import (
    compute_mean,
    default_query_embedder_backend,
    ensure_directory,
    flatten_timing_ms,
    json_dumps,
    load_project_env,
    parse_csv_list,
    save_documents_to_mongo,
    save_json,
    select_benchmark_questions,
    timestamp_slug,
    write_rows_to_csv,
)


EVALUATOR_COLLECTION = "eval_evaluator_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluator effectiveness evaluation on the benchmark set.")
    parser.add_argument("--category", default=None, help="Optional benchmark category filter.")
    parser.add_argument("--ids", default=None, help="Comma-separated benchmark ids to run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of questions to run.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k override passed into QueryOrchestrator.run().")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV output.")
    parser.add_argument("--mongo-db", default="open_day_knowledge")
    parser.add_argument("--mongo-collection", default=EVALUATOR_COLLECTION)
    parser.add_argument("--save-mongo", action="store_true", help="Also save row-level results to MongoDB.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_project_env()
    batch_started = time.perf_counter()

    benchmark_rows = select_benchmark_questions(
        BENCHMARK_QUESTIONS,
        category=args.category,
        ids=parse_csv_list(args.ids),
        limit=args.limit,
    )
    if not benchmark_rows:
        raise SystemExit("No benchmark questions matched the provided filters.")

    orchestrator = QueryOrchestrator(
        mongo_db=args.mongo_db,
        mongo_collection="kb_chuncks",
        embedder_backend=default_query_embedder_backend(),
    )

    output_dir = ensure_directory(args.output_dir)
    run_id = timestamp_slug()
    csv_path = output_dir / f"evaluator_eval_{run_id}.csv"
    json_path = output_dir / f"evaluator_eval_{run_id}.json"

    rows: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    for item in benchmark_rows:
        try:
            response = orchestrator.run(
                user_query=item["question"],
                top_k_override=args.top_k,
            )
            answerer_run = response.get("answerer_run", {})
            evaluator_run = response.get("evaluator_run", {})
            orchestration = response.get("orchestration_decision", {})
            retrieval_run = response.get("retrieval_run", {})
            timing = flatten_timing_ms(response)
            row = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "evaluator_decision": evaluator_run.get("effective_verdict")
                or evaluator_run.get("verdict")
                or orchestration.get("effective_verdict")
                or orchestration.get("final_verdict"),
                "initial_verdict": orchestration.get("initial_verdict"),
                "final_verdict": orchestration.get("final_verdict"),
                "effective_verdict": orchestration.get("effective_verdict"),
                "runtime_action": orchestration.get("runtime_action"),
                "retry_count": int(orchestration.get("revise_retries_used", 0) or 0),
                "fallback_used": bool(answerer_run.get("fallback_used"))
                or orchestration.get("runtime_action") == "fallback",
                "clarification_requested": _clarification_requested(
                    evaluator_run=evaluator_run,
                    orchestration=orchestration,
                ),
                "revised": int(orchestration.get("revise_retries_used", 0) or 0) > 0,
                "retry_improved_result": _retry_improved_result(
                    evaluator_run=evaluator_run,
                    orchestration=orchestration,
                    retrieval_run=retrieval_run,
                ),
                **timing,
                "final_answer": answerer_run.get("answer", ""),
                "answer_confidence": answerer_run.get("confidence"),
                "answer_grounded": answerer_run.get("grounded"),
                "used_evidence_count": answerer_run.get("used_evidence_count"),
                "evaluator_grounded": evaluator_run.get("grounded"),
                "evaluator_relevant": evaluator_run.get("relevant"),
                "evaluator_clear": evaluator_run.get("clear"),
                "evaluator_safe": evaluator_run.get("safe"),
                "issues": json_dumps(evaluator_run.get("issues", [])),
                "suggested_action": evaluator_run.get("suggested_action"),
                "suggested_filters": json_dumps(evaluator_run.get("suggested_filters")),
                "clarification_question": evaluator_run.get("clarification_question"),
                "notes": evaluator_run.get("notes"),
                "attempt_used": retrieval_run.get("attempt_used"),
                "result_count": retrieval_run.get("result_count"),
                "retrieved_result_count": retrieval_run.get("retrieved_result_count"),
                "attempts_log": json_dumps(retrieval_run.get("attempts_log", [])),
                "success": True,
                "error_message": "",
            }
            rows.append(row)
            raw_results.append(
                {
                    "benchmark_item": dict(item),
                    "response": response,
                }
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "evaluator_decision": "",
                    "initial_verdict": "",
                    "final_verdict": "",
                    "effective_verdict": "",
                    "runtime_action": "",
                    "retry_count": 0,
                    "fallback_used": False,
                    "clarification_requested": False,
                    "revised": False,
                    "retry_improved_result": None,
                    "total_time_ms": 0.0,
                    "processor_time_ms": 0.0,
                    "retriever_time_ms": 0.0,
                    "answerer_time_ms": 0.0,
                    "evaluator_time_ms": 0.0,
                    "final_answer": "",
                    "answer_confidence": None,
                    "answer_grounded": None,
                    "used_evidence_count": 0,
                    "evaluator_grounded": None,
                    "evaluator_relevant": None,
                    "evaluator_clear": None,
                    "evaluator_safe": None,
                    "issues": "[]",
                    "suggested_action": "",
                    "suggested_filters": "null",
                    "clarification_question": "",
                    "notes": "",
                    "attempt_used": None,
                    "result_count": 0,
                    "retrieved_result_count": 0,
                    "attempts_log": "[]",
                    "success": False,
                    "error_message": error_message,
                }
            )
            raw_results.append(
                {
                    "benchmark_item": dict(item),
                    "error": error_message,
                }
            )

    batch_total_runtime_ms = round((time.perf_counter() - batch_started) * 1000.0, 3)
    summary = _build_summary(rows, batch_total_runtime_ms=batch_total_runtime_ms)
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "summary": summary,
        "rows": rows,
        "raw_results": raw_results,
    }

    write_rows_to_csv(rows, csv_path)
    save_json(json_path, payload)

    mongo_error: Optional[str] = None
    if args.save_mongo:
        try:
            documents = [
                {
                    "run_id": run_id,
                    "generated_at": payload["generated_at"],
                    "row": row,
                    "raw_result": raw_result,
                }
                for row, raw_result in zip(rows, raw_results)
            ]
            save_documents_to_mongo(
                documents=documents,
                db_name=args.mongo_db,
                collection_name=args.mongo_collection,
            )
        except Exception as exc:
            mongo_error = f"{type(exc).__name__}: {exc}"

    _print_summary(summary, csv_path=csv_path, json_path=json_path, mongo_error=mongo_error)
    return 1 if mongo_error else 0


def _clarification_requested(*, evaluator_run: Dict[str, Any], orchestration: Dict[str, Any]) -> bool:
    if evaluator_run.get("clarification_question"):
        return True
    verdicts = {
        evaluator_run.get("verdict"),
        evaluator_run.get("effective_verdict"),
        orchestration.get("final_verdict"),
        orchestration.get("effective_verdict"),
        orchestration.get("runtime_action"),
    }
    return "ask_clarification" in verdicts


def _retry_improved_result(
    *,
    evaluator_run: Dict[str, Any],
    orchestration: Dict[str, Any],
    retrieval_run: Dict[str, Any],
) -> Optional[bool]:
    retry_count = int(orchestration.get("revise_retries_used", 0) or 0)
    if retry_count <= 0:
        return None

    initial_verdict = orchestration.get("initial_verdict")
    effective_verdict = orchestration.get("effective_verdict") or evaluator_run.get("effective_verdict")
    attempts_log = retrieval_run.get("attempts_log", [])
    if not isinstance(attempts_log, list) or not attempts_log:
        return None

    first_hits = attempts_log[0].get("hits") if isinstance(attempts_log[0], dict) else None
    last_hits = attempts_log[-1].get("hits") if isinstance(attempts_log[-1], dict) else None
    if first_hits is None or last_hits is None:
        return None
    if effective_verdict == "pass" and initial_verdict != "pass":
        return True
    return bool(last_hits > first_hits)


def _build_summary(rows: List[Dict[str, Any]], *, batch_total_runtime_ms: float) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("evaluator_decision") or "unknown")
        counts[key] = counts.get(key, 0) + 1

    success_count = sum(bool(row.get("success")) for row in rows)
    fallback_count = sum(bool(row.get("fallback_used")) for row in rows)
    clarification_count = sum(bool(row.get("clarification_requested")) for row in rows)
    revised_count = sum(bool(row.get("revised")) for row in rows)
    retry_count_mean = compute_mean(row.get("retry_count") for row in rows)
    total_time_mean = compute_mean(row.get("total_time_ms") for row in rows)

    return {
        "tests_run": len(rows),
        "success_count": success_count,
        "error_count": len(rows) - success_count,
        "batch_total_runtime_ms": batch_total_runtime_ms,
        "batch_total_runtime_human": _format_duration_ms(batch_total_runtime_ms),
        "average_total_time_ms": round(float(total_time_mean), 3) if total_time_mean is not None else None,
        "average_retry_count": round(float(retry_count_mean), 3) if retry_count_mean is not None else None,
        "fallback_count": fallback_count,
        "clarification_requested_count": clarification_count,
        "revised_count": revised_count,
        "evaluator_outcome_counts": counts,
    }


def _print_summary(
    summary: Dict[str, Any],
    *,
    csv_path: Path,
    json_path: Path,
    mongo_error: Optional[str],
) -> None:
    counts = summary.get("evaluator_outcome_counts", {})

    print(
        "Batch total runtime: "
        f"{summary['batch_total_runtime_human']} ({summary['batch_total_runtime_ms']} ms)"
    )
    print("Evaluator outcome counts:")
    for key in sorted(counts.keys()):
        print(f"  {key}: {counts[key]}")
    print(f"Tests run: {summary['tests_run']}")
    print(f"Successful runs: {summary['success_count']}")
    print(f"Errored runs: {summary['error_count']}")
    print(f"Fallback used: {summary['fallback_count']}")
    print(f"Clarification requested: {summary['clarification_requested_count']}")
    print(f"Revised answers: {summary['revised_count']}")
    print(f"Average retry count: {summary['average_retry_count']}")
    print(f"Average total time (ms): {summary['average_total_time_ms']}")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    if mongo_error:
        print(f"Mongo warning: {mongo_error}")


def _format_duration_ms(duration_ms: float) -> str:
    total_seconds = max(duration_ms, 0.0) / 1000.0
    minutes, seconds = divmod(total_seconds, 60.0)
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"


if __name__ == "__main__":
    raise SystemExit(main())
