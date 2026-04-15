from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.orchestrator import QueryOrchestrator
from scripts.benchmark_questions import BENCHMARK_QUESTIONS
from scripts.eval_common import (
    compute_mean,
    default_query_embedder_backend,
    ensure_directory,
    extract_context_texts,
    flatten_timing_ms,
    has_openai_api_key,
    load_project_env,
    parse_csv_list,
    save_documents_to_mongo,
    save_json,
    select_benchmark_questions,
    timestamp_slug,
    write_rows_to_csv,
)


ACCURACY_COLLECTION = "eval_accuracy_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run accuracy evaluation on the benchmark set using RAGAS.")
    parser.add_argument("--category", default=None, help="Optional benchmark category filter.")
    parser.add_argument("--ids", default=None, help="Comma-separated benchmark ids to run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of questions to run.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k override passed into QueryOrchestrator.run().")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--mongo-db", default="open_day_knowledge")
    parser.add_argument("--mongo-collection", default=ACCURACY_COLLECTION)
    parser.add_argument("--save-mongo", action="store_true", help="Also save row-level results to MongoDB.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_project_env()
    batch_started = time.perf_counter()
    ragas_model = os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini")
    ragas_embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

    benchmark_rows = select_benchmark_questions(
        BENCHMARK_QUESTIONS,
        category=args.category,
        ids=parse_csv_list(args.ids),
        limit=args.limit,
    )
    if not benchmark_rows:
        raise SystemExit("No benchmark questions matched the provided filters.")

    print(
        "RAGAS config: "
        f"model={ragas_model}, embedding={ragas_embedding_model}, "
        f"questions={len(benchmark_rows)}"
    )

    orchestrator = QueryOrchestrator(
        mongo_db=args.mongo_db,
        mongo_collection="kb_chuncks",
        embedder_backend=default_query_embedder_backend(),
    )

    output_dir = ensure_directory(args.output_dir)
    run_id = timestamp_slug()
    csv_path = output_dir / f"accuracy_eval_{run_id}.csv"
    json_path = output_dir / f"accuracy_eval_{run_id}.json"

    rows: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    ragas_inputs: List[Dict[str, Any]] = []
    ragas_row_indexes: List[int] = []

    for item in benchmark_rows:
        try:
            response = orchestrator.run(
                user_query=item["question"],
                top_k_override=args.top_k,
            )
            answerer_run = response.get("answerer_run", {})
            evaluator_run = response.get("evaluator_run", {})
            orchestration = response.get("orchestration_decision", {})
            contexts = extract_context_texts(response)
            timing = flatten_timing_ms(response)
            row = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "ground_truth_answer": item["ground_truth_answer"],
                "generated_answer": answerer_run.get("answer", ""),
                "retrieved_context_count": len(contexts),
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "overall_metric_score": None,
                **timing,
                "evaluator_decision": evaluator_run.get("effective_verdict")
                or evaluator_run.get("verdict")
                or orchestration.get("effective_verdict")
                or orchestration.get("final_verdict"),
                "retry_count": int(orchestration.get("revise_retries_used", 0) or 0),
                "success": True,
                "error_message": "",
                "metric_error_message": "",
            }
            rows.append(row)
            raw_results.append(
                {
                    "benchmark_item": dict(item),
                    "retrieved_contexts": contexts,
                    "response": response,
                }
            )
            ragas_inputs.append(
                {
                    "question": item["question"],
                    "answer": answerer_run.get("answer", ""),
                    "contexts": contexts,
                    "ground_truth": item["ground_truth_answer"],
                }
            )
            ragas_row_indexes.append(len(rows) - 1)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "ground_truth_answer": item["ground_truth_answer"],
                    "generated_answer": "",
                    "retrieved_context_count": 0,
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                    "overall_metric_score": None,
                    "total_time_ms": 0.0,
                    "processor_time_ms": 0.0,
                    "retriever_time_ms": 0.0,
                    "answerer_time_ms": 0.0,
                    "evaluator_time_ms": 0.0,
                    "evaluator_decision": "",
                    "retry_count": 0,
                    "success": False,
                    "error_message": error_message,
                    "metric_error_message": "",
                }
            )
            raw_results.append(
                {
                    "benchmark_item": dict(item),
                    "error": error_message,
                }
            )

    ragas_error: Optional[str] = None
    if ragas_inputs:
        if not has_openai_api_key():
            ragas_error = (
                "OPENAI_API_KEY (or OPEN_API_KEY) is required to compute RAGAS metrics. "
                "Query outputs were still saved locally."
            )
        else:
            ragas_error = _apply_ragas_scores(rows, ragas_inputs, ragas_row_indexes)

    batch_total_runtime_ms = round((time.perf_counter() - batch_started) * 1000.0, 3)
    summary = {
        "tests_run": len(rows),
        "batch_total_runtime_ms": batch_total_runtime_ms,
        "batch_total_runtime_human": _format_duration_ms(batch_total_runtime_ms),
        "average_faithfulness": _round_or_none(compute_mean(row.get("faithfulness") for row in rows)),
        "average_answer_relevancy": _round_or_none(compute_mean(row.get("answer_relevancy") for row in rows)),
        "average_context_precision": _round_or_none(compute_mean(row.get("context_precision") for row in rows)),
        "average_context_recall": _round_or_none(compute_mean(row.get("context_recall") for row in rows)),
        "average_overall_metric_score": _round_or_none(compute_mean(row.get("overall_metric_score") for row in rows)),
        "scored_rows_faithfulness": sum(row.get("faithfulness") is not None for row in rows),
        "scored_rows_answer_relevancy": sum(row.get("answer_relevancy") is not None for row in rows),
        "scored_rows_context_precision": sum(row.get("context_precision") is not None for row in rows),
        "scored_rows_context_recall": sum(row.get("context_recall") is not None for row in rows),
    }

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "summary": summary,
        "ragas_error": ragas_error,
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

    _print_summary(summary, csv_path=csv_path, json_path=json_path, ragas_error=ragas_error, mongo_error=mongo_error)
    return 1 if ragas_error or mongo_error else 0


def _apply_ragas_scores(
    rows: List[Dict[str, Any]],
    ragas_inputs: Sequence[Dict[str, Any]],
    ragas_row_indexes: Sequence[int],
) -> Optional[str]:
    try:
        dataset_cls, ragas_evaluate, ragas_metrics, ragas_llm, ragas_embeddings = _load_ragas_components()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    try:
        dataset = dataset_cls.from_list(list(ragas_inputs))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*evaluate\(\) is deprecated.*",
                category=DeprecationWarning,
            )
            result = ragas_evaluate(
                dataset,
                metrics=list(ragas_metrics.values()),
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
            )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    score_rows = _result_rows(result)
    if len(score_rows) != len(ragas_row_indexes):
        return (
            "RAGAS returned a different number of score rows than inputs: "
            f"{len(score_rows)} != {len(ragas_row_indexes)}"
        )

    for row_index, score_row in zip(ragas_row_indexes, score_rows):
        rows[row_index]["faithfulness"] = _metric_value_from_row(score_row, "faithfulness")
        rows[row_index]["answer_relevancy"] = _metric_value_from_row(score_row, "answer_relevancy")
        rows[row_index]["context_precision"] = _metric_value_from_row(score_row, "context_precision")
        rows[row_index]["context_recall"] = _metric_value_from_row(score_row, "context_recall")
        rows[row_index]["overall_metric_score"] = _overall_metric_score(rows[row_index])
    return None


def _load_ragas_components() -> Tuple[Any, Any, Dict[str, Any], Any, Any]:
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The 'datasets' package is required for accuracy evaluation.") from exc

    try:
        from openai import OpenAI
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
        from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'ragas', 'openai', and 'langchain-openai' packages are required for accuracy evaluation."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY (or OPEN_API_KEY) is required for RAGAS evaluation.")

    client = OpenAI(api_key=api_key)
    ragas_llm = llm_factory(
        os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini"),
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=1024,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*LangchainEmbeddingsWrapper is deprecated.*",
            category=DeprecationWarning,
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            LangchainOpenAIEmbeddings(
                api_key=api_key,
                model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
        )

    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }
    return Dataset, ragas_evaluate, metrics, ragas_llm, ragas_embeddings


def _result_rows(result: Any) -> List[Dict[str, Any]]:
    scores = getattr(result, "scores", None)
    if isinstance(scores, list):
        return [dict(item) if isinstance(item, dict) else {} for item in scores]

    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        return dataframe.to_dict(orient="records")

    if isinstance(result, dict):
        return [dict(result)]

    raise ValueError("Unable to extract row-level scores from the RAGAS result object.")


def _metric_value_from_row(score_row: Dict[str, Any], key: str) -> Optional[float]:
    candidate_keys = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "answer_relevance"],
        "context_precision": ["context_precision"],
        "context_recall": ["context_recall"],
    }.get(key, [key])

    for candidate in candidate_keys:
        if candidate in score_row:
            return _to_optional_float(score_row.get(candidate))
    return None


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _round_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _format_duration_ms(duration_ms: float) -> str:
    total_seconds = max(duration_ms, 0.0) / 1000.0
    minutes, seconds = divmod(total_seconds, 60.0)
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"


def _overall_metric_score(row: Dict[str, Any]) -> Optional[float]:
    return _round_or_none(
        compute_mean(
            [
                row.get("faithfulness"),
                row.get("answer_relevancy"),
                row.get("context_precision"),
                row.get("context_recall"),
            ]
        )
    )


def _print_summary(
    summary: Dict[str, Any],
    *,
    csv_path: Path,
    json_path: Path,
    ragas_error: Optional[str],
    mongo_error: Optional[str],
) -> None:
    print(f"Tests run: {summary['tests_run']}")
    print(
        "Batch total runtime: "
        f"{summary['batch_total_runtime_human']} ({summary['batch_total_runtime_ms']} ms)"
    )
    print(f"Average faithfulness: {summary['average_faithfulness']}")
    print(f"Scored rows for faithfulness: {summary['scored_rows_faithfulness']}/{summary['tests_run']}")
    print(f"Average answer relevancy: {summary['average_answer_relevancy']}")
    print(f"Scored rows for answer relevancy: {summary['scored_rows_answer_relevancy']}/{summary['tests_run']}")
    print(f"Average context precision: {summary['average_context_precision']}")
    print(f"Scored rows for context precision: {summary['scored_rows_context_precision']}/{summary['tests_run']}")
    print(f"Average context recall: {summary['average_context_recall']}")
    print(f"Scored rows for context recall: {summary['scored_rows_context_recall']}/{summary['tests_run']}")
    print(f"Average overall metric score: {summary['average_overall_metric_score']}")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    if ragas_error:
        print(f"RAGAS warning: {ragas_error}")
    if mongo_error:
        print(f"Mongo warning: {mongo_error}")


if __name__ == "__main__":
    raise SystemExit(main())
