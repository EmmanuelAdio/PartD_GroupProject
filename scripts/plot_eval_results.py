from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_helpers import (
    available_numeric_columns,
    ensure_figures_dir,
    find_latest_csv,
    grouped_averages_by_category,
    load_accuracy_run_comparison,
    load_csv_safe,
    round_numeric_dataframe,
    save_figure,
    warn_missing_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready charts from evaluation CSV outputs.")
    parser.add_argument("--accuracy-csv", default=None, help="Path to an accuracy evaluation CSV.")
    parser.add_argument("--evaluator-csv", default=None, help="Path to an evaluator evaluation CSV.")
    parser.add_argument("--results-dir", default="results", help="Directory containing evaluation outputs.")
    parser.add_argument("--output-dir", default="results/figures", help="Directory to save generated figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = ensure_figures_dir(args.output_dir)

    accuracy_path = Path(args.accuracy_csv) if args.accuracy_csv else find_latest_csv("accuracy_eval_", results_dir=results_dir)
    evaluator_path = Path(args.evaluator_csv) if args.evaluator_csv else find_latest_csv("evaluator_eval_", results_dir=results_dir)

    if accuracy_path is None:
        raise SystemExit("No accuracy evaluation CSV was found.")
    if evaluator_path is None:
        raise SystemExit("No evaluator evaluation CSV was found.")

    accuracy_df = load_csv_safe(accuracy_path)
    evaluator_df = load_csv_safe(evaluator_path)

    print(f"Using accuracy CSV: {accuracy_path}")
    print(f"Using evaluator CSV: {evaluator_path}")
    print(f"Saving figures to: {output_dir}")

    saved_paths: List[Path] = []
    saved_paths.extend(plot_accuracy_sections(accuracy_df, output_dir=output_dir, stem=accuracy_path.stem))
    saved_paths.extend(plot_timing_sections(accuracy_df, output_dir=output_dir, stem=accuracy_path.stem))
    saved_paths.extend(plot_evaluator_sections(evaluator_df, output_dir=output_dir, stem=evaluator_path.stem))
    saved_paths.extend(plot_accuracy_run_comparison(results_dir=results_dir, output_dir=output_dir))

    print("")
    print("Accuracy summary:")
    print(round_numeric_dataframe(build_accuracy_summary_table(accuracy_df)).to_string(index=False))
    print("")
    print("Evaluator summary:")
    print(round_numeric_dataframe(build_evaluator_summary_table(evaluator_df)).to_string(index=False))
    print("")
    print("Saved figures:")
    for path in saved_paths:
        print(f"- {path}")
    return 0


def build_accuracy_summary_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall_metric_score"]
    available = available_numeric_columns(dataframe, metric_columns)
    rows = []
    for column in available:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        rows.append({"metric": column, "average": series.mean()})
    return pd.DataFrame(rows)


def build_evaluator_summary_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "evaluator_decision" not in dataframe.columns:
        print("[WARN] evaluator summary table skipped because 'evaluator_decision' is missing.")
        return pd.DataFrame()
    counts = dataframe["evaluator_decision"].fillna("unknown").astype(str).value_counts().reset_index()
    counts.columns = ["decision", "count"]
    counts["percentage"] = counts["count"] / max(len(dataframe), 1) * 100.0
    return counts


def plot_accuracy_sections(dataframe: pd.DataFrame, *, output_dir: Path, stem: str) -> List[Path]:
    saved_paths: List[Path] = []
    metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = available_numeric_columns(dataframe, metric_columns)
    if not available:
        print("[WARN] accuracy plots skipped because no RAGAS metric columns were present.")
        return saved_paths

    overall_means = pd.Series(
        {column: pd.to_numeric(dataframe[column], errors="coerce").mean() for column in available}
    )
    fig, ax = plt.subplots()
    overall_means.plot(kind="bar", ax=ax)
    ax.set_title("Overall average RAGAS metrics")
    ax.set_ylabel("Average score")
    saved_paths.append(save_figure(fig, f"{stem}_overall_ragas_metrics.png", output_dir=output_dir))

    if "category" in dataframe.columns:
        by_category = grouped_averages_by_category(dataframe, available)
        if not by_category.empty:
            fig, ax = plt.subplots()
            by_category = by_category.set_index("category")
            by_category.plot(kind="bar", ax=ax)
            ax.set_title("Average RAGAS metrics by category")
            ax.set_ylabel("Average score")
            saved_paths.append(save_figure(fig, f"{stem}_ragas_metrics_by_category.png", output_dir=output_dir))

    for column in ("faithfulness", "answer_relevancy"):
        if column not in dataframe.columns:
            print(f"[WARN] histogram skipped because '{column}' is missing.")
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            print(f"[WARN] histogram skipped because '{column}' had no numeric values.")
            continue
        fig, ax = plt.subplots()
        values.plot(kind="hist", bins=10, ax=ax)
        ax.set_title(f"{column.replace('_', ' ').title()} distribution")
        ax.set_xlabel(column)
        saved_paths.append(save_figure(fig, f"{stem}_{column}_histogram.png", output_dir=output_dir))

    return saved_paths


def plot_timing_sections(dataframe: pd.DataFrame, *, output_dir: Path, stem: str) -> List[Path]:
    saved_paths: List[Path] = []
    timing_columns = [
        "processor_time_ms",
        "retriever_time_ms",
        "answerer_time_ms",
        "evaluator_time_ms",
        "total_time_ms",
    ]
    available = available_numeric_columns(dataframe, timing_columns)
    if not available:
        print("[WARN] timing plots skipped because no timing columns were present.")
        return saved_paths

    mean_series = pd.Series(
        {column: pd.to_numeric(dataframe[column], errors="coerce").mean() for column in available}
    )
    fig, ax = plt.subplots()
    mean_series.plot(kind="bar", ax=ax)
    ax.set_title("Average stage times")
    ax.set_ylabel("Milliseconds")
    saved_paths.append(save_figure(fig, f"{stem}_average_stage_times.png", output_dir=output_dir))

    if "total_time_ms" in dataframe.columns:
        total_values = pd.to_numeric(dataframe["total_time_ms"], errors="coerce").dropna()
        if not total_values.empty:
            fig, ax = plt.subplots()
            total_values.plot(kind="hist", bins=10, ax=ax)
            ax.set_title("Total response time distribution")
            ax.set_xlabel("total_time_ms")
            saved_paths.append(save_figure(fig, f"{stem}_total_time_histogram.png", output_dir=output_dir))

            fig, ax = plt.subplots()
            ax.boxplot(total_values)
            ax.set_title("Total response time boxplot")
            ax.set_ylabel("Milliseconds")
            saved_paths.append(save_figure(fig, f"{stem}_total_time_boxplot.png", output_dir=output_dir))
    return saved_paths


def plot_evaluator_sections(dataframe: pd.DataFrame, *, output_dir: Path, stem: str) -> List[Path]:
    saved_paths: List[Path] = []
    if "evaluator_decision" not in dataframe.columns:
        print("[WARN] evaluator plots skipped because 'evaluator_decision' is missing.")
        return saved_paths

    decision_counts = dataframe["evaluator_decision"].fillna("unknown").astype(str).value_counts().sort_index()
    fig, ax = plt.subplots()
    decision_counts.plot(kind="bar", ax=ax)
    ax.set_title("Evaluator decision counts")
    ax.set_ylabel("Count")
    saved_paths.append(save_figure(fig, f"{stem}_decision_counts.png", output_dir=output_dir))

    decision_percentages = decision_counts / max(decision_counts.sum(), 1) * 100.0
    fig, ax = plt.subplots()
    decision_percentages.plot(kind="bar", ax=ax)
    ax.set_title("Evaluator decision percentages")
    ax.set_ylabel("Percentage")
    saved_paths.append(save_figure(fig, f"{stem}_decision_percentages.png", output_dir=output_dir))

    if "retry_count" in dataframe.columns:
        retry_values = pd.to_numeric(dataframe["retry_count"], errors="coerce").dropna()
        if not retry_values.empty:
            retry_counts = retry_values.astype(int).value_counts().sort_index()
            fig, ax = plt.subplots()
            retry_counts.plot(kind="bar", ax=ax)
            ax.set_title("Retry count distribution")
            ax.set_xlabel("retry_count")
            ax.set_ylabel("Questions")
            saved_paths.append(save_figure(fig, f"{stem}_retry_distribution.png", output_dir=output_dir))

    return saved_paths


def plot_accuracy_run_comparison(*, results_dir: Path, output_dir: Path) -> List[Path]:
    comparison = load_accuracy_run_comparison(results_dir=results_dir)
    if comparison.empty or len(comparison) < 2:
        print("[INFO] Multiple-run comparison skipped because fewer than two accuracy runs were found.")
        return []

    saved_paths: List[Path] = []
    metric_columns = [
        "avg_faithfulness",
        "avg_answer_relevancy",
        "avg_context_precision",
        "avg_context_recall",
    ]
    available = [column for column in metric_columns if column in comparison.columns]
    if available:
        plot_df = comparison.set_index("filename")[available]
        fig, ax = plt.subplots()
        plot_df.plot(kind="bar", ax=ax)
        ax.set_title("Accuracy run comparison")
        ax.set_ylabel("Average score")
        saved_paths.append(save_figure(fig, "accuracy_run_comparison_metrics.png", output_dir=output_dir))

    if "avg_total_time_ms" in comparison.columns:
        fig, ax = plt.subplots()
        comparison.set_index("filename")["avg_total_time_ms"].plot(kind="bar", ax=ax)
        ax.set_title("Average total response time by run")
        ax.set_ylabel("Milliseconds")
        saved_paths.append(save_figure(fig, "accuracy_run_comparison_total_time.png", output_dir=output_dir))
    return saved_paths


if __name__ == "__main__":
    raise SystemExit(main())
