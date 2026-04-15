from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis_helpers import (
    available_locust_prefixes,
    ensure_figures_dir,
    load_locust_comparison,
    load_locust_result_set,
    round_numeric_dataframe,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready charts from Locust CSV outputs.")
    parser.add_argument("--results-dir", default="results", help="Directory containing Locust CSV outputs.")
    parser.add_argument("--output-dir", default="results/figures", help="Directory to save generated figures.")
    parser.add_argument("--prefixes", default=None, help="Optional comma-separated Locust run prefixes, e.g. locust_5_users,locust_10_users")
    parser.add_argument("--user-counts", default=None, help="Optional comma-separated user counts, e.g. 5,10,20")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = ensure_figures_dir(args.output_dir)

    prefixes = _parse_csv_list(args.prefixes)
    user_counts = [int(value) for value in _parse_csv_list(args.user_counts)]
    comparison = load_locust_comparison(
        results_dir=results_dir,
        prefixes=prefixes or None,
        user_counts=user_counts or None,
    )
    if comparison.empty:
        raise SystemExit("No Locust stats files were found for plotting.")

    print(f"Detected Locust runs: {', '.join(comparison['prefix'].tolist())}")
    print("")
    print("Load summary:")
    print(round_numeric_dataframe(comparison).to_string(index=False))

    saved_paths: List[Path] = []
    saved_paths.extend(plot_summary_charts(comparison, output_dir=output_dir))
    saved_paths.extend(plot_history_charts(comparison["prefix"].tolist(), results_dir=results_dir, output_dir=output_dir))
    saved_paths.extend(plot_failure_analysis(comparison["prefix"].tolist(), results_dir=results_dir, output_dir=output_dir))

    print("")
    print("Saved figures:")
    for path in saved_paths:
        print(f"- {path}")
    return 0


def plot_summary_charts(comparison: pd.DataFrame, *, output_dir: Path) -> List[Path]:
    saved_paths: List[Path] = []
    chart_specs = [
        ("average_response_time", "Average response time by user count", "Milliseconds", "locust_average_response_time_by_users.png"),
        ("median_response_time", "Median response time by user count", "Milliseconds", "locust_median_response_time_by_users.png"),
        ("requests_per_second", "Requests per second by user count", "Requests/s", "locust_requests_per_second_by_users.png"),
        ("failure_count", "Failure count by user count", "Failures", "locust_failure_count_by_users.png"),
    ]

    plot_df = comparison.sort_values("user_count").set_index("user_count")
    for column, title, ylabel, filename in chart_specs:
        if column not in plot_df.columns:
            print(f"[WARN] summary chart skipped because '{column}' is missing.")
            continue
        values = pd.to_numeric(plot_df[column], errors="coerce").dropna()
        if values.empty:
            print(f"[WARN] summary chart skipped because '{column}' had no numeric values.")
            continue
        fig, ax = plt.subplots()
        values.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Users")
        saved_paths.append(save_figure(fig, filename, output_dir=output_dir))

    if "failure_rate" in plot_df.columns:
        values = pd.to_numeric(plot_df["failure_rate"], errors="coerce").dropna()
        if not values.empty:
            fig, ax = plt.subplots()
            values.plot(kind="bar", ax=ax)
            ax.set_title("Failure rate by user count")
            ax.set_ylabel("Failure rate")
            ax.set_xlabel("Users")
            saved_paths.append(save_figure(fig, "locust_failure_rate_by_users.png", output_dir=output_dir))

    return saved_paths


def plot_history_charts(prefixes: List[str], *, results_dir: Path, output_dir: Path) -> List[Path]:
    saved_paths: List[Path] = []
    history_specs = [
        ("Total Average Response Time", "Average response time over time", "Milliseconds", "locust_history_average_response_time.png"),
        ("Requests/s", "Requests per second over time", "Requests/s", "locust_history_requests_per_second.png"),
        ("Failures/s", "Failures per second over time", "Failures/s", "locust_history_failures_per_second.png"),
    ]

    for column, title, ylabel, filename in history_specs:
        fig, ax = plt.subplots()
        plotted = False
        for prefix in prefixes:
            result_set = load_locust_result_set(prefix, results_dir=results_dir)
            history_df = result_set.get("stats_history")
            if history_df is None or history_df.empty or column not in history_df.columns:
                continue

            working = history_df.copy()
            if "Name" in working.columns:
                working = working[working["Name"].astype(str).str.strip() == "Aggregated"]
            if working.empty:
                continue

            working[column] = pd.to_numeric(working[column], errors="coerce")
            if "Timestamp" in working.columns:
                x_values = pd.to_datetime(pd.to_numeric(working["Timestamp"], errors="coerce"), unit="s", errors="coerce")
            else:
                x_values = pd.RangeIndex(start=0, stop=len(working), step=1)

            series = working[column].dropna()
            if series.empty:
                continue

            x_aligned = x_values[working[column].notna()]
            ax.plot(x_aligned, series, label=prefix)
            plotted = True

        if not plotted:
            plt.close(fig)
            print(f"[WARN] history chart skipped because '{column}' was unavailable across runs.")
            continue

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        saved_paths.append(save_figure(fig, filename, output_dir=output_dir))
    return saved_paths


def plot_failure_analysis(prefixes: List[str], *, results_dir: Path, output_dir: Path) -> List[Path]:
    saved_paths: List[Path] = []
    failure_rows = []
    exception_frames = []

    for prefix in prefixes:
        result_set = load_locust_result_set(prefix, results_dir=results_dir)
        failures_df = result_set.get("failures")
        exceptions_df = result_set.get("exceptions")

        failure_total = 0.0
        if failures_df is not None and not failures_df.empty and "Occurrences" in failures_df.columns:
            failure_total = pd.to_numeric(failures_df["Occurrences"], errors="coerce").fillna(0).sum()
        failure_rows.append(
            {
                "prefix": prefix,
                "user_count": result_set.get("user_count"),
                "failure_total": failure_total,
            }
        )

        if exceptions_df is not None and not exceptions_df.empty:
            frame = exceptions_df.copy()
            frame["prefix"] = prefix
            frame["user_count"] = result_set.get("user_count")
            exception_frames.append(frame)

    failure_df = pd.DataFrame(failure_rows).sort_values("user_count")
    if not failure_df.empty:
        fig, ax = plt.subplots()
        failure_df.set_index("user_count")["failure_total"].plot(kind="bar", ax=ax)
        ax.set_title("Locust failures by user count")
        ax.set_ylabel("Failures")
        ax.set_xlabel("Users")
        saved_paths.append(save_figure(fig, "locust_failures_by_users.png", output_dir=output_dir))

    if exception_frames:
        combined = pd.concat(exception_frames, ignore_index=True)
        grouped = (
            combined.groupby(["user_count", "Message"], dropna=False)
            .agg(total_count=("Count", "sum"))
            .reset_index()
            .sort_values(["user_count", "total_count"], ascending=[True, False])
        )
        if not grouped.empty:
            fig, ax = plt.subplots()
            top_grouped = grouped.groupby("user_count", dropna=False)["total_count"].sum()
            top_grouped.plot(kind="bar", ax=ax)
            ax.set_title("Exceptions by user count")
            ax.set_ylabel("Exception count")
            ax.set_xlabel("Users")
            saved_paths.append(save_figure(fig, "locust_exceptions_by_users.png", output_dir=output_dir))
    else:
        print("[INFO] No Locust exceptions were recorded.")

    return saved_paths


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
