from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis_helpers import (
    build_figure_filename,
    compute_p95,
    compute_signature_hash,
    extract_timestamp_from_filename,
    find_latest_csv,
    parse_locust_user_count,
    save_figure,
    summarize_locust_stats_file,
)


def test_extract_timestamp_from_filename_returns_datetime() -> None:
    parsed = extract_timestamp_from_filename("accuracy_eval_20260414_162339.csv")

    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 4
    assert parsed.day == 14
    assert parsed.hour == 16
    assert parsed.minute == 23
    assert parsed.second == 39


def test_parse_locust_user_count_reads_filename() -> None:
    assert parse_locust_user_count("locust_20_users_stats.csv") == 20
    assert parse_locust_user_count("not_a_locust_file.csv") is None


def test_compute_p95_returns_expected_percentile() -> None:
    value = compute_p95([1, 2, 3, 4, 5])

    assert value is not None
    assert round(value, 1) == 4.8


def test_find_latest_csv_uses_timestamp_in_filename(tmp_path: Path) -> None:
    older = tmp_path / "accuracy_eval_20260414_120000.csv"
    newer = tmp_path / "accuracy_eval_20260414_130000.csv"
    older.write_text("value\n1\n", encoding="utf-8")
    newer.write_text("value\n2\n", encoding="utf-8")

    latest = find_latest_csv("accuracy_eval_", results_dir=tmp_path)

    assert latest == newer


def test_summarize_locust_stats_file_prefers_aggregated_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "locust_5_users_stats.csv"
    dataframe = pd.DataFrame(
        [
            {
                "Type": "POST",
                "Name": "POST /query",
                "Request Count": 10,
                "Failure Count": 2,
                "Median Response Time": 1000,
                "Average Response Time": 1200,
                "Requests/s": 1.0,
                "Failures/s": 0.2,
            },
            {
                "Type": "",
                "Name": "Aggregated",
                "Request Count": 20,
                "Failure Count": 3,
                "Median Response Time": 2000,
                "Average Response Time": 2200,
                "Requests/s": 2.0,
                "Failures/s": 0.3,
            },
        ]
    )
    dataframe.to_csv(csv_path, index=False)

    summary = summarize_locust_stats_file(csv_path)

    assert summary["user_count"] == 5
    assert summary["request_count"] == 20.0
    assert summary["failure_count"] == 3.0
    assert summary["average_response_time"] == 2200.0


def test_build_figure_filename_is_deterministic() -> None:
    filename = build_figure_filename("accuracy_eval_20260414_162339", "accuracy overview")

    assert filename == "accuracy_eval_20260414_162339_accuracy_overview.png"


def test_compute_signature_hash_supports_dataframes() -> None:
    dataframe = pd.DataFrame(
        [
            {"metric": "faithfulness", "average": 0.8, "created_at": datetime(2026, 4, 15, 12, 0, 0)},
            {"metric": "answer_relevancy", "average": 0.9, "created_at": datetime(2026, 4, 15, 12, 5, 0)},
        ]
    )

    first = compute_signature_hash({"chart": "overall", "data": dataframe})
    second = compute_signature_hash({"chart": "overall", "data": dataframe.copy()})

    assert first == second


def test_save_figure_skips_rewrite_when_signature_is_unchanged(tmp_path: Path) -> None:
    figure_path = tmp_path / "figure.png"
    signature = {"chart": "demo", "values": [1, 2, 3]}

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    save_figure(fig, figure_path.name, output_dir=tmp_path, signature_parts=signature)
    first_mtime = figure_path.stat().st_mtime_ns

    time.sleep(0.01)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    save_figure(fig, figure_path.name, output_dir=tmp_path, signature_parts=signature)
    second_mtime = figure_path.stat().st_mtime_ns

    assert first_mtime == second_mtime


def test_save_figure_rewrites_when_signature_changes(tmp_path: Path) -> None:
    figure_path = tmp_path / "figure.png"

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    save_figure(fig, figure_path.name, output_dir=tmp_path, signature_parts={"chart": "demo", "values": [1, 2, 3]})
    first_mtime = figure_path.stat().st_mtime_ns

    time.sleep(0.01)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])
    save_figure(fig, figure_path.name, output_dir=tmp_path, signature_parts={"chart": "demo", "values": [1, 2, 3, 4]})
    second_mtime = figure_path.stat().st_mtime_ns

    assert second_mtime >= first_mtime
    assert second_mtime != first_mtime
