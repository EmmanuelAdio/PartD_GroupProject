from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{6})")
LOCUST_USER_COUNT_PATTERN = re.compile(r"locust_(\d+)_users")
NON_FILENAME_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
FIGURE_MANIFEST_FILENAME = "_figure_manifest.json"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_figures_dir(path: str | Path = "results/figures") -> Path:
    return ensure_directory(path)


def build_figure_filename(*parts: Any, extension: str = ".png") -> str:
    cleaned_parts = [_slugify_filename_part(part) for part in parts if part not in (None, "")]
    base_name = "_".join(part for part in cleaned_parts if part)
    if not base_name:
        raise ValueError("At least one non-empty filename part is required.")
    if not extension.startswith("."):
        extension = f".{extension}"
    return f"{base_name}{extension}"


def extract_timestamp_from_filename(filename: str | Path) -> Optional[datetime]:
    match = TIMESTAMP_PATTERN.search(Path(filename).name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def parse_locust_user_count(filename: str | Path) -> Optional[int]:
    match = LOCUST_USER_COUNT_PATTERN.search(Path(filename).name)
    if not match:
        return None
    return int(match.group(1))


def find_matching_files(
    pattern: str,
    results_dir: str | Path = "results",
) -> List[Path]:
    base_dir = Path(results_dir)
    files = [path for path in base_dir.glob(pattern) if path.is_file()]
    return sorted(files, key=_sort_key_for_results_file)


def find_files_by_prefix(
    prefix: str,
    *,
    extension: str = ".csv",
    results_dir: str | Path = "results",
) -> List[Path]:
    return find_matching_files(f"{prefix}*{extension}", results_dir=results_dir)


def find_latest_csv(
    prefix: str,
    *,
    results_dir: str | Path = "results",
) -> Optional[Path]:
    files = find_files_by_prefix(prefix, extension=".csv", results_dir=results_dir)
    return files[-1] if files else None


def load_csv_safe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")
    try:
        return pd.read_csv(csv_path, **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {csv_path}") from exc


def warn_missing_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str = "",
) -> List[str]:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        prefix = f"[WARN] {context}: " if context else "[WARN] "
        print(f"{prefix}missing columns: {', '.join(missing)}")
    return missing


def available_numeric_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> List[str]:
    return [column for column in columns if column in dataframe.columns]


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_p95(values: Iterable[Any]) -> Optional[float]:
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.quantile(0.95))


def grouped_averages_by_category(
    dataframe: pd.DataFrame,
    metric_columns: Sequence[str],
    *,
    category_column: str = "category",
) -> pd.DataFrame:
    if category_column not in dataframe.columns:
        print(f"[WARN] grouped averages skipped because '{category_column}' is missing.")
        return pd.DataFrame()

    numeric_columns = available_numeric_columns(dataframe, metric_columns)
    if not numeric_columns:
        print("[WARN] grouped averages skipped because no requested metric columns were present.")
        return pd.DataFrame()

    working = dataframe[[category_column, *numeric_columns]].copy()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    return working.groupby(category_column, dropna=False)[numeric_columns].mean().reset_index()


def round_numeric_dataframe(dataframe: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    rounded = dataframe.copy()
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
    return rounded


def save_figure(
    figure: Any,
    filename: str,
    *,
    output_dir: str | Path = "results/figures",
    dpi: int = 150,
    signature_parts: Any = None,
    close: bool = True,
) -> Path:
    figures_dir = ensure_figures_dir(output_dir)
    target_path = figures_dir / filename
    if signature_parts is not None:
        manifest = _load_figure_manifest(figures_dir)
        signature = compute_signature_hash(signature_parts)
        if manifest.get(filename) == signature and target_path.exists():
            if close:
                _close_figure(figure)
            return target_path
    else:
        manifest = None
        signature = None

    figure.savefig(target_path, bbox_inches="tight", dpi=dpi)
    if manifest is not None and signature is not None:
        manifest[filename] = signature
        _write_figure_manifest(figures_dir, manifest)
    if close:
        _close_figure(figure)
    return target_path


def save_and_display_figure(
    figure: Any,
    filename: str,
    *,
    output_dir: str | Path = "results/figures",
    dpi: int = 150,
    signature_parts: Any = None,
) -> Path:
    target_path = save_figure(
        figure,
        filename,
        output_dir=output_dir,
        dpi=dpi,
        signature_parts=signature_parts,
        close=False,
    )
    _display_figure_inline(figure, target_path=target_path)
    _close_figure(figure)
    return target_path


def load_multiple_matching_csvs(
    pattern: str,
    *,
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    files = find_matching_files(pattern, results_dir=results_dir)
    frames: List[pd.DataFrame] = []
    for file_path in files:
        frame = load_csv_safe(file_path)
        frame = frame.copy()
        frame["source_file"] = file_path.name
        timestamp = extract_timestamp_from_filename(file_path.name)
        frame["source_timestamp"] = timestamp
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_accuracy_run_comparison(results_dir: str | Path = "results") -> pd.DataFrame:
    files = find_matching_files("accuracy_eval_*.csv", results_dir=results_dir)
    rows: List[Dict[str, Any]] = []
    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "total_time_ms",
    ]

    for file_path in files:
        dataframe = load_csv_safe(file_path)
        row: Dict[str, Any] = {
            "filename": file_path.name,
            "timestamp": extract_timestamp_from_filename(file_path.name),
        }
        for column in metric_columns:
            if column not in dataframe.columns:
                row[f"avg_{column}"] = None
                continue
            row[f"avg_{column}"] = pd.to_numeric(dataframe[column], errors="coerce").mean()
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    comparison = pd.DataFrame(rows)
    if "timestamp" in comparison.columns:
        comparison = comparison.sort_values(["timestamp", "filename"], na_position="last").reset_index(drop=True)
    return comparison


def load_locust_result_set(
    prefix: str,
    *,
    results_dir: str | Path = "results",
) -> Dict[str, Any]:
    base_dir = Path(results_dir)
    base_prefix = prefix
    paths = {
        "stats": base_dir / f"{base_prefix}_stats.csv",
        "stats_history": base_dir / f"{base_prefix}_stats_history.csv",
        "failures": base_dir / f"{base_prefix}_failures.csv",
        "exceptions": base_dir / f"{base_prefix}_exceptions.csv",
    }

    frames: Dict[str, Any] = {
        "prefix": base_prefix,
        "user_count": parse_locust_user_count(base_prefix),
        "paths": paths,
    }
    for key, path in paths.items():
        if path.exists():
            frames[key] = load_csv_safe(path)
        else:
            frames[key] = None
    return frames


def summarize_locust_stats_file(path: str | Path) -> Dict[str, Any]:
    stats_path = Path(path)
    dataframe = load_csv_safe(stats_path)
    row = get_locust_aggregated_row(dataframe)
    prefix = stats_path.name.replace("_stats.csv", "")

    request_count = _coerce_float(row.get("Request Count"))
    failure_count = _coerce_float(row.get("Failure Count"))
    summary = {
        "filename": stats_path.name,
        "prefix": prefix,
        "user_count": parse_locust_user_count(stats_path.name),
        "request_count": request_count,
        "failure_count": failure_count,
        "average_response_time": _coerce_float(row.get("Average Response Time")),
        "median_response_time": _coerce_float(row.get("Median Response Time")),
        "requests_per_second": _coerce_float(row.get("Requests/s")),
        "failures_per_second": _coerce_float(row.get("Failures/s")),
    }
    if request_count is None or request_count == 0 or failure_count is None:
        summary["failure_rate"] = None
    else:
        summary["failure_rate"] = float(failure_count) / float(request_count)
    return summary


def load_locust_comparison(
    results_dir: str | Path = "results",
    *,
    prefixes: Optional[Sequence[str]] = None,
    user_counts: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    stats_files = find_matching_files("locust_*_stats.csv", results_dir=results_dir)
    rows = [summarize_locust_stats_file(path) for path in stats_files]
    if not rows:
        return pd.DataFrame()

    comparison = pd.DataFrame(rows)
    if prefixes:
        wanted_prefixes = {str(prefix) for prefix in prefixes}
        comparison = comparison[comparison["prefix"].isin(wanted_prefixes)]
    if user_counts:
        wanted_counts = {int(value) for value in user_counts}
        comparison = comparison[comparison["user_count"].isin(wanted_counts)]

    if comparison.empty:
        return comparison
    return comparison.sort_values(["user_count", "prefix"], na_position="last").reset_index(drop=True)


def get_locust_aggregated_row(dataframe: pd.DataFrame) -> pd.Series:
    if dataframe.empty:
        return pd.Series(dtype="object")

    if "Name" in dataframe.columns:
        aggregated = dataframe[dataframe["Name"].astype(str).str.strip() == "Aggregated"]
        if not aggregated.empty:
            return aggregated.iloc[0]

    if len(dataframe) == 1:
        return dataframe.iloc[0]

    return dataframe.iloc[-1]


def available_locust_prefixes(results_dir: str | Path = "results") -> List[str]:
    stats_files = find_matching_files("locust_*_stats.csv", results_dir=results_dir)
    return [path.name.replace("_stats.csv", "") for path in stats_files]


def summarize_exception_table(dataframe: Optional[pd.DataFrame], *, label: str) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    summary = dataframe.copy()
    summary["load_label"] = label
    return summary


def _sort_key_for_results_file(path: Path) -> tuple:
    timestamp = extract_timestamp_from_filename(path.name)
    if timestamp is not None:
        return (timestamp, path.name)
    return (datetime.fromtimestamp(path.stat().st_mtime), path.name)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_signature_hash(value: Any) -> str:
    serialized = json.dumps(_normalize_for_signature(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_for_signature(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "index": [_normalize_for_signature(item) for item in value.index.tolist()],
            "records": [
                {str(column): _normalize_for_signature(cell) for column, cell in row.items()}
                for row in value.to_dict(orient="records")
            ],
        }
    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "name": _normalize_for_signature(value.name),
            "index": [_normalize_for_signature(item) for item in value.index.tolist()],
            "values": [_normalize_for_signature(item) for item in value.tolist()],
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize_for_signature(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_signature(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _normalize_for_signature(value.item())
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def _load_figure_manifest(figures_dir: Path) -> Dict[str, str]:
    manifest_path = figures_dir / FIGURE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _write_figure_manifest(figures_dir: Path, manifest: Dict[str, str]) -> None:
    manifest_path = figures_dir / FIGURE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _display_figure_inline(figure: Any, *, target_path: Optional[Path] = None) -> None:
    try:
        from IPython.display import Image, display

        if target_path is not None and target_path.exists():
            display(Image(filename=str(target_path)))
        else:
            display(figure)
    except Exception:
        pass


def _close_figure(figure: Any) -> None:
    figure.clf()
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)
    except Exception:
        pass


def _slugify_filename_part(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    text = NON_FILENAME_CHARS_PATTERN.sub("_", text)
    return text.strip("._")
