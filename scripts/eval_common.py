from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover
    MongoClient = None


def load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"))


def default_query_embedder_backend() -> str:
    return "openai" if has_openai_api_key() else "fake"


def parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    values: List[str] = []
    seen = set()
    for part in raw.split(","):
        value = part.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def select_benchmark_questions(
    questions: Sequence[Dict[str, Any]],
    *,
    category: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    wanted_ids = {str(value).strip() for value in (ids or []) if str(value).strip()}
    out: List[Dict[str, Any]] = []
    for item in questions:
        if category and item.get("category") != category:
            continue
        if wanted_ids and item.get("id") not in wanted_ids:
            continue
        out.append(dict(item))
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out


def ensure_directory(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def write_rows_to_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if pd is not None:
        pd.DataFrame(list(rows)).to_csv(path, index=False)
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_timing_ms(response: Dict[str, Any]) -> Dict[str, float]:
    timing = response.get("timing_ms", {}) if isinstance(response, dict) else {}
    return {
        "processor_time_ms": _coerce_float(timing.get("processor")),
        "retriever_time_ms": _coerce_float(timing.get("retriever")),
        "answerer_time_ms": _coerce_float(timing.get("answerer")),
        "evaluator_time_ms": _coerce_float(timing.get("evaluator")),
        "total_time_ms": _coerce_float(timing.get("total")),
    }


def extract_context_texts(response: Dict[str, Any]) -> List[str]:
    retrieval_run = response.get("retrieval_run", {}) if isinstance(response, dict) else {}
    evidence_items = retrieval_run.get("evidence", [])
    contexts: List[str] = []
    for item in evidence_items if isinstance(evidence_items, list) else []:
        text = extract_text_from_payload(item)
        if not text:
            continue
        contexts.append(text)
    return contexts


def extract_text_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return _clean_text(payload)
    if isinstance(payload, dict):
        direct_value = payload.get("text") or payload.get("page_content") or payload.get("content")
        if isinstance(direct_value, str) and _clean_text(direct_value):
            return _clean_text(direct_value)

        for nested_key in ("item", "evidence", "document", "chunk", "payload", "source"):
            nested_value = payload.get(nested_key)
            text = extract_text_from_payload(nested_value)
            if text:
                return text
    if isinstance(payload, list):
        for item in payload:
            text = extract_text_from_payload(item)
            if text:
                return text
    return ""


def compute_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / float(len(nums))


def save_documents_to_mongo(
    *,
    documents: Sequence[Dict[str, Any]],
    db_name: str,
    collection_name: str,
) -> int:
    if not documents:
        return 0
    if MongoClient is None:
        raise ImportError("pymongo is not installed. Install it to use --save-mongo.")

    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI is not set, so MongoDB result saving is unavailable.")

    client = MongoClient(mongo_uri)
    try:
        collection = client[db_name][collection_name]
        result = collection.insert_many(list(documents))
        return len(result.inserted_ids)
    finally:
        client.close()


def _clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _coerce_float(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
