from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from services.embedding_service import DeterministicEmbeddingService, EmbeddingService
from services.index_manager import AtlasIndexManager
from services.mongo_repo import MongoRepo


def read_env_value(key: str, env_path: Path) -> str | None:
    if not env_path.exists():
        return None

    with env_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            if lhs.strip() == key:
                return rhs.strip().strip('"').strip("'")
    return None


def build_embedder(name: str, model: str):
    if name == "fake":
        return DeterministicEmbeddingService(dim=64)
    return EmbeddingService(model=model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integration test for AtlasIndexManager.")
    parser.add_argument("--mongo-db", default="open_day_knowledge")
    parser.add_argument("--mongo-collection", default="kb_chuncks")
    parser.add_argument("--embedder", choices=["fake", "openai"], default="fake")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--vector-index", default="kb_vector_index")
    parser.add_argument("--search-index", default="kb_text_index")
    parser.add_argument("--ensure-indexes", action="store_true", help="Create/reconcile Atlas indexes.")
    parser.add_argument(
        "--no-wait-for-ready",
        action="store_true",
        help="Do not wait for READY status after create/reconcile.",
    )
    parser.add_argument("--show-search-indexes", action="store_true", help="Print raw search index metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv()

    repo_root = Path(__file__).resolve().parent
    mongo_uri = os.getenv("MONGODB_URI") or read_env_value("MONGODB_URI", repo_root / ".env")
    if not mongo_uri:
        raise ValueError("MONGODB_URI is not set. Add it to .env before running index manager tests.")

    repo = MongoRepo(
        mongo_uri=mongo_uri,
        db_name=args.mongo_db,
        collection_name=args.mongo_collection,
    )
    repo.ping()

    embedder = build_embedder(args.embedder, args.embedding_model)
    manager = AtlasIndexManager(
        collection=repo.collection,
        embedder=embedder,
        vector_index_name=args.vector_index,
        atlas_search_index_name=args.search_index,
    )

    print("[INDEX MANAGER TEST]")
    print(f"  database: {args.mongo_db}")
    print(f"  collection: {args.mongo_collection}")
    print(f"  total docs: {repo.collection.count_documents({})}")

    print("\n[CHECK]")
    check_report = manager.check_index_health()
    print(check_report.summary())

    if args.ensure_indexes:
        print("\n[ENSURE]")
        ensure_report = manager.ensure_indexes(wait_for_ready=not args.no_wait_for_ready)
        print(ensure_report.summary())

    if args.show_search_indexes:
        print("\n[RAW SEARCH INDEXES]")
        try:
            indexes = manager._list_search_indexes()  # integration test helper output
            print(json.dumps(indexes, indent=2, default=str))
        except Exception as exc:
            print(f"Could not list search indexes: {exc}")


if __name__ == "__main__":
    main()
