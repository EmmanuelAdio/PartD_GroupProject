import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from services.embedding_service import DeterministicEmbeddingService, EmbeddingService
from services.ingestion_service import IngestionService
from services.llm_services import LLMService


def discover_json_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*.json")):
        if path.name.endswith("_chunk_records_preview.json"):
            continue
        yield path


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


def build_embedder(name: str):
    if name == "fake":
        return DeterministicEmbeddingService(dim=64)
    return EmbeddingService()


def build_tagger(name: str, llm_model: str):
    if name == "heuristic":
        return None
    return LLMService(model=llm_model)


def ingest_file(path: Path, service: IngestionService, output_dir: Path | None) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    source_id = path.stem
    count, records = service.ingest_json(
        data=data,
        source_id=source_id,
        title=source_id.replace("_", " ").title(),
    )

    print(f"\n[{path.name}]")
    print(f"  records: {count}")
    if records:
        print(f"  first chunk id: {records[0].chunk_id}")
        print(f"  first chunk domain: {records[0].domain}")
        print(f"  first chunk entities: {records[0].entity_tags[:5]}")
        print(f"  first chunk preview: {records[0].text[:120].replace(chr(10), ' ')}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{path.stem}_chunk_records_preview.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2, ensure_ascii=False)
        print(f"  wrote preview: {out_path}")

    return count, len(records)


def run_mongo_upload_test(
    data_dir: Path,
    files: Sequence[Path],
    embedder_name: str,
    tagger_name: str,
    llm_model: str,
    db_name: str,
    collection_name: str,
    clear_source_before_upload: bool,
) -> None:
    """Integration-style test: ingest JSON files and upsert chunks to MongoDB."""
    from services.mongo_repo import MongoRepo

    if load_dotenv is not None:
        load_dotenv()

    mongo_uri = os.getenv("MONGODB_URI") or read_env_value("MONGODB_URI", data_dir.parent / ".env")
    if not mongo_uri:
        raise ValueError("MONGODB_URI is not set. Add it to your .env before running --test-mongo-upload.")

    repo = MongoRepo(
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
    )
    repo.ping()

    service = IngestionService(
        repo=repo,
        embedder=build_embedder(embedder_name),
        tagger=build_tagger(tagger_name, llm_model),
        version="test-v2",
    )

    total_local_records = 0
    total_mongo_records = 0

    print("\n[MONGO UPLOAD TEST]")
    print(f"  database: {db_name}")
    print(f"  collection: {collection_name}")
    print(f"  embedder: {embedder_name}")
    print(f"  tagger: {tagger_name}")
    print(f"  files to ingest: {len(files)}")

    for path in files:
        source_id = path.stem
        if clear_source_before_upload:
            repo.collection.delete_many({"source_id": source_id})

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        local_count, records = service.ingest_json(
            data=data,
            source_id=source_id,
            title=source_id.replace("_", " ").title(),
        )
        mongo_count = repo.collection.count_documents({"source_id": source_id})

        total_local_records += local_count
        total_mongo_records += mongo_count

        print(f"\n[{path.name}]")
        print(f"  local records built: {local_count}")
        print(f"  records returned: {len(records)}")
        print(f"  records in Mongo by source_id={source_id}: {mongo_count}")

    print("\n[MONGO SUMMARY]")
    print(f"  total local records built: {total_local_records}")
    print(f"  total Mongo records across ingested source_ids: {total_mongo_records}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion against JSON files in data/ using real files.")
    parser.add_argument("--data-dir", default="data", help="Directory containing input JSON files.")
    parser.add_argument("--file", default=None, help="Single JSON file name inside data-dir.")
    parser.add_argument(
        "--embedder",
        choices=["fake", "openai"],
        default="fake",
        help="Embedding backend. 'fake' requires no API key. 'openai' requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--tagger",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Chunk tagging mode. 'llm' uses OPENAI_API_KEY and performs LLM domain/entity tagging.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI chat model used when --tagger llm is selected.",
    )
    parser.add_argument(
        "--write-previews",
        action="store_true",
        help="Write generated chunk records as JSON files under data/chunk_previews/.",
    )
    parser.add_argument(
        "--test-mongo-upload",
        action="store_true",
        help="Ingest selected/all JSON files and upsert chunks to MongoDB for integration testing.",
    )
    parser.add_argument(
        "--mongo-db",
        default="open_day_knowledge",
        help="MongoDB database name (matches your RAG demo default).",
    )
    parser.add_argument(
        "--mongo-collection",
        default="kb_chuncks",
        help="MongoDB collection name for chunk storage.",
    )
    parser.add_argument(
        "--clear-source-before-upload",
        action="store_true",
        help="Delete existing docs for each source_id before upload (clean test runs).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.file:
        target_files = [data_dir / args.file]
        if not target_files[0].exists():
            raise FileNotFoundError(f"JSON file not found: {target_files[0]}")
    else:
        target_files = list(discover_json_files(data_dir))

    if not target_files:
        print(f"No JSON files found in {data_dir}.")
        return

    if args.test_mongo_upload:
        run_mongo_upload_test(
            data_dir=data_dir,
            files=target_files,
            embedder_name=args.embedder,
            tagger_name=args.tagger,
            llm_model=args.llm_model,
            db_name=args.mongo_db,
            collection_name=args.mongo_collection,
            clear_source_before_upload=args.clear_source_before_upload,
        )
        return

    service = IngestionService(
        repo=None,
        embedder=build_embedder(args.embedder),
        tagger=build_tagger(args.tagger, args.llm_model),
        version="test-v2",
    )
    output_dir = data_dir / "chunk_previews" if args.write_previews else None

    total = 0
    for path in target_files:
        count, _ = ingest_file(path, service, output_dir)
        total += count
    print(f"\n[LOCAL SUMMARY] total records built: {total}")


if __name__ == "__main__":
    main()
