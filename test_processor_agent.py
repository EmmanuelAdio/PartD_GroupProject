from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from agents.processor_agent import ProcessorAgent
from schemas.models import RetrievalQuery
from services.embedding_service import DeterministicEmbeddingService, EmbeddingService
from services.mongo_repo import MongoRepo
from services.retriever_service import RetrieverService


def read_env_value(key: str, env_path: Path) -> str | None:
    """Read a single key from a .env-like file."""
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
    """Build query embedder for RetrieverService."""
    if name == "fake":
        return DeterministicEmbeddingService(dim=64)
    return EmbeddingService(model=model)


def truncate_text(text: str, max_chars: int) -> str:
    """Shorten retrieval evidence text for console readability."""
    value = " ".join((text or "").split())
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}..."


def print_retrieval_query_plan(plan: RetrievalQuery) -> None:
    """Print the processor output plan in JSON format."""
    print("\n[PROCESSOR PLAN]")
    print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))


def print_retrieval_results(
    *,
    plan: RetrievalQuery,
    evidence_items: List[Any],
    diagnostics: Dict[str, Any],
    show_text_chars: int,
) -> None:
    """Print retrieval diagnostics and ranked evidence."""
    fallback_used = bool(
        diagnostics.get("vector_fallback_used") or diagnostics.get("text_fallback_used")
    )

    print("\n[RETRIEVAL RUN]")
    print(f"  query_text: {plan.query_text}")
    print(f"  top_k: {plan.top_k}")
    print(
        "  fallback_used: "
        f"{'YES' if fallback_used else 'NO'} "
        f"(vector_mode={diagnostics.get('vector_mode')}, text_mode={diagnostics.get('text_mode')})"
    )
    print(f"  returned evidence: {len(evidence_items)}")

    for rank, item in enumerate(evidence_items, start=1):
        channels = ",".join(item.retrieval_channels)
        print(f"    {rank}. score={item.score:.4f} channels=[{channels}] source={item.source_id} chunk={item.chunk_id}")
        print(f"       domain={item.domain} section={item.section} title={item.title}")
        print(f"       text={truncate_text(item.text, show_text_chars)}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for processor + retrieval integration test."""
    parser = argparse.ArgumentParser(
        description=(
            "Run ProcessorAgent -> RetrievalQuery planning, then execute RetrieverService "
            "with that plan against MongoDB chunk records."
        )
    )
    parser.add_argument(
        "--query",
        required=True,
        help='Raw user query to plan and retrieve, e.g. "How much is Butler Court accommodation?"',
    )
    parser.add_argument("--processor-model", default="gpt-4o-mini")
    parser.add_argument(
        "--top-k-override",
        type=int,
        default=None,
        help="Optional override of ProcessorAgent planned top_k (for retrieval experiments).",
    )

    parser.add_argument("--mongo-db", default="open_day_knowledge")
    parser.add_argument("--mongo-collection", default="kb_chuncks")
    parser.add_argument("--embedder", choices=["fake", "openai"], default="fake")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--vector-index", default="kb_vector_index")
    parser.add_argument("--search-index", default="kb_text_index")
    parser.add_argument(
        "--auto-create-indexes",
        action="store_true",
        help="Attempt Atlas Search index creation/reconciliation on startup.",
    )
    parser.add_argument(
        "--no-wait-for-indexes",
        action="store_true",
        help="When auto-creating indexes, do not wait for READY status.",
    )

    parser.add_argument("--show-text-chars", type=int, default=220)
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional output JSON report path with plan, diagnostics, and evidence.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute end-to-end ProcessorAgent planning and retrieval test."""
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv()

    repo_root = Path(__file__).resolve().parent
    mongo_uri = os.getenv("MONGODB_URI") or read_env_value("MONGODB_URI", repo_root / ".env")
    if not mongo_uri:
        raise ValueError("MONGODB_URI is not set. Add it to .env before running this test.")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. ProcessorAgent planning requires it.")

    repo = MongoRepo(
        mongo_uri=mongo_uri,
        db_name=args.mongo_db,
        collection_name=args.mongo_collection,
    )
    repo.ping()
    doc_count = repo.collection.count_documents({})

    print("[MONGO STATUS]")
    print(f"  database: {args.mongo_db}")
    print(f"  collection: {args.mongo_collection}")
    print(f"  total chunk docs: {doc_count}")

    if doc_count == 0:
        print("  no records found. Run ingestion first, then retry.")
        return

    processor = ProcessorAgent(llm_model=args.processor_model)
    plan = processor.process(args.query)
    if args.top_k_override is not None:
        plan = plan.model_copy(update={"top_k": int(args.top_k_override)})

    print_retrieval_query_plan(plan)

    embedder = build_embedder(args.embedder, args.embedding_model)
    retriever = RetrieverService(
        repo=repo,
        embedder=embedder,
        vector_index_name=args.vector_index,
        atlas_search_index_name=args.search_index,
        auto_create_indexes=args.auto_create_indexes,
    )

    index_report = retriever.check_index_health()
    print("\n[INDEX HEALTH]")
    print(index_report.summary())
    if args.auto_create_indexes:
        print("\n[INDEX ENSURE]")
        ensured = retriever.ensure_indexes(wait_for_ready=not args.no_wait_for_indexes)
        print(ensured.summary())

    evidence_items = retriever.retrieve(plan)
    diagnostics = retriever.get_last_query_diagnostics()
    print_retrieval_results(
        plan=plan,
        evidence_items=evidence_items,
        diagnostics=diagnostics,
        show_text_chars=args.show_text_chars,
    )

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "raw_query": args.query,
            "retrieval_query_plan": plan.model_dump(),
            "retrieval_diagnostics": diagnostics,
            "result_count": len(evidence_items),
            "evidence": [item.model_dump() for item in evidence_items],
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nwrote processor retrieval report: {out_path}")


if __name__ == "__main__":
    main()
