from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from schemas.models import RetrievalQuery
from services.embedding_service import DeterministicEmbeddingService, EmbeddingService
from services.mongo_repo import MongoRepo
from services.retriever_service import RetrieverService


class _DummyCollection:
    """Minimal collection stub for unit tests that call private rerank logic only."""


class _DummyEmbedder:
    """Minimal embedder stub required by RetrieverService constructor."""

    def embed_one(self, _: str) -> List[float]:
        return [0.1, 0.2, 0.3]


def _build_unit_test_retriever() -> RetrieverService:
    """Build RetrieverService without index checks for isolated rerank tests."""
    return RetrieverService(
        embedder=_DummyEmbedder(),
        collection=_DummyCollection(),
        vector_weight=0.5,
        text_weight=0.5,
        overlap_boost=0.0,
        check_indexes_on_init=False,
    )


def test_merge_and_rerank_section_boost_reranks_upward() -> None:
    """A matching section should receive a positive soft boost in final score."""
    retriever = _build_unit_test_retriever()

    vector_hits = [
        {
            "chunk_id": "chunk_entry",
            "source_id": "s1",
            "text": "Entry requirements details",
            "section": "entry_requirements",
            "entity_tags": [],
            "_score": 0.8,
        }
    ]
    text_hits = [
        {
            "chunk_id": "chunk_overview",
            "source_id": "s2",
            "text": "Course overview details",
            "section": "overview",
            "entity_tags": [],
            "_score": 0.8,
        }
    ]

    merged = retriever._merge_and_rerank(
        vector_hits=vector_hits,
        text_hits=text_hits,
        top_k=2,
        boost_sections={"entry_requirements"},
        boost_entity_tags=set(),
    )

    by_chunk = {item.chunk_id: item for item in merged}
    assert by_chunk["chunk_entry"].score > by_chunk["chunk_overview"].score


def test_merge_and_rerank_entity_tag_boost_reranks_upward() -> None:
    """A matching entity tag should receive a positive soft boost in final score."""
    retriever = _build_unit_test_retriever()

    vector_hits = [
        {
            "chunk_id": "chunk_cs",
            "source_id": "s1",
            "text": "Computer Science information",
            "section": "overview",
            "entity_tags": ["Computer Science"],
            "_score": 0.8,
        }
    ]
    text_hits = [
        {
            "chunk_id": "chunk_generic",
            "source_id": "s2",
            "text": "Generic information",
            "section": "overview",
            "entity_tags": [],
            "_score": 0.8,
        }
    ]

    merged = retriever._merge_and_rerank(
        vector_hits=vector_hits,
        text_hits=text_hits,
        top_k=2,
        boost_sections=set(),
        boost_entity_tags={"computer science"},
    )

    by_chunk = {item.chunk_id: item for item in merged}
    assert by_chunk["chunk_cs"].score > by_chunk["chunk_generic"].score


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


def build_default_queries(top_k: int) -> List[RetrievalQuery]:
    return [
        RetrievalQuery(
            query_text="Butler Court",
            top_k=top_k,
            domains=["accommodation"],
            entity_tags=["Butler Court"],
        ),
        RetrievalQuery(
            query_text="UCAS entry requirements",
            top_k=top_k,
            domains=["courses", "admissions"],
            entity_tags=["UCAS"],
        ),
        RetrievalQuery(
            query_text="module codes for computer science",
            top_k=top_k,
            domains=["courses"],
        ),
    ]


def load_queries_from_file(path: Path) -> List[RetrievalQuery]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("--queries-file must contain a JSON object or list of JSON objects.")

    out: List[RetrievalQuery] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        out.append(RetrievalQuery.model_validate(row))
    return out


def parse_query_json_rows(values: Sequence[str]) -> List[RetrievalQuery]:
    out: List[RetrievalQuery] = []
    for raw in values:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("--query-json must be a JSON object.")
        out.append(RetrievalQuery.model_validate(parsed))
    return out


def parse_query_text_rows(
    values: Sequence[str],
    *,
    top_k: int,
    domains: Sequence[str],
    entity_tags: Sequence[str],
    sections: Sequence[str],
    source_ids: Sequence[str],
    version: str | None,
    vector_k: int | None,
    text_k: int | None,
) -> List[RetrievalQuery]:
    out: List[RetrievalQuery] = []
    for text in values:
        out.append(
            RetrievalQuery(
                query_text=text,
                top_k=top_k,
                domains=list(domains),
                entity_tags=list(entity_tags),
                sections=list(sections),
                source_ids=list(source_ids),
                version=version,
                vector_k=vector_k,
                text_k=text_k,
            )
        )
    return out


def build_queries(args: argparse.Namespace) -> List[RetrievalQuery]:
    queries: List[RetrievalQuery] = []

    if args.queries_file:
        queries.extend(load_queries_from_file(Path(args.queries_file)))
    if args.query_json:
        queries.extend(parse_query_json_rows(args.query_json))
    if args.query:
        queries.extend(
            parse_query_text_rows(
                args.query,
                top_k=args.top_k,
                domains=args.domain,
                entity_tags=args.entity_tag,
                sections=args.section,
                source_ids=args.source_id,
                version=args.version,
                vector_k=args.vector_k,
                text_k=args.text_k,
            )
        )

    if not queries:
        return build_default_queries(top_k=args.top_k)

    return queries


def estimate_embedding_dim(collection) -> int | None:
    probe = collection.find_one({"embedding.0": {"$exists": True}}, {"_id": 0, "embedding": 1})
    if not probe:
        return None
    embedding = probe.get("embedding")
    if isinstance(embedding, list):
        return len(embedding)
    return None


def embedder_probe_dim(embedder) -> int | None:
    try:
        if hasattr(embedder, "embed_text"):
            return len(embedder.embed_text("dimension probe"))
        if hasattr(embedder, "embed_one"):
            return len(embedder.embed_one("dimension probe"))
        if hasattr(embedder, "embed_many"):
            vectors = embedder.embed_many(["dimension probe"])
            return len(vectors[0]) if vectors else None
    except Exception:
        return None
    return None


def format_filters(query: RetrievalQuery) -> str:
    filters = {
        "domains": query.domains or ([query.domain] if query.domain else []),
        "entity_tags": query.entity_tags,
        "sections": query.sections or ([query.section] if query.section else []),
        "source_ids": query.source_ids,
        "version": query.version,
    }
    active = {k: v for k, v in filters.items() if v}
    return json.dumps(active) if active else "{}"


def truncate_text(text: str, max_chars: int) -> str:
    value = " ".join((text or "").split())
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}..."


def run_retrieval_test(
    retriever: RetrieverService,
    queries: Sequence[RetrievalQuery],
    *,
    show_text_chars: int,
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []

    print("\n[RETRIEVAL TEST]")
    print(f"  total queries: {len(queries)}")

    for idx, query in enumerate(queries, start=1):
        print(f"\n[QUERY {idx}]")
        print(f"  query_text: {query.query_text}")
        print(f"  top_k: {query.top_k}")
        print(f"  filters: {format_filters(query)}")

        evidence_items = retriever.retrieve(query)
        diagnostics = retriever.get_last_query_diagnostics()
        fallback_used = bool(
            diagnostics.get("vector_fallback_used") or diagnostics.get("text_fallback_used")
        )
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

        report.append(
            {
                "query": query.model_dump(),
                "retrieval_diagnostics": diagnostics,
                "result_count": len(evidence_items),
                "evidence": [item.model_dump() for item in evidence_items],
            }
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval tests against MongoDB chunk records.")
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
        "--check-indexes-only",
        action="store_true",
        help="Check/ensure indexes and exit without running retrieval queries.",
    )
    parser.add_argument(
        "--no-wait-for-indexes",
        action="store_true",
        help="When auto-creating indexes, do not wait for READY status.",
    )

    parser.add_argument("--query", action="append", default=[], help="Raw query text (can be repeated).")
    parser.add_argument(
        "--query-json",
        action="append",
        default=[],
        help='Structured query JSON (can be repeated), e.g. \'{"query_text":"Butler Court","top_k":5}\'',
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Path to JSON file containing one query object or list of query objects.",
    )

    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--vector-k", type=int, default=None)
    parser.add_argument("--text-k", type=int, default=None)
    parser.add_argument("--domain", action="append", default=[], help="Metadata domain filter (repeatable).")
    parser.add_argument("--entity-tag", action="append", default=[], help="Metadata entity tag filter (repeatable).")
    parser.add_argument("--section", action="append", default=[], help="Metadata section filter (repeatable).")
    parser.add_argument("--source-id", action="append", default=[], help="Metadata source_id filter (repeatable).")
    parser.add_argument("--version", default=None, help="Metadata version filter.")

    parser.add_argument("--show-text-chars", type=int, default=220)
    parser.add_argument("--write-report", default=None, help="Optional output JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv()

    repo_root = Path(__file__).resolve().parent
    mongo_uri = os.getenv("MONGODB_URI") or read_env_value("MONGODB_URI", repo_root / ".env")
    if not mongo_uri:
        raise ValueError("MONGODB_URI is not set. Add it to .env before running retrieval tests.")

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
        print("  no records found. Run ingestion first, then retry retrieval test.")
        return

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

    if args.check_indexes_only:
        return

    stored_dim = estimate_embedding_dim(repo.collection)
    query_dim = embedder_probe_dim(embedder)
    if stored_dim is not None and query_dim is not None and stored_dim != query_dim:
        print("\n[WARNING]")
        print(
            f"  embedding dim mismatch: stored={stored_dim}, query_embedder={query_dim}. "
            "Use the same embedding backend/model as ingestion for best vector retrieval."
        )

    queries = build_queries(args)
    report = run_retrieval_test(
        retriever=retriever,
        queries=queries,
        show_text_chars=args.show_text_chars,
    )

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nwrote retrieval report: {out_path}")


if __name__ == "__main__":
    main()
