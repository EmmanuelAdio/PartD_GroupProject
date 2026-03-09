from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from pymongo import MongoClient, ReplaceOne
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from schemas.models import ChunkRecord
except ImportError:  # pragma: no cover
    from ..schemas.models import ChunkRecord


class MongoRepo:
    """MongoDB persistence adapter for chunk records."""

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: str = "open_day_knowledge",
        collection_name: str = "kb_chuncks",
        manifest_collection_name: str = "kb_ingestion_manifest",
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()

        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI") or self._read_env_value("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI is not set. Add it to your environment or pass mongo_uri explicitly.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.manifest_collection = self.db[manifest_collection_name]
        self.ensure_indexes()

    def ping(self) -> None:
        self.client.admin.command("ping")

    def upsert_chunks(self, records: List[ChunkRecord]) -> Dict[str, int]:
        if not records:
            return {"upserted_count": 0, "modified_count": 0, "matched_count": 0}

        ops = [
            ReplaceOne(
                {"chunk_id": rec.chunk_id},
                rec.model_dump(),
                upsert=True,
            )
            for rec in records
        ]
        result = self.collection.bulk_write(ops, ordered=False)
        return {
            "upserted_count": int(result.upserted_count),
            "modified_count": int(result.modified_count),
            "matched_count": int(result.matched_count),
        }

    def ensure_indexes(self) -> None:
        self.collection.create_index("chunk_id", unique=True, name="chunk_id_unique")
        self.collection.create_index("source_id", name="source_id_idx")
        self.collection.create_index("version", name="version_idx")
        # Supports lexical fallback path in RetrieverService when Atlas Search is unavailable.
        self.collection.create_index(
            [("text", "text"), ("title", "text"), ("entity_tags", "text")],
            name="chunk_text_search_idx",
            default_language="english",
        )
        self.manifest_collection.create_index("source_id", unique=True, name="manifest_source_unique")

    def get_existing_chunk_ids(self, chunk_ids: Iterable[str]) -> Set[str]:
        chunk_ids = [c for c in chunk_ids if c]
        if not chunk_ids:
            return set()
        cursor = self.collection.find(
            {"chunk_id": {"$in": chunk_ids}},
            {"_id": 0, "chunk_id": 1},
        )
        return {doc["chunk_id"] for doc in cursor if "chunk_id" in doc}

    def count_source_records(self, source_id: str) -> int:
        return int(self.collection.count_documents({"source_id": source_id}))

    def delete_source_records(self, source_id: str) -> int:
        result = self.collection.delete_many({"source_id": source_id})
        return int(result.deleted_count)

    def get_source_manifest(self, source_id: str) -> Optional[Dict[str, Any]]:
        return self.manifest_collection.find_one({"source_id": source_id}, {"_id": 0})

    def upsert_source_manifest(
        self,
        source_id: str,
        source_hash: str,
        pipeline_hash: str,
        pipeline_signature: Dict[str, Any],
        source_path: str,
        records_in_db: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "source_id": source_id,
            "source_hash": source_hash,
            "pipeline_hash": pipeline_hash,
            "pipeline_signature": pipeline_signature,
            "source_path": source_path,
            "records_in_db": int(records_in_db),
            "updated_at": now,
        }
        self.manifest_collection.replace_one({"source_id": source_id}, payload, upsert=True)

    @staticmethod
    def _read_env_value(key: str, env_path: str = ".env") -> Optional[str]:
        if not os.path.exists(env_path):
            return None

        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                lhs, rhs = line.split("=", 1)
                if lhs.strip() == key:
                    return rhs.strip().strip('"').strip("'")
        return None
