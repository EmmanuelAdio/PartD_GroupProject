from __future__ import annotations

import os
from typing import Dict, List, Optional

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
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()

        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI") or self._read_env_value("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI is not set. Add it to your environment or pass mongo_uri explicitly.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

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
