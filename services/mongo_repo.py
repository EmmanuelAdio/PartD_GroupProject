from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from pymongo import MongoClient, ReplaceOne
from pymongo.errors import ServerSelectionTimeoutError
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

try:
    from schemas.models import ChunkRecord
except ImportError:  # pragma: no cover
    from ..schemas.models import ChunkRecord


class MongoRepo:
    """MongoDB persistence adapter for chunk records and ingestion manifests.

    Provides upsert, query, and delete operations for `ChunkRecord` documents
    and tracks per-source ingestion state via a manifest collection so the
    orchestrator can skip re-embedding unchanged sources.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: str = "open_day_knowledge",
        collection_name: str = "kb_chuncks",
        manifest_collection_name: str = "kb_ingestion_manifest",
        server_selection_timeout_ms: int = 30_000,
        connect_timeout_ms: int = 20_000,
        socket_timeout_ms: int = 20_000,
        tls_ca_file: Optional[str] = None,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()

        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI") or self._read_env_value("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI is not set. Add it to your environment or pass mongo_uri explicitly.")

        self.server_selection_timeout_ms = int(
            os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", str(server_selection_timeout_ms))
        )
        self.connect_timeout_ms = int(
            os.getenv("MONGODB_CONNECT_TIMEOUT_MS", str(connect_timeout_ms))
        )
        self.socket_timeout_ms = int(
            os.getenv("MONGODB_SOCKET_TIMEOUT_MS", str(socket_timeout_ms))
        )
        self.tls_ca_file = tls_ca_file or os.getenv("MONGODB_TLS_CA_FILE")
        if not self.tls_ca_file and certifi is not None:
            self.tls_ca_file = certifi.where()

        client_kwargs: Dict[str, Any] = {
            "serverSelectionTimeoutMS": self.server_selection_timeout_ms,
            "connectTimeoutMS": self.connect_timeout_ms,
            "socketTimeoutMS": self.socket_timeout_ms,
        }
        if self.tls_ca_file:
            client_kwargs["tlsCAFile"] = self.tls_ca_file

        self.client = MongoClient(self.mongo_uri, **client_kwargs)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.manifest_collection = self.db[manifest_collection_name]
        try:
            self.ensure_indexes()
        except ServerSelectionTimeoutError as exc:
            raise RuntimeError(self._format_connection_error(exc)) from exc

    def ping(self) -> None:
        """Verify connectivity by issuing a lightweight admin ping.

        Raises:
            RuntimeError: If the server cannot be reached within the configured timeout.
        """
        try:
            self.client.admin.command("ping")
        except ServerSelectionTimeoutError as exc:
            raise RuntimeError(self._format_connection_error(exc)) from exc

    def upsert_chunks(self, records: List[ChunkRecord]) -> Dict[str, int]:
        """Bulk-upsert chunk records matched by `chunk_id`.

        Args:
            records: List of ChunkRecord objects to write.

        Returns:
            Dict with keys ``upserted_count``, ``modified_count``, ``matched_count``.
        """
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
        """Create required MongoDB indexes if they do not already exist.

        Creates a unique index on ``chunk_id``, standard indexes on ``source_id``
        and ``version``, a compound text index for lexical fallback retrieval,
        and a unique manifest index on ``source_id``.
        """
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
        """Return the subset of the given chunk IDs that already exist in the collection."""
        chunk_ids = [c for c in chunk_ids if c]
        if not chunk_ids:
            return set()
        cursor = self.collection.find(
            {"chunk_id": {"$in": chunk_ids}},
            {"_id": 0, "chunk_id": 1},
        )
        return {doc["chunk_id"] for doc in cursor if "chunk_id" in doc}

    def count_source_records(self, source_id: str) -> int:
        """Return the number of chunk documents stored for the given source ID."""
        return int(self.collection.count_documents({"source_id": source_id}))

    def delete_source_records(self, source_id: str) -> int:
        """Delete all chunk documents for the given source ID and return the deleted count."""
        result = self.collection.delete_many({"source_id": source_id})
        return int(result.deleted_count)

    def get_source_manifest(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Return the ingestion manifest for the given source ID, or None if not found."""
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
        """Write or update the ingestion manifest entry for a source.

        The manifest stores the source file hash and pipeline configuration hash so
        the orchestrator can skip re-embedding unchanged sources on subsequent runs.

        Args:
            source_id: Unique identifier for the data source.
            source_hash: SHA-256 hash of the raw source file content.
            pipeline_hash: Hash of the ingestion configuration (embedder, version, etc.).
            pipeline_signature: Human-readable dict of pipeline config values for debugging.
            source_path: Filesystem path to the source file.
            records_in_db: Number of chunk records written in the last ingestion run.
        """
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

    def _format_connection_error(self, exc: ServerSelectionTimeoutError) -> str:
        tls_ca = self.tls_ca_file or "not-set"
        return (
            "MongoDB connection failed before startup completed.\n"
            f"python_executable={sys.executable}\n"
            f"serverSelectionTimeoutMS={self.server_selection_timeout_ms}, "
            f"connectTimeoutMS={self.connect_timeout_ms}, socketTimeoutMS={self.socket_timeout_ms}\n"
            f"tlsCAFile={tls_ca}\n"
            f"error={exc}\n"
            "Troubleshooting: run `python scripts/mongo_tls_probe.py`, verify Atlas IP allowlist, "
            "verify DB user permissions, and ensure you are using the project .venv interpreter."
        )
