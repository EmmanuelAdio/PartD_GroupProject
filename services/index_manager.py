from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from pymongo.collection import Collection
except ImportError:  # pragma: no cover
    Collection = Any  # type: ignore[assignment]

try:
    from pymongo.operations import SearchIndexModel
except ImportError:  # pragma: no cover
    SearchIndexModel = None

logger = logging.getLogger(__name__)


@dataclass
class IndexHealthReport:
    """Result of an Atlas Search index health check."""

    vector_index_found: bool = False
    text_index_found: bool = False
    vector_index_status: Optional[str] = None   # e.g. "READY", "PENDING", "FAILED"
    text_index_status: Optional[str] = None
    vector_index_dimensions: Optional[int] = None
    errors: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return (
            self.vector_index_found
            and self.text_index_found
            and self.vector_index_status == "READY"
            and self.text_index_status == "READY"
        )

    def summary(self) -> str:
        lines = [
            f"Vector index '{self.vector_index_status}' found={self.vector_index_found}",
            f"Text index   '{self.text_index_status}' found={self.text_index_found}",
        ]
        if self.vector_index_dimensions:
            lines.append(f"Vector dimensions: {self.vector_index_dimensions}")
        if self.errors:
            lines.append("Errors: " + "; ".join(self.errors))
        return "\n".join(lines)


class AtlasIndexManager:
    """
    Manages Atlas Search indexes (vector + text) for a single MongoDB collection.

    Responsibilities:
    - Check whether required indexes exist and are READY
    - Create indexes dynamically, detecting embedding dimensions from stored chunks
    - Drop and recreate indexes when dimensions change (e.g. fake → OpenAI embeddings)
    - Poll until indexes become READY after creation

    Intended to be owned by RetrieverService and called when needed.
    """

    def __init__(
        self,
        *,
        collection: Collection,
        embedder: Any,
        vector_index_name: str = "kb_vector_index",
        atlas_search_index_name: str = "kb_text_index",
        embedding_field: str = "embedding",
        index_ready_timeout_s: int = 120,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.vector_index_name = vector_index_name
        self.atlas_search_index_name = atlas_search_index_name
        self.embedding_field = embedding_field
        self.index_ready_timeout_s = index_ready_timeout_s

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def check_index_health(self) -> IndexHealthReport:
        """
        Check whether the required Atlas Search indexes exist and are READY.

        Uses $listSearchIndexes aggregation (works on Atlas M0+).
        Falls back to driver list_search_indexes() for older driver versions.
        """
        report = IndexHealthReport()

        try:
            indexes = self._list_search_indexes()
        except Exception as exc:
            report.errors.append(f"Could not list search indexes: {exc}")
            logger.debug("list_search_indexes failed: %s", exc, exc_info=True)
            return report

        for idx in indexes:
            name = idx.get("name", "")
            status = idx.get("status", "UNKNOWN")   # Atlas returns READY / PENDING / FAILED

            if name == self.vector_index_name:
                report.vector_index_found = True
                report.vector_index_status = status
                # Drill into the mapping to find numDimensions
                fields = idx.get("latestDefinition", {}).get("fields", [])
                for f in fields:
                    if f.get("type") == "vector":
                        report.vector_index_dimensions = f.get("numDimensions")

            if name == self.atlas_search_index_name:
                report.text_index_found = True
                report.text_index_status = status

        # Surface explicit warnings for each missing / non-ready index
        if not report.vector_index_found:
            report.errors.append(
                f"Vector index '{self.vector_index_name}' NOT FOUND. "
                "Create it in Atlas UI under Search -> Create Search Index (type: Vector Search)."
            )
        elif report.vector_index_status != "READY":
            report.errors.append(
                f"Vector index '{self.vector_index_name}' exists but status="
                f"{report.vector_index_status}. Wait for it to become READY."
            )

        if not report.text_index_found:
            report.errors.append(
                f"Text index '{self.atlas_search_index_name}' NOT FOUND. "
                "Create it in Atlas UI under Search -> Create Search Index (type: Search)."
            )
        elif report.text_index_status != "READY":
            report.errors.append(
                f"Text index '{self.atlas_search_index_name}' exists but status="
                f"{report.text_index_status}. Wait for it to become READY."
            )

        # Warn if embedder dimension doesn't match the Atlas index definition
        if report.vector_index_dimensions is not None:
            try:
                test_vec = self._embed_query("test")
                if len(test_vec) != report.vector_index_dimensions:
                    report.errors.append(
                        f"Embedder produces {len(test_vec)}-dim vectors but Atlas index "
                        f"expects {report.vector_index_dimensions}-dim. "
                        "Re-ingest with the correct embedder and recreate the index!"
                    )
            except Exception:
                pass

        return report

    def ensure_indexes(self, *, wait_for_ready: bool = True) -> IndexHealthReport:
        """
        Create vector and text search indexes if they do not exist or are not READY.
        Dynamically detects the correct embedding dimensions from stored chunks.

        Args:
            wait_for_ready: If True, polls until both indexes are READY (or timeout).

        Returns:
            Updated IndexHealthReport after creation attempt.
        """
        report = self.check_index_health()
        dims = self.detect_embedding_dimensions()

        logger.info(
            "ensure_indexes: dims=%d  vector_found=%s  text_found=%s",
            dims, report.vector_index_found, report.text_index_found,
        )

        if not report.vector_index_found:
            ok, error = self._create_vector_index(dims)
            if not ok and error:
                report.errors.append(error)
        elif report.vector_index_dimensions and report.vector_index_dimensions != dims:
            logger.warning(
                "Vector index has %d dims but stored embeddings are %d dims — "
                "dropping and recreating.",
                report.vector_index_dimensions, dims,
            )
            self._drop_search_index(self.vector_index_name)
            ok, error = self._create_vector_index(dims)
            if not ok and error:
                report.errors.append(error)

        if not report.text_index_found:
            ok, error = self._create_text_index()
            if not ok and error:
                report.errors.append(error)

        if wait_for_ready:
            return self._wait_for_indexes_ready()
        return self.check_index_health()

    def detect_embedding_dimensions(self) -> int:
        """
        Detect embedding dimensions dynamically:
        1. Sample a chunk from MongoDB (reflects what is ACTUALLY stored)
        2. Fall back to asking the embedder directly
        3. Default to 1536 if both fail
        """
        # Method 1: sample from MongoDB
        try:
            sample = self.collection.find_one(
                {self.embedding_field: {"$exists": True, "$not": {"$size": 0}}},
                {self.embedding_field: 1, "_id": 0},
            )
            if sample:
                embedding = sample.get(self.embedding_field, [])
                if isinstance(embedding, list) and len(embedding) > 0:
                    dims = len(embedding)
                    logger.info("Detected %d-dim embeddings from MongoDB sample.", dims)
                    return dims
        except Exception as exc:
            logger.debug("Could not sample embedding from MongoDB: %s", exc)

        # Method 2: ask the embedder
        try:
            test_vec = self._embed_query("dimension probe")
            dims = len(test_vec)
            logger.info("Detected %d-dim embeddings from embedder.", dims)
            return dims
        except Exception as exc:
            logger.debug("Could not get dims from embedder: %s", exc)

        logger.warning("Could not detect embedding dimensions — defaulting to 1536.")
        return 1536

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _list_search_indexes(self) -> List[Dict[str, Any]]:
        """
        Try two methods to list Atlas Search indexes:
        1. $listSearchIndexes aggregation stage  (Atlas, any driver version)
        2. collection.list_search_indexes()      (pymongo >= 4.6)
        """
        # Method 1: aggregation stage — works on all recent Atlas tiers
        try:
            results = list(self.collection.aggregate([{"$listSearchIndexes": {}}]))
            if results is not None:
                return results
        except Exception as agg_exc:
            logger.debug("$listSearchIndexes aggregation failed: %s", agg_exc)

        # Method 2: driver helper (pymongo >= 4.6)
        try:
            return list(self.collection.list_search_indexes())
        except AttributeError:
            raise RuntimeError(
                "Cannot list search indexes. "
                "Upgrade pymongo to >= 4.6 or ensure your cluster supports $listSearchIndexes."
            )

    def _create_vector_index(self, dims: int) -> tuple[bool, Optional[str]]:
        """Create the Atlas Vector Search index with the correct dimensionality."""
        definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": self.embedding_field,
                    "numDimensions": dims,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "domain"},
                {"type": "filter", "path": "section"},
                {"type": "filter", "path": "source_id"},
                {"type": "filter", "path": "entity_tags"},
            ]
        }
        try:
            self._create_search_index(
                name=self.vector_index_name,
                index_type="vectorSearch",
                definition=definition,
            )
            logger.info("Created vector search index '%s' (%d dims).", self.vector_index_name, dims)
            return True, None
        except Exception as exc:
            if "already exists" in str(exc).lower() or "IndexAlreadyExists" in type(exc).__name__:
                logger.info("Vector index '%s' already exists — skipping.", self.vector_index_name)
                return True, None
            else:
                logger.error("Failed to create vector index: %s", exc)
                return False, f"Failed to create vector index '{self.vector_index_name}': {exc}"

    def _create_text_index(self) -> tuple[bool, Optional[str]]:
        """Create the Atlas full-text Search index."""
        definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text":        {"type": "string"},
                    "title":       {"type": "string"},
                    "entity_tags": {"type": "string"},
                    "domain":      {"type": "token"},
                    "section":     {"type": "token"},
                    "source_id":   {"type": "token"},
                },
            }
        }
        try:
            self._create_search_index(
                name=self.atlas_search_index_name,
                index_type="search",
                definition=definition,
            )
            logger.info("Created text search index '%s'.", self.atlas_search_index_name)
            return True, None
        except Exception as exc:
            if "already exists" in str(exc).lower() or "IndexAlreadyExists" in type(exc).__name__:
                logger.info("Text index '%s' already exists — skipping.", self.atlas_search_index_name)
                return True, None
            else:
                logger.error("Failed to create text index: %s", exc)
                return False, f"Failed to create text index '{self.atlas_search_index_name}': {exc}"

    def _drop_search_index(self, index_name: str) -> None:
        """Drop an Atlas Search index by name."""
        try:
            self.collection.drop_search_index(index_name)
            logger.info("Dropped search index '%s'.", index_name)
            time.sleep(3)  # Give Atlas a moment to register the drop before recreating
        except Exception as exc:
            logger.warning("Could not drop index '%s': %s", index_name, exc)

    def _wait_for_indexes_ready(self) -> IndexHealthReport:
        """
        Poll check_index_health() until both indexes are READY
        or index_ready_timeout_s is exceeded.
        """
        deadline = time.monotonic() + self.index_ready_timeout_s
        poll_interval = 5  # seconds

        logger.info(
            "Waiting up to %ds for indexes to become READY...",
            self.index_ready_timeout_s,
        )

        while time.monotonic() < deadline:
            report = self.check_index_health()
            if report.is_healthy:
                return report

            statuses = (
                f"vector={report.vector_index_status} "
                f"text={report.text_index_status}"
            )
            logger.info("Indexes not ready yet (%s) — retrying in %ds...", statuses, poll_interval)
            time.sleep(poll_interval)

        logger.warning(
            "Timed out after %ds waiting for indexes to become READY.",
            self.index_ready_timeout_s,
        )
        return self.check_index_health()

    def _embed_query(self, text: str) -> List[float]:
        """Thin embed wrapper supporting multiple embedder method signatures."""
        if hasattr(self.embedder, "embed_one"):
            vector = self.embedder.embed_one(text)
        elif hasattr(self.embedder, "embed_text"):
            vector = self.embedder.embed_text(text)
        elif hasattr(self.embedder, "embed_many"):
            vectors = self.embedder.embed_many([text])
            vector = vectors[0] if vectors else []
        else:
            raise TypeError("Embedder must implement embed_one(), embed_text(), or embed_many().")
        return [float(v) for v in vector]

    def _create_search_index(self, *, name: str, index_type: str, definition: Dict[str, Any]) -> None:
        """Create Atlas Search index using driver helper or DB command fallback."""
        if hasattr(self.collection, "create_search_index"):
            if SearchIndexModel is not None:
                model = SearchIndexModel(definition=definition, name=name, type=index_type)
                self.collection.create_search_index(model=model)
            else:
                self.collection.create_search_index(
                    {"name": name, "type": index_type, "definition": definition}
                )
            return

        # Fallback for older drivers that do not expose create_search_index helper.
        self.collection.database.command(
            {
                "createSearchIndexes": self.collection.name,
                "indexes": [{"name": name, "type": index_type, "definition": definition}],
            }
        )
