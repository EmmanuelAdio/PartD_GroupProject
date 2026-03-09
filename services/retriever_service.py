from __future__ import annotations

import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from pymongo.collection import Collection
except ImportError:  # pragma: no cover
    Collection = Any  # type: ignore[assignment]

try:
    from schemas.models import EvidenceItem, RetrievalQuery
except ImportError:  # pragma: no cover
    from ..schemas.models import EvidenceItem, RetrievalQuery

try:
    from services.index_manager import AtlasIndexManager, IndexHealthReport
except ImportError:  # pragma: no cover
    from index_manager import AtlasIndexManager, IndexHealthReport  # type: ignore[no-redef]


@dataclass
class _MergedCandidate:
    """Internal merge/rerank container for one chunk candidate."""

    doc: Dict[str, Any]
    channels: Set[str] = field(default_factory=set)
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    vector_rank: Optional[int] = None
    text_rank: Optional[int] = None


class RetrieverService:
    """Hybrid retriever over MongoDB chunk documents.

    Retrieval flow:
    1) Embed the query text.
    2) Run vector search (+ metadata filters).
    3) Run text/lexical search (+ metadata filters).
    4) Merge and rerank with overlap boosting.
    5) Return clean `EvidenceItem` rows for the Answerer Agent.

    Notes on MongoDB syntax:
    - Atlas Vector Search (`$vectorSearch`) and Atlas Search (`$search`) can differ by
      index names/mappings between deployments.
    - This implementation uses sensible defaults and catches failures, then falls back
      to simpler queries so the service remains usable in local/student environments.
    """

    def __init__(
        self,
        *,
        embedder: Any,
        collection: Optional[Collection] = None,
        repo: Any = None,
        vector_index_name: str = "kb_vector_index",
        atlas_search_index_name: str = "kb_text_index",
        embedding_field: str = "embedding",
        text_field: str = "text",
        vector_weight: float = 0.55,
        text_weight: float = 0.45,
        overlap_boost: float = 0.2,
        fallback_scan_limit: int = 5000,
        check_indexes_on_init: bool = True,
        auto_create_indexes: bool = True,
        index_ready_timeout_s: int = 120,
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required.")

        if collection is None and repo is not None and hasattr(repo, "collection"):
            collection = repo.collection
        if collection is None:
            raise ValueError("Provide either collection or repo (with .collection).")

        if vector_weight < 0 or text_weight < 0:
            raise ValueError("vector_weight and text_weight must be non-negative.")
        if vector_weight == 0 and text_weight == 0:
            raise ValueError("At least one of vector_weight/text_weight must be > 0.")
        if fallback_scan_limit <= 0:
            raise ValueError("fallback_scan_limit must be > 0.")

        self.embedder = embedder
        self.collection = collection
        self.repo = repo

        self.vector_index_name = vector_index_name
        self.atlas_search_index_name = atlas_search_index_name
        self.embedding_field = embedding_field
        self.text_field = text_field

        total_weight = vector_weight + text_weight
        self.vector_weight = vector_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.overlap_boost = overlap_boost
        self.fallback_scan_limit = fallback_scan_limit
        self.last_query_diagnostics: Dict[str, Any] = {}
        self._index_health_report: Optional[IndexHealthReport] = None
        self._prefer_atlas_vector: bool = True
        self._prefer_atlas_text: bool = True

        self._index_manager = AtlasIndexManager(
            collection=collection,
            embedder=embedder,
            vector_index_name=vector_index_name,
            atlas_search_index_name=atlas_search_index_name,
            embedding_field=embedding_field,
            index_ready_timeout_s=index_ready_timeout_s,
        )

        if check_indexes_on_init:
            report = self._index_manager.check_index_health()
            self._index_health_report = report
            if not report.is_healthy:
                if auto_create_indexes:
                    logger.info("Indexes not healthy — attempting to create them automatically.")
                    self._index_health_report = self._index_manager.ensure_indexes(wait_for_ready=True)
                else:
                    warnings.warn(
                        f"\n[RetrieverService] Atlas index health check FAILED:\n{report.summary()}\n"
                        "Falling back to Python-side scan. Performance will be degraded.\n"
                        "Pass auto_create_indexes=True to create them automatically.",
                        stacklevel=2,
                    )
                    logger.warning("Atlas index health check failed.\n%s", report.summary())
            else:
                logger.info("Atlas index health check passed.\n%s", report.summary())
            self._sync_atlas_preferences_from_health_report()

    def retrieve(self, query: RetrievalQuery | Dict[str, Any]) -> List[EvidenceItem]:
        """Run hybrid retrieval and return top-k evidence items."""
        query_model = query if isinstance(query, RetrievalQuery) else RetrievalQuery.model_validate(query)
        query_text = self._clean_text(query_model.query_text)
        if not query_text:
            return []
        self._reset_diagnostics(query_text=query_text, top_k=query_model.top_k)

        metadata_filter = self._build_metadata_filter(query_model)
        top_k = query_model.top_k
        vector_k = query_model.vector_k or max(20, top_k * 4)
        text_k = query_model.text_k or max(20, top_k * 4)

        query_embedding = self._embed_query(query_text)
        vector_hits = self._vector_search(query_embedding, metadata_filter, vector_k)
        text_hits = self._text_search(query_text, metadata_filter, text_k)

        merged = self._merge_and_rerank(vector_hits=vector_hits, text_hits=text_hits, top_k=top_k)
        self.last_query_diagnostics["vector_hits"] = len(vector_hits)
        self.last_query_diagnostics["text_hits"] = len(text_hits)
        self.last_query_diagnostics["final_hits"] = len(merged)
        return merged

    def _vector_search(
        self,
        query_embedding: Sequence[float],
        metadata_filter: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """Run Atlas vector search, with Python cosine fallback for local/dev use."""
        if not self._prefer_atlas_vector:
            self.last_query_diagnostics["vector_mode"] = "fallback_cosine"
            self.last_query_diagnostics["vector_fallback_used"] = True
            self.last_query_diagnostics["vector_search_error"] = "Skipped Atlas vector search: index not ready."
            return self._vector_search_fallback(query_embedding, metadata_filter, candidate_k)

        try:
            vector_stage: Dict[str, Any] = {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": self.embedding_field,
                    "queryVector": list(query_embedding),
                    "numCandidates": max(candidate_k * 5, candidate_k + 10),
                    "limit": candidate_k,
                }
            }
            if metadata_filter:
                vector_stage["$vectorSearch"]["filter"] = metadata_filter

            pipeline = [
                vector_stage,
                {"$project": self._project_fields(with_vector_score=True)},
            ]
            docs = list(self.collection.aggregate(pipeline))
            self.last_query_diagnostics["vector_mode"] = "atlas_vector"
            self.last_query_diagnostics["vector_fallback_used"] = False
            return [self._as_hit(doc, channel="vector") for doc in docs]
        except Exception as exc:
            # Atlas index/stage syntax can vary between environments.
            self.last_query_diagnostics["vector_mode"] = "fallback_cosine"
            self.last_query_diagnostics["vector_fallback_used"] = True
            self.last_query_diagnostics["vector_search_error"] = f"{type(exc).__name__}: {exc}"
            return self._vector_search_fallback(query_embedding, metadata_filter, candidate_k)

    def _text_search(
        self,
        query_text: str,
        metadata_filter: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """Run text retrieval with Atlas Search -> Mongo $text -> regex fallback."""
        atlas_hits: List[Dict[str, Any]] = []
        if self._prefer_atlas_text:
            atlas_hits = self._atlas_text_search(query_text, metadata_filter, candidate_k)

        if atlas_hits:
            self.last_query_diagnostics["text_mode"] = "atlas_search"
            self.last_query_diagnostics["text_fallback_used"] = False
            return atlas_hits

        self.last_query_diagnostics["text_fallback_used"] = True
        if not self._prefer_atlas_text:
            self.last_query_diagnostics["atlas_search_error"] = "Skipped Atlas search: index not ready."
        text_index_hits = self._mongo_text_index_search(query_text, metadata_filter, candidate_k)
        if text_index_hits:
            self.last_query_diagnostics["text_mode"] = "mongo_text"
            return text_index_hits

        self.last_query_diagnostics["text_mode"] = "regex_scan"
        return self._regex_text_search_fallback(query_text, metadata_filter, candidate_k)

    def _atlas_text_search(
        self,
        query_text: str,
        metadata_filter: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """Atlas Search path.

        Placeholder note:
        Atlas Search index mappings and filter clauses are deployment-specific.
        If this returns no hits/errors in your environment, tune the index name and
        filter clauses in `_build_atlas_filter_clauses`.
        """
        try:
            compound: Dict[str, Any] = {
                "must": [
                    {
                        "text": {
                            "query": query_text,
                            "path": [self.text_field, "title", "entity_tags"],
                        }
                    }
                ]
            }
            atlas_filter = self._build_atlas_filter_clauses(metadata_filter)
            if atlas_filter:
                compound["filter"] = atlas_filter

            pipeline = [
                {
                    "$search": {
                        "index": self.atlas_search_index_name,
                        "compound": compound,
                    }
                },
                {"$limit": candidate_k},
                {"$project": self._project_fields(with_text_score=True)},
            ]
            docs = list(self.collection.aggregate(pipeline))
            return [self._as_hit(doc, channel="text") for doc in docs]
        except Exception as exc:
            self.last_query_diagnostics["atlas_search_error"] = f"{type(exc).__name__}: {exc}"
            return []

    def _mongo_text_index_search(
        self,
        query_text: str,
        metadata_filter: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """Fallback to native Mongo `$text` search if a text index exists."""
        try:
            match_query: Dict[str, Any] = {"$text": {"$search": query_text}}
            if metadata_filter:
                match_query.update(metadata_filter)

            projection = self._project_fields()
            projection["score"] = 1

            pipeline = [
                {"$match": match_query},
                {"$addFields": {"score": {"$meta": "textScore"}}},
                {"$sort": {"score": -1}},
                {"$limit": candidate_k},
                {"$project": projection},
            ]
            docs = list(self.collection.aggregate(pipeline))
            return [self._as_hit(doc, channel="text") for doc in docs]
        except Exception as exc:
            self.last_query_diagnostics["mongo_text_error"] = f"{type(exc).__name__}: {exc}"
            return []

    def _regex_text_search_fallback(
        self,
        query_text: str,
        metadata_filter: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """Last-resort lexical fallback for environments without text indexes."""
        tokens = self._tokenize(query_text)
        query_lc = query_text.lower()
        if not tokens and not query_lc:
            return []

        projection = self._project_fields(include_embedding=False)
        cursor = (
            self.collection.find(metadata_filter, projection)
            .limit(self.fallback_scan_limit)
        )

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in cursor:
            text_value = self._clean_text(str(doc.get(self.text_field, "")))
            if not text_value:
                continue

            text_lc = text_value.lower()
            score = 0.0
            if query_lc and query_lc in text_lc:
                score += 3.0

            for token in tokens:
                if token in text_lc:
                    score += 1.0

            score += self._entity_exact_bonus(query_text, doc.get("entity_tags"))
            if score <= 0:
                continue

            doc["score"] = score
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._as_hit(doc, channel="text") for _, doc in scored[:candidate_k]]

    def _vector_search_fallback(
        self,
        query_embedding: Sequence[float],
        metadata_filter: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """Cosine-similarity fallback when Atlas Vector Search is unavailable."""
        projection = self._project_fields(include_embedding=True)
        cursor = (
            self.collection.find(metadata_filter, projection)
            .limit(self.fallback_scan_limit)
        )

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in cursor:
            candidate_embedding = doc.get(self.embedding_field)
            if not isinstance(candidate_embedding, list) or not candidate_embedding:
                continue

            score = self._cosine_similarity(query_embedding, candidate_embedding)
            doc["score"] = score
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._as_hit(doc, channel="vector") for _, doc in scored[:candidate_k]]

    def _merge_and_rerank(
        self,
        *,
        vector_hits: List[Dict[str, Any]],
        text_hits: List[Dict[str, Any]],
        top_k: int,
    ) -> List[EvidenceItem]:
        """Merge channels, dedupe by chunk_id, and compute a simple weighted score."""
        candidates: Dict[str, _MergedCandidate] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            self._accumulate_hit(candidates, hit, rank=rank, channel="vector")
        for rank, hit in enumerate(text_hits, start=1):
            self._accumulate_hit(candidates, hit, rank=rank, channel="text")

        vector_bounds = self._score_bounds(vector_hits)
        text_bounds = self._score_bounds(text_hits)

        merged: List[EvidenceItem] = []
        for chunk_id, cand in candidates.items():
            vector_component = self._channel_component(
                rank=cand.vector_rank,
                total=len(vector_hits),
                raw_score=cand.vector_score,
                bounds=vector_bounds,
                weight=self.vector_weight,
            )
            text_component = self._channel_component(
                rank=cand.text_rank,
                total=len(text_hits),
                raw_score=cand.text_score,
                bounds=text_bounds,
                weight=self.text_weight,
            )
            overlap_component = self.overlap_boost if len(cand.channels) > 1 else 0.0
            final_score = vector_component + text_component + overlap_component

            merged.append(
                self._to_evidence_item(
                    doc=cand.doc,
                    chunk_id=chunk_id,
                    score=final_score,
                    vector_score=cand.vector_score,
                    text_score=cand.text_score,
                    channels=cand.channels,
                )
            )

        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def _accumulate_hit(
        self,
        candidates: Dict[str, _MergedCandidate],
        hit: Dict[str, Any],
        *,
        rank: int,
        channel: str,
    ) -> None:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id:
            return

        doc = self._normalize_doc(hit)
        bucket = candidates.get(chunk_id)
        if bucket is None:
            bucket = _MergedCandidate(doc=doc)
            candidates[chunk_id] = bucket
        else:
            bucket.doc = self._prefer_richer_doc(bucket.doc, doc)

        bucket.channels.add(channel)
        score_value = self._coerce_float(hit.get("_score"), default=0.0)
        if channel == "vector":
            bucket.vector_rank = rank if bucket.vector_rank is None else min(bucket.vector_rank, rank)
            bucket.vector_score = score_value
        else:
            bucket.text_rank = rank if bucket.text_rank is None else min(bucket.text_rank, rank)
            bucket.text_score = score_value

    def _to_evidence_item(
        self,
        *,
        doc: Dict[str, Any],
        chunk_id: str,
        score: float,
        vector_score: Optional[float],
        text_score: Optional[float],
        channels: Set[str],
    ) -> EvidenceItem:
        ordered_channels = [channel for channel in ("vector", "text") if channel in channels]
        return EvidenceItem(
            chunk_id=chunk_id,
            source_id=str(doc.get("source_id", "")),
            source_type=doc.get("source_type"),
            title=doc.get("title"),
            url=doc.get("url"),
            text=str(doc.get("text", "")),
            domain=doc.get("domain"),
            entity_tags=self._as_str_list(doc.get("entity_tags")),
            section=doc.get("section"),
            order=int(doc.get("order", 0) or 0),
            version=doc.get("version"),
            score=float(score),
            vector_score=vector_score,
            text_score=text_score,
            retrieval_channels=ordered_channels,
            metadata=doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {},
        )

    def _build_metadata_filter(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Build shared Mongo filter from structured metadata constraints."""
        filters: Dict[str, Any] = {}

        domains = self._unique_non_empty([query.domain, *query.domains], to_lower=True)
        if domains:
            filters["domain"] = domains[0] if len(domains) == 1 else {"$in": domains}

        sections = self._unique_non_empty([query.section, *query.sections], to_lower=False)
        if sections:
            filters["section"] = sections[0] if len(sections) == 1 else {"$in": sections}

        tags = self._unique_non_empty(query.entity_tags, to_lower=False)
        if tags:
            filters["entity_tags"] = {"$in": tags}

        source_ids = self._unique_non_empty(query.source_ids, to_lower=False)
        if source_ids:
            filters["source_id"] = {"$in": source_ids}

        if query.version:
            filters["version"] = query.version

        return filters

    def get_last_query_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent `retrieve` call."""
        return dict(self.last_query_diagnostics)

    # ------------------------------------------------------------------ #
    #  Index management — delegates to AtlasIndexManager                  #
    # ------------------------------------------------------------------ #

    def check_index_health(self) -> IndexHealthReport:
        """Check whether Atlas Search indexes exist and are READY."""
        report = self._index_manager.check_index_health()
        self._index_health_report = report
        self._sync_atlas_preferences_from_health_report()
        return report

    def ensure_indexes(self, *, wait_for_ready: bool = True) -> IndexHealthReport:
        """
        Create vector and text search indexes if missing or mismatched.
        Detects embedding dimensions dynamically from stored chunks.
        """
        report = self._index_manager.ensure_indexes(wait_for_ready=wait_for_ready)
        self._index_health_report = report
        self._sync_atlas_preferences_from_health_report()
        return report

    def _build_atlas_filter_clauses(self, metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Translate basic Mongo metadata filters into Atlas Search filter clauses.

        This is intentionally conservative; for richer Atlas mappings you can extend this
        helper with phrase/wildcard/path-specific logic.
        """
        clauses: List[Dict[str, Any]] = []
        for field_name, value in metadata_filter.items():
            if isinstance(value, dict) and "$in" in value:
                clauses.append({"in": {"path": field_name, "value": list(value["$in"])}})
            elif not isinstance(value, dict):
                clauses.append({"equals": {"path": field_name, "value": value}})
        return clauses

    def _project_fields(
        self,
        *,
        include_embedding: bool = False,
        with_vector_score: bool = False,
        with_text_score: bool = False,
    ) -> Dict[str, Any]:
        projection: Dict[str, Any] = {
            "_id": 0,
            "chunk_id": 1,
            "source_id": 1,
            "source_type": 1,
            "title": 1,
            "url": 1,
            self.text_field: 1,
            "domain": 1,
            "entity_tags": 1,
            "section": 1,
            "order": 1,
            "version": 1,
            "metadata": 1,
        }
        if include_embedding:
            projection[self.embedding_field] = 1
        if with_vector_score:
            projection["score"] = {"$meta": "vectorSearchScore"}
        if with_text_score:
            projection["score"] = {"$meta": "searchScore"}
        return projection

    def _as_hit(self, doc: Dict[str, Any], *, channel: str) -> Dict[str, Any]:
        payload = self._normalize_doc(doc)
        payload["_channel"] = channel
        payload["_score"] = self._coerce_float(doc.get("score"), default=0.0)
        return payload

    def _normalize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(doc)
        if self.text_field != "text":
            normalized["text"] = normalized.get(self.text_field, "")
        normalized["text"] = str(normalized.get("text", ""))
        if not isinstance(normalized.get("entity_tags"), list):
            normalized["entity_tags"] = []
        if not isinstance(normalized.get("metadata"), dict):
            normalized["metadata"] = {}
        return normalized

    @staticmethod
    def _prefer_richer_doc(existing: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
        existing_text = str(existing.get("text", ""))
        new_text = str(new_doc.get("text", ""))
        if len(new_text) > len(existing_text):
            return new_doc
        return existing

    @staticmethod
    def _score_bounds(hits: List[Dict[str, Any]]) -> Tuple[float, float]:
        values = [float(hit["_score"]) for hit in hits if hit.get("_score") is not None]
        if not values:
            return (0.0, 0.0)
        return (min(values), max(values))

    def _channel_component(
        self,
        *,
        rank: Optional[int],
        total: int,
        raw_score: Optional[float],
        bounds: Tuple[float, float],
        weight: float,
    ) -> float:
        if rank is None or total <= 0:
            return 0.0

        rank_score = self._rank_to_unit(rank, total)
        raw_score_norm = self._normalize_score(raw_score, bounds, fallback=rank_score)
        return weight * ((0.7 * rank_score) + (0.3 * raw_score_norm))

    @staticmethod
    def _rank_to_unit(rank: int, total: int) -> float:
        if total <= 1:
            return 1.0
        return max(0.0, 1.0 - ((rank - 1) / (total - 1)))

    @staticmethod
    def _normalize_score(
        value: Optional[float],
        bounds: Tuple[float, float],
        *,
        fallback: float,
    ) -> float:
        if value is None:
            return fallback
        low, high = bounds
        if high <= low:
            return 1.0
        return (float(value) - low) / (high - low)

    def _embed_query(self, query_text: str) -> List[float]:
        """Embed query text while supporting different embedder method names."""
        if hasattr(self.embedder, "embed_one"):
            vector = self.embedder.embed_one(query_text)
        elif hasattr(self.embedder, "embed_text"):
            vector = self.embedder.embed_text(query_text)
        elif hasattr(self.embedder, "embed_many"):
            vectors = self.embedder.embed_many([query_text])
            vector = vectors[0] if vectors else []
        else:
            raise TypeError("Embedder must implement embed_one(), embed_text(), or embed_many().")

        if not isinstance(vector, list) or not vector:
            raise ValueError("Query embedding is empty.")
        return [float(v) for v in vector]

    @staticmethod
    def _entity_exact_bonus(query_text: str, raw_entity_tags: Any) -> float:
        if not isinstance(raw_entity_tags, list):
            return 0.0
        query_lc = query_text.strip().lower()
        if not query_lc:
            return 0.0
        for tag in raw_entity_tags:
            if str(tag).strip().lower() == query_lc:
                return 2.0
        return 0.0

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        a_norm = math.sqrt(sum(float(x) * float(x) for x in a))
        b_norm = math.sqrt(sum(float(y) * float(y) for y in b))
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return dot / (a_norm * b_norm)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", text or "")]

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _as_str_list(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for item in values:
            value = str(item).strip()
            if value:
                out.append(value)
        return out

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _unique_non_empty(values: Iterable[Any], *, to_lower: bool) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for raw in values:
            value = str(raw or "").strip()
            if not value:
                continue
            normalized = value.lower() if to_lower else value
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    def _reset_diagnostics(self, *, query_text: str, top_k: int) -> None:
        r = self._index_health_report
        self.last_query_diagnostics = {
            "query_text": query_text,
            "top_k": int(top_k),
            "vector_mode": "unknown",
            "text_mode": "unknown",
            "vector_fallback_used": False,
            "text_fallback_used": False,
            "vector_search_error": None,
            "atlas_search_error": None,
            "mongo_text_error": None,
            "vector_hits": 0,
            "text_hits": 0,
            "final_hits": 0,
            "index_health": {
                "checked": r is not None,
                "healthy": r.is_healthy if r is not None else None,
                "vector_index_found": r.vector_index_found if r is not None else None,
                "vector_index_status": r.vector_index_status if r is not None else None,
                "vector_index_dimensions": r.vector_index_dimensions if r is not None else None,
                "text_index_found": r.text_index_found if r is not None else None,
                "text_index_status": r.text_index_status if r is not None else None,
                "errors": r.errors if r is not None else [],
            },
        }

    def _sync_atlas_preferences_from_health_report(self) -> None:
        r = self._index_health_report
        if r is None:
            self._prefer_atlas_vector = True
            self._prefer_atlas_text = True
            return
        self._prefer_atlas_vector = bool(r.vector_index_found and r.vector_index_status == "READY")
        self._prefer_atlas_text = bool(r.text_index_found and r.text_index_status == "READY")
