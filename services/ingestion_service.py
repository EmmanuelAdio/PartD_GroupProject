from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from schemas.models import Chunk, ChunkRecord, ChunkTags, Document
except ImportError:  # pragma: no cover
    from ..schemas.models import Chunk, ChunkRecord, ChunkTags, Document


class IngestionService:
    """
    Offline ingestion pipeline:
    input -> Document -> Chunks -> Tags -> Embeddings -> MongoDB upsert
    """

    def __init__(
        self,
        repo=None,  # MongoRepo (optional)
        embedder=None,  # EmbeddingService-like object with embed_many()
        tagger=None,  # Optional LLMService for domain/entity extraction
        chunker=None,  # Optional custom chunker implementation
        version: str = "v1",
        json_group_size: int = 30,
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required.")
        self.repo = repo
        self.embedder = embedder
        self.tagger = tagger
        self.chunker = chunker
        self.version = version
        self.json_group_size = json_group_size

    # ---------------------------
    # Public entrypoints
    # ---------------------------

    def ingest_json(
        self,
        data: Union[Dict[str, Any], List[Any]],
        source_id: str,
        title: Optional[str] = None,
    ) -> tuple[int, List[ChunkRecord]]:
        doc = self._normalize_json(data=data, source_id=source_id, title=title)
        return self._ingest_document(doc)

    def ingest_text(
        self,
        text: str,
        source_id: str,
        title: Optional[str] = None,
    ) -> tuple[int, List[ChunkRecord]]:
        doc = Document(source_id=source_id, source_type="txt", title=title, text=text, raw={})
        return self._ingest_document(doc)

    def ingest_web_qna(
        self,
        qna_list: List[Dict[str, str]],
        source_id: str,
        url: Optional[str] = None,
    ) -> tuple[int, List[ChunkRecord]]:
        doc = self._normalize_qna(qna_list=qna_list, source_id=source_id, url=url)
        return self._ingest_document(doc)

    # ---------------------------
    # Core pipeline
    # ---------------------------

    def _ingest_document(self, doc: Document) -> tuple[int, List[ChunkRecord]]:
        chunks = self._segment(doc)

        # Skip chunks that already exist in DB to avoid unnecessary LLM/embedding cost.
        if self.repo is not None and hasattr(self.repo, "get_existing_chunk_ids"):
            existing_ids = self.repo.get_existing_chunk_ids([c.chunk_id for c in chunks])
            if existing_ids:
                chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not chunks:
            return 0, []

        tagged_pairs = [(chunk, self._tag_chunk(doc, chunk)) for chunk in chunks]

        texts = [chunk.text for chunk, _ in tagged_pairs]
        vectors = self.embedder.embed_many(texts)
        if len(vectors) != len(tagged_pairs):
            raise ValueError(
                f"Embedding count mismatch: got {len(vectors)} vectors for {len(tagged_pairs)} chunks."
            )

        records = []
        for (chunk, tags), vector in zip(tagged_pairs, vectors):
            record = self._build_record(doc=doc, chunk=chunk, tags=tags, embedding=vector)
            records.append(record)

        if self.repo is not None:
            self.repo.upsert_chunks(records)
        return len(records), records

    # ---------------------------
    # Normalizers
    # ---------------------------

    def _normalize_json(
        self,
        data: Union[Dict[str, Any], List[Any]],
        source_id: str,
        title: Optional[str],
    ) -> Document:
        text = self._json_to_text(data)
        return Document(
            source_id=source_id,
            source_type="json",
            title=title,
            text=text,
            raw={"json": data},
        )

    def _normalize_qna(self, qna_list: List[Dict[str, str]], source_id: str, url: Optional[str]) -> Document:
        lines = []
        for item in qna_list:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q:
                lines.append(f"Q: {q}\nA: {a}")
        text = "\n\n---\n\n".join(lines)
        return Document(
            source_id=source_id,
            source_type="web",
            title="Web Q&A",
            url=url,
            text=text,
            raw={"qna": qna_list},
        )

    def _json_to_text(self, obj: Any, prefix: str = "") -> str:
        lines: List[str] = []

        def walk(x: Any, path: str) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]")
            else:
                val = str(x).strip()
                if val:
                    lines.append(f"{path}: {val}")

        walk(obj, prefix)
        return "\n".join(lines)

    # ---------------------------
    # Segmenter (chunking)
    # ---------------------------

    def _segment(self, doc: Document) -> List[Chunk]:
        if self.chunker is not None:
            return self.chunker.segment(doc)

        if doc.source_type == "web":
            blocks = [b.strip() for b in doc.text.split("---") if b.strip()]
            return [self._make_chunk(doc.source_id, b, order=i, section="qna") for i, b in enumerate(blocks)]

        if doc.source_type == "json":
            return self._line_group_chunk(
                source_id=doc.source_id,
                text=doc.text,
                group_size=self.json_group_size,
                section="json_fields",
            )

        return self._line_group_chunk(doc.source_id, doc.text, group_size=40, section="text")

    def _line_group_chunk(self, source_id: str, text: str, group_size: int, section: str) -> List[Chunk]:
        lines = [line for line in text.splitlines() if line.strip()]
        chunks: List[Chunk] = []
        for i in range(0, len(lines), group_size):
            block = "\n".join(lines[i : i + group_size])
            chunks.append(self._make_chunk(source_id, block, order=len(chunks), section=section))
        return chunks

    def _make_chunk(self, source_id: str, text: str, order: int, section: Optional[str]) -> Chunk:
        chunk_id = self._stable_id(f"{source_id}:{order}:{text[:120]}")
        return Chunk(chunk_id=chunk_id, source_id=source_id, text=text, order=order, section=section)

    # ---------------------------
    # Tagging (LLM + heuristic fallback)
    # ---------------------------

    def _tag_chunk(self, doc: Document, chunk: Chunk) -> ChunkTags:
        hint_domain = self._heuristic_domain(doc, chunk)
        heuristic_entities = self._heuristic_entities(chunk.text)
        heuristic_key_fields = self._extract_key_fields(chunk.text)

        fallback = ChunkTags(
            domain=hint_domain or "unknown",
            entity_tags=heuristic_entities,
            key_fields=heuristic_key_fields,
            confidence=0.35,
        )

        if self.tagger is None:
            return fallback

        try:
            llm_raw = self._tag_with_llm(doc=doc, chunk=chunk, hint_domain=hint_domain)
            llm_tags = self._coerce_chunk_tags(llm_raw, fallback=fallback)
            return ChunkTags(
                domain=llm_tags.domain or fallback.domain,
                entity_tags=llm_tags.entity_tags or fallback.entity_tags,
                key_fields={**heuristic_key_fields, **llm_tags.key_fields},
                confidence=max(fallback.confidence, llm_tags.confidence),
            )
        except Exception:
            # Keep ingestion robust in case LLM tagging fails.
            return fallback

    def _tag_with_llm(self, doc: Document, chunk: Chunk, hint_domain: Optional[str]) -> Any:
        if hasattr(self.tagger, "tag_chunk"):
            return self.tagger.tag_chunk(
                text=chunk.text,
                hint_domain=hint_domain,
                source_id=doc.source_id,
                source_type=doc.source_type,
            )
        if hasattr(self.tagger, "tag"):
            return self.tagger.tag(text=chunk.text, hint_domain=hint_domain)
        raise TypeError("tagger must implement tag_chunk() or tag().")

    def _coerce_chunk_tags(self, raw: Any, fallback: ChunkTags) -> ChunkTags:
        if isinstance(raw, ChunkTags):
            return raw
        if not isinstance(raw, dict):
            return fallback

        key_fields = raw.get("key_fields")
        if not isinstance(key_fields, dict):
            key_fields = {}

        confidence = raw.get("confidence", fallback.confidence)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = fallback.confidence
        confidence = max(0.0, min(1.0, confidence))

        domain = str(raw.get("domain") or fallback.domain).strip().lower()
        if not domain:
            domain = fallback.domain

        return ChunkTags(
            domain=domain,
            entity_tags=self._normalize_entities(raw.get("entity_tags")),
            key_fields=key_fields,
            confidence=confidence,
        )

    def _heuristic_domain(self, doc: Document, chunk: Chunk) -> Optional[str]:
        source = doc.source_id.lower()
        text = chunk.text.lower()

        if "accommodation" in source:
            return "accommodation"
        if "course" in source or "ug_" in source:
            return "courses"
        if "sport" in source:
            return "sports"
        if "offer" in source or "faq" in source or "entry" in source:
            return "admissions"

        if any(k in text for k in ["ensuite", "self_catered", "launderette", "tenancy_weeks", "hall"]):
            return "accommodation"
        if any(k in text for k in ["ucas", "entry requirements", "modules", "course", "undergraduate"]):
            return "courses"
        if any(k in text for k in ["sport", "athletic", "club", "gym"]):
            return "sports"
        if any(k in text for k in ["contextual offer", "offer", "admissions", "application"]):
            return "admissions"
        return None

    def _heuristic_entities(self, text: str) -> List[str]:
        entities: List[str] = []
        seen = set()

        for line in text.splitlines():
            if ":" not in line:
                continue
            path, value = line.split(":", 1)
            value = value.strip()
            path_lower = path.lower()
            if not value:
                continue
            if any(marker in path_lower for marker in ["name", "title", "course", "question", "hall", "sport"]):
                key = value.lower()
                if key not in seen:
                    seen.add(key)
                    entities.append(value)
                    if len(entities) >= 12:
                        return entities

        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z&'\-]+){0,3}\b", text):
            key = match.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(match)
            if len(entities) >= 12:
                break

        return entities

    def _normalize_entities(self, raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        seen = set()
        for item in raw:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= 12:
                break
        return out

    def _extract_key_fields(self, text: str) -> Dict[str, Any]:
        lines = text.splitlines()
        lower = text.lower()
        key_fields: Dict[str, Any] = {
            "contains_price": "per_week_gbp" in lower or "total_contract_gbp" in lower or "fees" in lower,
            "contains_entry_requirements": "entry requirements" in lower or "ucas" in lower,
            "contains_location": ".address" in lower or "campus" in lower,
            "contains_question_answer": "question" in lower and "answer" in lower,
        }

        sample_prices = [line.strip() for line in lines if "per_week_gbp" in line or "fees" in line][:3]
        if sample_prices:
            key_fields["sample_price_lines"] = sample_prices

        for line in lines:
            if ".name:" in line:
                key_fields["sample_name"] = line.split(":", 1)[1].strip()
                break
        return key_fields

    # ---------------------------
    # Record builder + DB
    # ---------------------------

    def _build_record(self, doc: Document, chunk: Chunk, tags: ChunkTags, embedding: List[float]) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=chunk.chunk_id,
            source_id=doc.source_id,
            source_type=doc.source_type,
            title=doc.title,
            url=doc.url,
            text=chunk.text,
            embedding=embedding,
            domain=tags.domain,
            entity_tags=tags.entity_tags,
            section=chunk.section,
            order=chunk.order,
            metadata={"key_fields": tags.key_fields, "confidence": tags.confidence},
            version=self.version,
        )

    @staticmethod
    def _stable_id(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()


class OpenAIEmbedder:
    """Temporary embedder for local ingestion testing.

    Uses OpenAI embeddings API and requires OPENAI_API_KEY.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is not installed. Install it with: pip install openai"
            )
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
