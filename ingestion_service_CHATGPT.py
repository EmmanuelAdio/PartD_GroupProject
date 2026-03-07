from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


try:
    from openai import OpenAI
    from dotenv import load_dotenv


    load_dotenv()
except ImportError:  # pragma: no cover
    OpenAI = None


SourceType = Literal["json", "pdf", "web", "txt"]


class Document(BaseModel):
    source_id: str
    source_type: SourceType
    title: Optional[str] = None
    url: Optional[str] = None
    text: str
    raw: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    section: Optional[str] = None
    order: int = 0
    raw_path: Optional[str] = None


class ChunkTags(BaseModel):
    domain: str
    entity_tags: List[str] = Field(default_factory=list)
    key_fields: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class ChunkRecord(BaseModel):
    chunk_id: str
    source_id: str
    source_type: SourceType
    title: Optional[str] = None
    url: Optional[str] = None
    text: str
    embedding: List[float]
    domain: str
    entity_tags: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    order: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: str = "v1"


class OpenAIEmbedder:
    """Temporary embedder for local ingestion testing.

    Uses OpenAI's embeddings API in the same spirit as your earlier MVP.
    Requires OPENAI_API_KEY to be set in the environment.
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


class IngestionService:
    """Offline ingestion pipeline for JSON-first testing.

    For this test harness we stop right before any MongoDB upsert.
    The final output is a list of ChunkRecord objects, exactly the shape that
    would be upserted later.
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        version: str = "v1",
        json_group_size: int = 25,
    ) -> None:
        self.embedder = embedder
        self.version = version
        self.json_group_size = json_group_size

    # ---------------------------------------------------------------------
    # Public entrypoint for JSON ingestion
    # ---------------------------------------------------------------------
    def ingest_json(
        self,
        data: Union[Dict[str, Any], List[Any]],
        source_id: str,
        title: Optional[str] = None,
    ) -> List[ChunkRecord]:
        doc = self._normalize_json(data=data, source_id=source_id, title=title)
        return self._prepare_records(doc)

    # ---------------------------------------------------------------------
    # Core pipeline (stops before DB upsert)
    # ---------------------------------------------------------------------
    def _prepare_records(self, doc: Document) -> List[ChunkRecord]:
        chunks = self._segment(doc)
        tagged_pairs = [(chunk, self._tag_chunk(doc, chunk)) for chunk in chunks]
        embeddings = self.embedder.embed_many([chunk.text for chunk, _ in tagged_pairs])

        records: List[ChunkRecord] = []
        for (chunk, tags), embedding in zip(tagged_pairs, embeddings):
            records.append(self._build_record(doc, chunk, tags, embedding))
        return records

    # ---------------------------------------------------------------------
    # Normalizer
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Chunking / segmentation
    # ---------------------------------------------------------------------
    def _segment(self, doc: Document) -> List[Chunk]:
        if doc.source_type != "json":
            raise ValueError("This test harness currently only supports JSON ingestion.")
        return self._segment_json_by_items(doc)

    def _segment_json_by_items(self, doc: Document) -> List[Chunk]:
        """Prefer one hall per source chunk group when the JSON is a list of halls.

        Each hall is turned into line-based mini-sections, then grouped to avoid
        over-long chunks while still preserving hall identity.
        """
        raw_json = doc.raw.get("json")
        chunks: List[Chunk] = []

        if isinstance(raw_json, list):
            for item_index, item in enumerate(raw_json):
                hall_name = None
                if isinstance(item, dict):
                    hall_name = item.get("name")
                item_text = self._json_to_text(item, prefix=f"item[{item_index}]")
                item_lines = [ln for ln in item_text.splitlines() if ln.strip()]
                section_name = f"json_item:{hall_name or item_index}"
                for i in range(0, len(item_lines), self.json_group_size):
                    block = "\n".join(item_lines[i : i + self.json_group_size])
                    order = len(chunks)
                    chunks.append(
                        self._make_chunk(
                            source_id=doc.source_id,
                            text=block,
                            order=order,
                            section=section_name,
                            raw_path=f"item[{item_index}]",
                        )
                    )
            return chunks

        # Fallback for a single JSON object
        lines = [ln for ln in doc.text.splitlines() if ln.strip()]
        for i in range(0, len(lines), self.json_group_size):
            block = "\n".join(lines[i : i + self.json_group_size])
            chunks.append(
                self._make_chunk(
                    source_id=doc.source_id,
                    text=block,
                    order=len(chunks),
                    section="json_object",
                    raw_path=None,
                )
            )
        return chunks

    def _make_chunk(
        self,
        source_id: str,
        text: str,
        order: int,
        section: Optional[str],
        raw_path: Optional[str],
    ) -> Chunk:
        chunk_id = self._stable_id(f"{source_id}:{order}:{text[:120]}")
        return Chunk(
            chunk_id=chunk_id,
            source_id=source_id,
            text=text,
            order=order,
            section=section,
            raw_path=raw_path,
        )

    # ---------------------------------------------------------------------
    # Tagging (simple heuristic MVP)
    # ---------------------------------------------------------------------
    def _tag_chunk(self, doc: Document, chunk: Chunk) -> ChunkTags:
        domain = self._heuristic_domain(chunk.text)
        entity_tags = self._heuristic_entities(chunk.text)
        key_fields = self._extract_key_fields(chunk.text)
        return ChunkTags(
            domain=domain,
            entity_tags=entity_tags,
            key_fields=key_fields,
            confidence=0.45,
        )

    def _heuristic_domain(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["catering", "launderette", "ensuite", "hall", "tenancy_weeks"]):
            return "accommodation"
        if any(k in t for k in ["module", "ucas", "entry requirements"]):
            return "courses"
        return "unknown"

    def _heuristic_entities(self, text: str) -> List[str]:
        entities: List[str] = []
        for line in text.splitlines():
            lowered = line.lower()
            if ".name:" in lowered:
                value = line.split(":", 1)[1].strip()
                if value and value not in entities:
                    entities.append(value)
                    if len(entities) >= 5:
                        break
        return entities

    def _extract_key_fields(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "contains_price": "per_week_gbp" in text or "total_contract_gbp" in text,
            "contains_address": ".address:" in text,
            "contains_contact": ".contacts." in text,
        }
        for line in text.splitlines():
            if "per_week_gbp" in line:
                out.setdefault("sample_price_lines", []).append(line.strip())
            if len(out.get("sample_price_lines", [])) >= 3:
                break
        return out

    # ---------------------------------------------------------------------
    # Final record builder
    # ---------------------------------------------------------------------
    def _build_record(
        self,
        doc: Document,
        chunk: Chunk,
        tags: ChunkTags,
        embedding: List[float],
    ) -> ChunkRecord:
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
            metadata={
                "raw_path": chunk.raw_path,
                "key_fields": tags.key_fields,
                "confidence": tags.confidence,
            },
            version=self.version,
        )

    def _stable_id(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------------
# Local test harness
# -------------------------------------------------------------------------

def run_json_ingestion_tests(json_path: str) -> None:
    """Runs step-by-step tests for the JSON ingestion pipeline.

    Stops before MongoDB upsert and prints the final ChunkRecord payloads.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    service = IngestionService(embedder=OpenAIEmbedder(), version="test-v1", json_group_size=25)

    print("=" * 80)
    print("TEST 1: LOAD JSON")
    print(f"Loaded file: {path}")
    print(f"Top-level type: {type(data).__name__}")
    print(f"Top-level length: {len(data) if isinstance(data, list) else 'N/A'}")

    print("\n" + "=" * 80)
    print("TEST 2: NORMALIZE JSON -> DOCUMENT")
    doc = service._normalize_json(data=data, source_id="accommodation_halls", title="Accommodation Halls")
    print(f"Document source_id: {doc.source_id}")
    print(f"Document source_type: {doc.source_type}")
    print(f"Document title: {doc.title}")
    print(f"Normalized text preview:\n{doc.text[:1200]}\n")

    print("=" * 80)
    print("TEST 3: SEGMENT DOCUMENT -> CHUNKS")
    chunks = service._segment(doc)
    print(f"Number of chunks created: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"\nChunk order={chunk.order} id={chunk.chunk_id} section={chunk.section} raw_path={chunk.raw_path}")
        print(chunk.text[:600])
        print("-" * 60)

    print("\n" + "=" * 80)
    print("TEST 4: TAG FIRST 3 CHUNKS")
    for chunk in chunks[:3]:
        tags = service._tag_chunk(doc, chunk)
        print(f"\nChunk {chunk.order} tags:")
        print(tags.model_dump_json(indent=2))

    print("\n" + "=" * 80)
    print("TEST 5: EMBED FIRST 3 CHUNKS")
    sample_texts = [chunk.text for chunk in chunks[:3]]
    sample_embeddings = service.embedder.embed_many(sample_texts)
    for i, emb in enumerate(sample_embeddings):
        print(f"Chunk {i} embedding dimension: {len(emb)}")
        print(f"Embedding preview: {emb[:8]}")

    print("\n" + "=" * 80)
    print("TEST 6: BUILD FINAL RECORDS (PRE-UPSERT OUTPUT)")
    final_records = service.ingest_json(data=data, source_id="accommodation_halls", title="Accommodation Halls")
    print(f"Total final records built: {len(final_records)}")

    preview_count = min(3, len(final_records))
    for i in range(preview_count):
        print(f"\nFinal record {i + 1}/{preview_count}:")
        print(final_records[i].model_dump_json(indent=2)[:4000])

    output_path = path.parent / "accommodation_chunk_records_preview.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in final_records], f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Saved full pre-upsert record preview to: {output_path}")
    print("These records are the exact shape you would upsert into MongoDB next.")


if __name__ == "__main__":
    # Adjust this path if you move the test file.
    run_json_ingestion_tests("C:\\Users\\Emman\\OneDrive\\Documents\\GitHub\\PartD_GroupProject\\data\\accommodation_halls.json")
