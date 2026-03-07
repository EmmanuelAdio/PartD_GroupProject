from typing import Any, Dict, Iterable, List, Optional, Union
import hashlib
import json
import os
from pathlib import Path

from langchain_openai import OpenAI
from ..schemas.models import Document, Chunk, ChunkTags, ChunkRecord

class IngestionService:
    """
    Offline pipeline:
    input -> Document -> Chunks -> Tags -> Embeddings -> MongoDB upsert
    """

    def __init__(
        self,
        repo,                 # MongoRepo (your wrapper around MongoDB)
        embedder,             # EmbeddingService
        tagger=None,          # Optional LLMTagger for domain/entity extraction
        chunker=None,         # Optional chunker implementation
        version: str = "v1",
    ):
        self.repo = repo
        self.embedder = embedder
        self.tagger = tagger
        self.chunker = chunker
        self.version = version

    # ---------------------------
    # Public entrypoints
    # ---------------------------

    def ingest_json(self, data: Union[Dict[str, Any], List[Any]], source_id: str, title: Optional[str] = None) -> tuple[int, List[ChunkRecord]]:
        doc = self._normalize_json(data=data, source_id=source_id, title=title)
        return self._ingest_document(doc)

    def ingest_text(self, text: str, source_id: str, title: Optional[str] = None) -> int:
        doc = Document(source_id=source_id, source_type="txt", title=title, text=text, raw={})
        return self._ingest_document(doc)

    def ingest_web_qna(self, qna_list: List[Dict[str, str]], source_id: str, url: Optional[str] = None) -> int:
        doc = self._normalize_qna(qna_list=qna_list, source_id=source_id, url=url)
        return self._ingest_document(doc)

    # You can add ingest_pdf(path) later when you plug in a PDF parser.

    # ---------------------------
    # Core pipeline
    # ---------------------------

    def _ingest_document(self, doc: "Document") -> tuple[int, List[ChunkRecord]]:
        # 1) segment
        chunks = self._segment(doc)

        # 2) tag (domain/entity/section)
        tagged = []
        for c in chunks:
            tags = self._tag_chunk(doc, c)
            tagged.append((c, tags))

        # 3) embed (batch for speed)
        texts = [c.text for (c, _) in tagged]
        vectors = self.embedder.embed_many(texts)

        # 4) build DB records
        records = []
        for (i, ((chunk, tags), vec)) in enumerate(zip(tagged, vectors)):
            record = self._build_record(doc, chunk, tags, vec)
            records.append(record)

        # 5) upsert
        # self.repo.upsert_chunks(records)
        return len(records), records
    
    # ---------------------------
    # Normalizers
    # ---------------------------

    def _normalize_json(self, data: Union[Dict[str, Any], List[Any]], source_id: str, title: Optional[str]) -> "Document":
        """
        Convert arbitrary JSON into a Document.text that is meaningful for retrieval.
        """
        # Simple strategy: pretty stringify + keep raw
        text = self._json_to_text(data)
        return Document(source_id=source_id, source_type="json", title=title, text=text, raw={"json": data})

    def _normalize_qna(self, qna_list: List[Dict[str, str]], source_id: str, url: Optional[str]) -> "Document":
        """
        Q/A documents should be formatted so each Q/A is self-contained.
        """
        lines = []
        for item in qna_list:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q:
                lines.append(f"Q: {q}\nA: {a}")
        text = "\n\n---\n\n".join(lines)
        return Document(source_id=source_id, source_type="web", title="Web Q&A", url=url, text=text, raw={"qna": qna_list})

    def _json_to_text(self, obj: Any, prefix: str = "") -> str:
        """
        Better than flattening everything into one sentence:
        keep key paths so retrieval can match sections.
        """
        lines: List[str] = []

        def walk(x: Any, path: str):
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

    def _segment(self, doc: "Document") -> List["Chunk"]:
        """
        Swap this strategy per source_type.
        """
        if doc.source_type == "web":
            # Q/A already separated by ---; chunk per block
            blocks = [b.strip() for b in doc.text.split("---") if b.strip()]
            return [self._make_chunk(doc.source_id, b, order=i, section="qna") for i, b in enumerate(blocks)]

        if doc.source_type == "json":
            # Chunk by groups of lines (simple first version)
            return self._line_group_chunk(doc.source_id, doc.text, group_size=30, section="json_fields")

        # default: use a text splitter if you want
        return self._line_group_chunk(doc.source_id, doc.text, group_size=40, section="text")

    def _line_group_chunk(self, source_id: str, text: str, group_size: int, section: str) -> List["Chunk"]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        chunks: List[Chunk] = []
        for i in range(0, len(lines), group_size):
            block = "\n".join(lines[i:i+group_size])
            chunks.append(self._make_chunk(source_id, block, order=len(chunks), section=section))
        return chunks

    def _make_chunk(self, source_id: str, text: str, order: int, section: Optional[str]) -> "Chunk":
        chunk_id = self._stable_id(f"{source_id}:{order}:{text[:80]}")
        return Chunk(chunk_id=chunk_id, source_id=source_id, text=text, order=order, section=section)

    # ---------------------------
    # Tagging (domain/entities)
    # ---------------------------

    def _tag_chunk(self, doc: "Document", chunk: "Chunk") -> "ChunkTags":
        """
        Make this dynamic:
        - heuristics first (fast)
        - LLM tagger second (better)
        """
        # heuristic hinting
        domain_guess = self._heuristic_domain(doc, chunk)

        if self.tagger:
            # Ask an LLM to tag it, but provide the heuristic as a hint
            return self.tagger.tag(text=chunk.text, hint_domain=domain_guess)

        # fallback: heuristic only
        return ChunkTags(domain=domain_guess or "unknown", entity_tags=self._heuristic_entities(chunk.text), confidence=0.3)

    def _heuristic_domain(self, doc: "Document", chunk: "Chunk") -> Optional[str]:
        t = chunk.text.lower()
        if "catering" in t or "launderette" in t or "ensuite" in t or "self-catered" in t:
            return "accommodation"
        if "entry requirements" in t or "ucas" in t or "modules" in t:
            return "courses"
        return None

    def _heuristic_entities(self, text: str) -> List[str]:
        # Very basic placeholder — later you’ll use gazetteer/LLM NER
        # Example: match capitalised two-word phrases (not robust, just MVP)
        return []

    # ---------------------------
    # Record builder + DB
    # ---------------------------

    def _build_record(self, doc: "Document", chunk: "Chunk", tags: "ChunkTags", embedding: List[float]) -> "ChunkRecord":
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
            version=self.version
        )

    def _stable_id(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    


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