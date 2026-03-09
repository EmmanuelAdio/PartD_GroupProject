from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal

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
    raw_path: Optional[str] = None  # e.g. "facilities[3]" or "qna[12]"

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


class RetrievalQuery(BaseModel):
    """Structured query contract passed from Processor Agent to RetrieverService."""

    query_text: str
    top_k: int = Field(default=8, ge=1, le=50)

    # Metadata filters used during both vector and lexical retrieval.
    domain: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    entity_tags: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    sections: List[str] = Field(default_factory=list)

    # Optional filters that may be useful for orchestration.
    source_ids: List[str] = Field(default_factory=list)
    version: Optional[str] = None

    # Optional candidate sizes for each retrieval channel.
    vector_k: Optional[int] = Field(default=None, ge=1, le=250)
    text_k: Optional[int] = Field(default=None, ge=1, le=250)


class EvidenceItem(BaseModel):
    """Final evidence payload returned to the Answerer Agent."""

    chunk_id: str
    source_id: str
    source_type: Optional[SourceType] = None
    title: Optional[str] = None
    url: Optional[str] = None

    text: str
    domain: Optional[str] = None
    entity_tags: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    order: int = 0
    version: Optional[str] = None

    score: float = 0.0
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    retrieval_channels: List[Literal["vector", "text"]] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)
