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