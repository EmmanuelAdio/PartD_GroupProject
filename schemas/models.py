from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal

SourceType = Literal["json", "pdf", "web", "txt"]

class Document(BaseModel):
    """A raw source document before chunking — one entry per source file or web page."""

    source_id: str
    source_type: SourceType
    title: Optional[str] = None
    url: Optional[str] = None
    text: str
    raw: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    """An intermediate text chunk produced by the ingestion splitter before tagging and embedding."""

    chunk_id: str
    source_id: str
    text: str
    section: Optional[str] = None
    order: int = 0
    raw_path: Optional[str] = None  # e.g. "facilities[3]" or "qna[12]"

class ChunkTags(BaseModel):
    """Domain classification and entity tags assigned to a chunk by the tagger."""

    domain: str
    entity_tags: List[str] = Field(default_factory=list)
    key_fields: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0

class ChunkRecord(BaseModel):
    """A fully processed chunk ready to be written to MongoDB, including its embedding vector."""

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
    """A ranked evidence chunk returned by RetrieverService to the AnswererAgent.

    Carries both the raw text and retrieval scores so the answerer and evaluator
    can cite specific chunks and assess grounding quality.
    """

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


class AnswerCitation(BaseModel):
    """A reference linking a claim in the final answer back to a specific evidence chunk."""

    evidence_id: int = Field(ge=1)
    chunk_id: str
    source_id: str
    source_type: Optional[SourceType] = None
    title: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None


class AnswerResult(BaseModel):
    """Final answer payload produced by AnswererAgent and consumed by the orchestrator.

    ``grounded`` and ``confidence`` are set by the answerer; the evaluator may
    override the effective verdict if grounding checks fail.
    """

    answer: str
    grounded: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: List[AnswerCitation] = Field(default_factory=list)
    used_evidence_count: int = 0
    fallback_used: bool = False


EvaluationVerdict = Literal["pass", "revise", "ask_clarification", "fallback"]


class RuleCheckResult(BaseModel):
    """Output of the EvaluatorAgent's deterministic rule-check layer.

    Always runs before the optional LLM judge. Populated with boolean quality
    signals and optional suggested filters for the orchestrator retry path.
    """

    grounded: bool = True
    relevant: bool = True
    clear: bool = True
    safe: bool = True
    issues: List[str] = Field(default_factory=list)

    no_evidence: bool = False
    weak_evidence: bool = False
    ambiguous_query: bool = False
    needs_llm_judge: bool = False

    suggested_action: Optional[str] = None
    suggested_filters: Optional[Dict[str, Any]] = None
    clarification_question: Optional[str] = None
    notes: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class LLMJudgeResult(BaseModel):
    """Optional second-stage evaluator output from the LLM judge.

    Only populated for borderline cases where rule checks are inconclusive.
    Fields are Optional so the orchestrator can merge only the non-null signals
    with the rule-check result.
    """

    grounded: Optional[bool] = None
    relevant: Optional[bool] = None
    clear: Optional[bool] = None
    safe: Optional[bool] = None
    issues: List[str] = Field(default_factory=list)
    suggested_verdict: Optional[EvaluationVerdict] = None
    suggested_action: Optional[str] = None
    suggested_filters: Optional[Dict[str, Any]] = None
    clarification_question: Optional[str] = None
    notes: Optional[str] = None


class EvaluationResult(BaseModel):
    """Final evaluator verdict consumed by the orchestration policy.

    ``verdict`` drives the runtime action: ``pass`` serves the answer as-is,
    ``revise`` triggers one retrieval/answer retry, ``ask_clarification`` returns
    a clarifying question, and ``fallback`` returns a safe generic response.
    """

    verdict: EvaluationVerdict
    grounded: bool
    relevant: bool
    clear: bool
    safe: bool
    issues: List[str] = Field(default_factory=list)
    suggested_action: Optional[str] = None
    suggested_filters: Optional[Dict[str, Any]] = None
    clarification_question: Optional[str] = None
    notes: Optional[str] = None
