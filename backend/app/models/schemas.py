from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ServiceStatus(BaseModel):
    configured: bool
    status: str


class HealthResponse(BaseModel):
    api_status: str
    ollama: ServiceStatus
    qdrant: ServiceStatus
    tigergraph: ServiceStatus


# ---------- Phase 2: RAG schemas ----------


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3


class Citation(BaseModel):
    chunk_id: str
    source: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float


class RAGQueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float


# ---------- Phase 3: Ingestion schemas ----------


class IngestFileResult(BaseModel):
    file: str
    pages: int
    chunks: int
    status: str


class IngestResponse(BaseModel):
    files_processed: int
    total_chunks: int
    results: List[IngestFileResult]


class IngestStatusResponse(BaseModel):
    configured: bool
    status: str
    collection_name: str | None = None
    points_count: int | None = None
    url: str | None = None

# ---------- Entity Extraction Schema ----------

class ExtractedEntity(BaseModel):
    name: str
    type: str

class EntityExtraction(BaseModel):
    entities: List[ExtractedEntity]
    collection_name: str
    vector_dimension: Optional[int] = None
    points_count: Optional[int] = None
    status: Optional[str] = None
    storage_path: Optional[str] = None
    error: Optional[str] = None


# ---------- Phase 4: GraphRAG schemas ----------

class GraphRAGQueryResponse(BaseModel):
    answer: str
    citations: List[str]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    latency_ms: float

    # Retrieval diagnostics
    extracted_phrases: List[str] = []
    linked_entities: List[str] = []
    graph_nodes_returned: int = 0
    retrieved_chunk_count: int = 0
    unique_source_count: int = 0


# ---------- Phase 5: Agentic GraphRAG schemas ----------

class OrchestratorAction(BaseModel):
    action: str
    target: str
    reason: str
    fallback_used: bool = False

class EvidenceEvalResult(BaseModel):
    sufficient: bool
    confidence: float
    missing_information: List[str]
    contradiction_detected: bool
    recommended_next_action: str

class NormalizedToolResult(BaseModel):
    tool: str
    success: bool
    results: List[Dict[str, Any]]
    evidence: List[str]
    source_ids: List[str]
    latency_ms: int

class AgenticTraceStep(BaseModel):
    step: int
    action: str
    target: str
    reason: str
    start_time: str
    timeout: float
    status: str
    latency_ms: int
    input_size: int
    output_size: int
    candidate_count: int
    result_count: int
    think_enabled: bool
    fallback_used: bool = False

class StrategyChange(BaseModel):
    initial_strategy: str
    reason: str
    new_strategy: str

class AgenticTrace(BaseModel):
    question: str
    steps: List[AgenticTraceStep]
    tools_called: List[str]
    strategy_changes: List[StrategyChange]
    evidence_count: int
    citations: List[str]
    stop_reason: str
    total_steps: int
    total_latency_ms: int

class AgenticQueryResponse(BaseModel):
    answer: str
    citations: List[str]
    trace: AgenticTrace
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    total_steps: int
    latency_ms: int
    stop_reason: str

class EvaluationResultSchema(BaseModel):
    result_id: str
    evaluator_version: str
    evaluator_model: str
    evaluation_method: str
    status: str
    correctness: Optional[float] = None
    completeness: Optional[float] = None
    groundedness: Optional[float] = None
    eval_details: Dict[str, Any]

class CorrectnessEval(BaseModel):
    correct: bool
    reason: str = ""

class GroundednessEval(BaseModel):
    grounded: bool
    reason: str = ""
