"""
API routes for the Traditional RAG pipeline.

Phase 3: Uses Qdrant-backed vector store instead of in-memory retriever.
Falls back to synthetic corpus if Qdrant collection is empty.
"""

from fastapi import APIRouter
from app.models.schemas import RAGQueryRequest, RAGQueryResponse, GraphRAGQueryResponse
from app.rag.pipeline import run_rag_pipeline
from app.api._shared_store import get_vector_store
from app.rag.retriever import build_default_retriever

router = APIRouter(prefix="/api/query", tags=["rag"])

# Keep the in-memory retriever as a fallback for empty Qdrant
_fallback_retriever = build_default_retriever()


@router.post("/rag", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Run a Traditional RAG query.

    Uses Qdrant if it has vectors; otherwise falls back to the
    Phase 2 in-memory synthetic corpus retriever.
    """
    store = get_vector_store()

    if store.count() > 0:
        retriever = store  # QdrantVectorStore has .retrieve()
    else:
        retriever = _fallback_retriever

    result = await run_rag_pipeline(
        question=request.question,
        retriever=retriever,
        top_k=request.top_k,
    )
    return RAGQueryResponse(**result)

@router.post("/graphrag", response_model=GraphRAGQueryResponse)
async def graphrag_query(request: RAGQueryRequest):
    """Run a GraphRAG query."""
    from app.graph.tigergraph_client import TigerGraphClient
    from app.graph.graphrag_pipeline import run_graphrag

    client = TigerGraphClient()
    result = await run_graphrag(
        question=request.question,
        client=client,
        vector_store=get_vector_store()
    )
    return GraphRAGQueryResponse(**result)
