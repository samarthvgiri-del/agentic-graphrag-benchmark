"""
Traditional RAG pipeline.

Question → embed → retrieve top-k → build prompt → Ollama generate → return answer + citations.

Phase 3: retriever can be either InMemoryRetriever or QdrantVectorStore
(both expose a .retrieve(query, top_k) method).
"""

import time
import logging
from typing import Any, Dict, Protocol, List, Tuple

from app.llm.ollama_client import ollama_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise research assistant. "
    "Answer the user's question using ONLY the evidence provided below. "
    "If the evidence is insufficient to answer the question, explicitly state: "
    "'The provided evidence is insufficient to answer this question.' "
    "Never fabricate sources, facts, or citations. "
    "Cite evidence by referring to the source document names provided."
)


class Retriever(Protocol):
    """Protocol that both InMemoryRetriever and QdrantVectorStore satisfy."""
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]: ...


def _build_context_block(retrieved: list) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for idx, (chunk, score) in enumerate(retrieved, 1):
        lines.append(
            f"[{idx}] (Source: {chunk['source']}, Chunk: {chunk['chunk_id']}, "
            f"Score: {score:.3f})\n{chunk['text']}"
        )
    return "\n\n".join(lines)


async def run_rag_pipeline(
    question: str,
    retriever: Retriever,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Execute the full traditional RAG pipeline and return structured output."""
    t0 = time.perf_counter()

    # 1. Retrieve
    retrieved = retriever.retrieve(question, top_k=top_k)

    # 2. Build augmented prompt
    context_block = _build_context_block(retrieved)
    augmented_prompt = (
        f"### Evidence\n{context_block}\n\n"
        f"### Question\n{question}\n\n"
        f"### Answer"
    )

    from app.llm.factory import get_llm_provider
    llm_client = get_llm_provider()

    # 3. Generate via LLM
    answer = await llm_client.generate(
        prompt=augmented_prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
        think=False,
        num_predict=800
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 4. Structure response
    citations = [
        {"chunk_id": chunk["chunk_id"], "source": chunk["source"]}
        for chunk, _ in retrieved
    ]
    retrieved_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "score": round(score, 4),
        }
        for chunk, score in retrieved
    ]

    return {
        "answer": answer.strip(),
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
        "latency_ms": round(elapsed_ms, 1),
        "retrieved_chunk_count": len(retrieved),
        "unique_source_count": len(set(chunk["source"] for chunk, _ in retrieved))
    }
