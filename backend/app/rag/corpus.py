"""
Synthetic test corpus for Phase 2 in-memory RAG.

Each chunk has a unique id, source document, and text content.
The corpus covers a few distinct topics so retrieval quality is testable.
"""

from typing import List, TypedDict


class DocumentChunk(TypedDict):
    chunk_id: str
    source: str
    text: str


SYNTHETIC_CORPUS: List[DocumentChunk] = [
    # --- Retrieval-Augmented Generation ---
    {
        "chunk_id": "rag-001",
        "source": "intro_to_rag.pdf",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances "
            "large language models by retrieving relevant documents from an "
            "external knowledge base before generating an answer. This grounds "
            "the model's output in factual evidence and reduces hallucinations."
        ),
    },
    {
        "chunk_id": "rag-002",
        "source": "intro_to_rag.pdf",
        "text": (
            "A typical RAG pipeline has three stages: (1) the user query is "
            "embedded into a vector, (2) a similarity search retrieves the "
            "top-k most relevant document chunks from a vector store, and "
            "(3) the retrieved chunks are passed as context to the LLM which "
            "generates a grounded answer."
        ),
    },
    # --- Graph Databases ---
    {
        "chunk_id": "graph-001",
        "source": "graph_databases_overview.pdf",
        "text": (
            "Graph databases store data as nodes and edges, making them ideal "
            "for modelling relationships. TigerGraph is a distributed graph "
            "database that supports real-time deep-link analytics and can "
            "traverse billions of edges in seconds."
        ),
    },
    {
        "chunk_id": "graph-002",
        "source": "graph_databases_overview.pdf",
        "text": (
            "GraphRAG extends traditional RAG by augmenting retrieval with "
            "graph traversal. Instead of relying solely on vector similarity, "
            "it walks the knowledge graph to discover related entities and "
            "relationships, producing richer context for the LLM."
        ),
    },
    # --- Vector Databases ---
    {
        "chunk_id": "vector-001",
        "source": "vector_db_comparison.pdf",
        "text": (
            "Qdrant is an open-source vector similarity search engine written "
            "in Rust. It supports filtering, payload storage, and can run "
            "locally or in the cloud. It is well-suited for RAG applications "
            "that need fast approximate nearest-neighbor search."
        ),
    },
    {
        "chunk_id": "vector-002",
        "source": "vector_db_comparison.pdf",
        "text": (
            "When choosing a vector database for RAG, important factors include "
            "query latency, indexing speed, support for metadata filtering, "
            "and the ability to scale horizontally. Qdrant, Milvus, and Weaviate "
            "are popular open-source options."
        ),
    },
    # --- LLMs ---
    {
        "chunk_id": "llm-001",
        "source": "local_llm_guide.pdf",
        "text": (
            "Ollama is a tool that allows running large language models locally. "
            "It supports models like Llama 3, Qwen, Mistral, and Phi. Running "
            "models locally eliminates API costs and keeps data private."
        ),
    },
    {
        "chunk_id": "llm-002",
        "source": "local_llm_guide.pdf",
        "text": (
            "The Qwen3 4B model is a compact yet capable language model from "
            "Alibaba Cloud. With 4 billion parameters it fits comfortably on "
            "consumer GPUs and is well-suited for local RAG applications where "
            "low latency and zero cost are priorities."
        ),
    },
    # --- Benchmarking ---
    {
        "chunk_id": "bench-001",
        "source": "benchmarking_best_practices.pdf",
        "text": (
            "Benchmarking RAG systems requires measuring retrieval quality "
            "(precision@k, recall@k) and generation quality (faithfulness, "
            "relevance, coherence). Automated metrics should be complemented "
            "with human evaluation for a complete picture."
        ),
    },
    {
        "chunk_id": "bench-002",
        "source": "benchmarking_best_practices.pdf",
        "text": (
            "Latency is a critical benchmark metric for RAG systems. The total "
            "pipeline latency includes embedding time, retrieval time, and LLM "
            "generation time. For interactive applications, end-to-end latency "
            "should ideally be under 5 seconds."
        ),
    },
]
