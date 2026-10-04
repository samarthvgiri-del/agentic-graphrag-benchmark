"""
Shared singleton for the Qdrant vector store.

Used by both routes_rag.py and routes_ingest.py so they
operate on the same collection.
"""

from app.rag.vector_store import QdrantVectorStore, build_vector_store

_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = build_vector_store()
    return _store
