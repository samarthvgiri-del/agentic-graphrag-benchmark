"""
In-memory vector retriever for Phase 2.

Embeds the synthetic corpus on startup and retrieves the top-k
most similar chunks for a given query using cosine similarity.
This will be replaced by Qdrant in Phase 3.
"""

import logging
from typing import List, Tuple

import numpy as np

from app.embeddings.embedder import embedder
from app.rag.corpus import SYNTHETIC_CORPUS, DocumentChunk

logger = logging.getLogger(__name__)


class InMemoryRetriever:
    """Simple in-memory vector store backed by numpy."""

    def __init__(self):
        self._chunks: List[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Embed and store a list of document chunks."""
        self._chunks = chunks
        texts = [c["text"] for c in chunks]
        self._embeddings = embedder.embed(texts)
        logger.info(
            "Indexed %d chunks  (embedding dim=%d)",
            len(chunks),
            self._embeddings.shape[1],
        )

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return the top-k most similar chunks with their cosine scores."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query_vec = embedder.embed_query(query)

        # Cosine similarity: dot product of L2-normalised vectors
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normed = self._embeddings / (norms + 1e-10)
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        scores = normed @ q_norm

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self._chunks[i], float(scores[i])) for i in top_indices
        ]
        return results


def build_default_retriever() -> InMemoryRetriever:
    """Build a retriever pre-loaded with the synthetic corpus."""
    retriever = InMemoryRetriever()
    retriever.index(SYNTHETIC_CORPUS)
    return retriever
