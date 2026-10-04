"""
Local embedding generation using Sentence Transformers.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# A small, fast, high-quality model that runs locally with zero cost.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """Wraps a SentenceTransformer model for local embedding generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) numpy array of embeddings for the given texts."""
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string and return a 1-D vector."""
        return self.model.encode(query, convert_to_numpy=True)


# Module-level singleton (loaded once, reused across requests).
embedder = LocalEmbedder()
