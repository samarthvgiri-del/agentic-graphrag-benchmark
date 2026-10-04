"""
Qdrant-backed vector store for persistent local storage (Phase 3).

Replaces the Phase 2 in-memory retriever with a real vector database.
Uses Qdrant in local persistent mode — no server process required.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.embeddings.embedder import embedder

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "documents"


class QdrantVectorStore:
    """Persistent local Qdrant vector store."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name
        self.dimension = embedder.dimension

        self.client = QdrantClient(url=self.url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                self.collection_name,
                self.dimension,
            )
        else:
            logger.info("Qdrant collection '%s' already exists", self.collection_name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    def upsert(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Embed and upsert a list of chunk dicts into Qdrant in batches.

        Each chunk dict must have at least:
          - chunk_id: str
          - text: str
        All other keys are stored as payload metadata.

        Returns the number of points upserted.
        """
        if not chunks:
            return 0

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch_chunks]

            # Embed the batch
            vectors = embedder.embed(texts)

            points = []
            for idx, chunk in enumerate(batch_chunks):
                # Use a deterministic UUID derived from chunk_id for idempotent upserts
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
                payload = {k: v for k, v in chunk.items() if k != "vector"}
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vectors[idx].tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)
            logger.info("Upserted batch of %d points into '%s' (Total: %d)", len(points), self.collection_name, total_upserted)

        return total_upserted

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Semantic similarity search.

        Returns a list of (payload_dict, score) tuples.
        """
        query_vector = embedder.embed_query(query).tolist()

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            (hit.payload, hit.score) for hit in results.points
        ]

    # ------------------------------------------------------------------
    # Retriever interface (compatible with pipeline.py)
    # ------------------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Drop-in replacement for InMemoryRetriever.retrieve()."""
        return self.search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vector_dimension": self.dimension,
                "points_count": info.points_count,
                "status": str(info.status),
                "url": self.url,
            }
        except Exception as exc:
            logger.error("Failed to get Qdrant status: %s", exc)
            return {
                "collection_name": self.collection_name,
                "error": str(exc),
            }

    def count(self) -> int:
        """Return the number of stored vectors."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_collection(self) -> None:
        """Delete the collection (useful for testing)."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)


def build_vector_store(
    url: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> QdrantVectorStore:
    """Factory function for creating a QdrantVectorStore."""
    return QdrantVectorStore(
        url=url,
        collection_name=collection_name,
    )
