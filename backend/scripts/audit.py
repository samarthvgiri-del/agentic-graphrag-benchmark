import os
import sys
import logging
from typing import Set, Tuple
from collections import Counter

from app.rag.vector_store import build_vector_store
from app.graph.tigergraph_client import TigerGraphClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def audit():
    store = build_vector_store()
    tg_client = TigerGraphClient()

    logger.info("Connecting to Qdrant and TigerGraph...")

    # 1. Qdrant fetch
    logger.info("Fetching all chunk IDs and document sources from Qdrant...")
    qdrant_chunk_ids = set()
    qdrant_doc_ids = set()

    offset = None
    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for point in points:
            chunk_id = point.payload.get("chunk_id")
            source = point.payload.get("source")
            if chunk_id:
                qdrant_chunk_ids.add(chunk_id)
            if source:
                qdrant_doc_ids.add(source)

        if next_offset is None:
            break
        offset = next_offset

    # 2. TigerGraph fetch
    logger.info("Fetching all Chunk and Document IDs from TigerGraph...")
    tg_chunks = tg_client.client.getVertices("Chunk", limit=200000)
    tg_chunk_ids = set(c["v_id"] for c in tg_chunks)

    tg_docs = tg_client.client.getVertices("Document", limit=200000)
    tg_doc_ids = set(d["v_id"] for d in tg_docs)

    # 3. Analyze
    overlap_chunks = qdrant_chunk_ids & tg_chunk_ids
    qdrant_only_chunks = qdrant_chunk_ids - tg_chunk_ids
    tg_only_chunks = tg_chunk_ids - qdrant_chunk_ids

    overlap_docs = qdrant_doc_ids & tg_doc_ids
    qdrant_only_docs = qdrant_doc_ids - tg_doc_ids
    tg_only_docs = tg_doc_ids - qdrant_doc_ids

    # Output Report
    logger.info(f"--- AUDIT REPORT ---")
    logger.info(f"Total Qdrant chunks: {len(qdrant_chunk_ids)}")
    logger.info(f"Total TigerGraph chunks: {len(tg_chunk_ids)}")
    logger.info(f"Overlapping chunk IDs: {len(overlap_chunks)}")
    logger.info(f"Qdrant-only chunk IDs: {len(qdrant_only_chunks)}")
    logger.info(f"TigerGraph-only chunk IDs: {len(tg_only_chunks)}")

    logger.info(f"Total Qdrant documents (sources): {len(qdrant_doc_ids)}")
    logger.info(f"Total TigerGraph documents: {len(tg_doc_ids)}")
    logger.info(f"Document overlap: {len(overlap_docs)}")
    logger.info(f"Qdrant-only document IDs: {len(qdrant_only_docs)}")
    logger.info(f"TigerGraph-only document IDs: {len(tg_only_docs)}")

if __name__ == "__main__":
    audit()
