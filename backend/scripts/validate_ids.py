import asyncio
import logging
from app.rag.vector_store import build_vector_store
from app.graph.tigergraph_client import TigerGraphClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Connecting to Qdrant...")
    store = build_vector_store()

    logger.info("Connecting to TigerGraph...")
    tg_client = TigerGraphClient()

    # Scroll all points in Qdrant
    logger.info("Fetching all chunk IDs from Qdrant...")
    qdrant_ids = set()
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for point in points:
            qdrant_ids.add(point.payload.get("chunk_id"))

        if offset is None:
            break

    logger.info(f"Total Qdrant Chunk IDs: {len(qdrant_ids)}")

    # Get all chunk IDs from TigerGraph
    logger.info("Fetching all chunk IDs from TigerGraph...")
    tg_chunks = tg_client.client.getVertices("Chunk", limit=100000)
    tg_ids = set(c["v_id"] for c in tg_chunks)

    logger.info(f"Total TigerGraph Chunk IDs: {len(tg_ids)}")

    missing_in_tg = qdrant_ids - tg_ids
    missing_in_qdrant = tg_ids - qdrant_ids

    if missing_in_tg:
        logger.error(f"FAIL: {len(missing_in_tg)} chunk IDs found in Qdrant but missing in TigerGraph.")
        logger.error(f"Sample: {list(missing_in_tg)[:5]}")

    if missing_in_qdrant:
        logger.error(f"FAIL: {len(missing_in_qdrant)} chunk IDs found in TigerGraph but missing in Qdrant.")
        logger.error(f"Sample: {list(missing_in_qdrant)[:5]}")

    if not missing_in_tg and not missing_in_qdrant:
        logger.info("SUCCESS: Qdrant chunk_ids and TigerGraph Chunk vertex IDs match perfectly!")

if __name__ == "__main__":
    main()
