import logging
from collections import Counter
from app.rag.vector_store import build_vector_store
from app.rag.canonical_chunker import get_canonical_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def test_canonical_reproduction():
    logger.info("Fetching existing Qdrant chunk IDs...")
    store = build_vector_store()
    qdrant_chunk_ids = set()
    offset = None

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for p in points:
            cid = p.payload.get("chunk_id")
            if cid:
                qdrant_chunk_ids.add(cid)
        if next_offset is None:
            break
        offset = next_offset

    logger.info("Generating canonical chunks...")
    # we know corpus.jsonl is at data/corpus.jsonl
    # from scripts/ we use ../data/corpus.jsonl
    canonical_chunks = get_canonical_chunks("data/corpus/corpus.jsonl", chunk_size=500, chunk_overlap=100)

    canonical_ids = [c.chunk_id for c in canonical_chunks]
    canonical_id_set = set(canonical_ids)

    id_counts = Counter(canonical_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    overlap = qdrant_chunk_ids & canonical_id_set
    qdrant_only = qdrant_chunk_ids - canonical_id_set
    canonical_only = canonical_id_set - qdrant_chunk_ids

    logger.info("--- CANONICAL REPRODUCTION REPORT ---")
    logger.info(f"Qdrant IDs: {len(qdrant_chunk_ids)}")
    logger.info(f"Canonical IDs: {len(canonical_ids)}")
    logger.info(f"Exact overlap count: {len(overlap)}")
    logger.info(f"Qdrant-only: {len(qdrant_only)}")
    logger.info(f"Canonical-only: {len(canonical_only)}")
    logger.info(f"Duplicate canonical IDs: {len(duplicates)}")

    if len(overlap) != len(qdrant_chunk_ids) or len(overlap) != len(canonical_id_set):
        logger.error("OVERLAP IS NOT 100%. Mismatch detected.")
        if qdrant_only:
            logger.error(f"Sample Qdrant-only: {list(qdrant_only)[:5]}")
        if canonical_only:
            logger.error(f"Sample Canonical-only: {list(canonical_only)[:5]}")
    else:
        logger.info("SUCCESS: 100% Overlap achieved!")

if __name__ == "__main__":
    test_canonical_reproduction()
