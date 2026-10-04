"""
Text chunker for the ingestion pipeline (Phase 3).

Splits documents into overlapping fixed-size chunks while
preserving source metadata (document name, page, chunk index).
"""

import hashlib
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 100  # characters


def chunk_document(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Split loaded document pages into chunks with metadata.

    Each output chunk dict has:
      - chunk_id: str  (deterministic hash)
      - source: str
      - page: int | None
      - chunk_index: int
      - text: str
    """
    chunks: List[Dict[str, Any]] = []
    global_idx = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page.get("page")

        # Normalise whitespace
        text = " ".join(text.split())
        if not text:
            continue

        start = 0
        local_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Deterministic chunk_id based on source + content position
            raw_id = f"{source}::{page_num}::{start}::{chunk_text[:50]}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:12]

            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "source_path": source,
                "page": page_num,
                "chunk_index": global_idx,
                "text": chunk_text,
            })

            global_idx += 1
            local_idx += 1
            start += chunk_size - chunk_overlap

    logger.info(
        "Chunked %d page(s) → %d chunk(s)  (size=%d, overlap=%d)",
        len(pages),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
