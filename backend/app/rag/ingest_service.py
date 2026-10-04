"""
Ingestion service (Phase 3).

Orchestrates: load document → chunk → embed → store in Qdrant.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.rag.document_loader import load_document
from app.rag.chunker import chunk_document
from app.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def ingest_file(
    file_path: str,
    vector_store: QdrantVectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Dict[str, Any]:
    """Ingest a single file into the vector store.

    Returns a summary dict with ingestion results.
    """
    path = Path(file_path)
    logger.info("Ingesting: %s", path.name)

    # 1. Load
    pages = load_document(str(path))

    # 2. Chunk
    chunks = chunk_document(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        return {
            "file": path.name,
            "pages": len(pages),
            "chunks": 0,
            "status": "empty",
        }

    # 3. Upsert into Qdrant
    count = vector_store.upsert(chunks)

    return {
        "file": path.name,
        "pages": len(pages),
        "chunks": count,
        "status": "success",
    }


def ingest_directory(
    dir_path: str,
    vector_store: QdrantVectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Ingest all supported files from a directory."""
    from app.rag.document_loader import SUPPORTED_EXTENSIONS

    results = []
    directory = Path(dir_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                result = ingest_file(str(f), vector_store, chunk_size, chunk_overlap)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", f.name, exc)
                results.append({
                    "file": f.name,
                    "pages": 0,
                    "chunks": 0,
                    "status": f"error: {exc}",
                })

    return results
