"""
API routes for document ingestion and status (Phase 3).
"""

import os
import tempfile
import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import IngestResponse, IngestStatusResponse, IngestFileResult
from app.rag.vector_store import build_vector_store
from app.rag.ingest_service import ingest_file
from app.rag.document_loader import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Shared vector store instance — same one used by RAG routes
from app.api._shared_store import get_vector_store


@router.post("", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest one or more documents into Qdrant."""
    store = get_vector_store()
    results = []

    for upload in files:
        ext = os.path.splitext(upload.filename or "")[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(IngestFileResult(
                file=upload.filename or "unknown",
                pages=0,
                chunks=0,
                status=f"error: unsupported file type '{ext}'",
            ))
            continue

        # Save to temp file for processing
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=ext, prefix="ingest_"
            ) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp_path = tmp.name

            result = ingest_file(tmp_path, store)
            # Override file name with original upload name
            result["file"] = upload.filename or result["file"]
            results.append(IngestFileResult(**result))
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", upload.filename, exc)
            results.append(IngestFileResult(
                file=upload.filename or "unknown",
                pages=0,
                chunks=0,
                status=f"error: {exc}",
            ))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    total_chunks = sum(r.chunks for r in results)
    return IngestResponse(
        files_processed=len(results),
        total_chunks=total_chunks,
        results=results,
    )


@router.get("/status", response_model=IngestStatusResponse)
async def ingest_status():
    """Return the current status of the Qdrant collection."""
    store = get_vector_store()
    status = store.get_status()
    return IngestStatusResponse(
        configured=status.get("status") is not None,
        status="online" if "error" not in status else "offline",
        collection_name=status.get("collection_name"),
        points_count=status.get("points_count"),
        url=status.get("url")
    )
