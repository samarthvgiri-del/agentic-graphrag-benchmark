"""
Phase 3 tests for: Qdrant vector store, document loading, chunking,
ingestion, persistence, and Qdrant-backed RAG retrieval.

All Qdrant tests use an isolated temp directory so they don't
interfere with the production data.
"""

import os
import shutil
import tempfile

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.rag.vector_store import QdrantVectorStore
from app.rag.document_loader import load_document, SUPPORTED_EXTENSIONS
from app.rag.chunker import chunk_document
from app.rag.ingest_service import ingest_file, ingest_directory
from app.embeddings.embedder import embedder

client = TestClient(app)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


import uuid

# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def store():
    """Provide a clean QdrantVectorStore in Qdrant server."""
    collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name=collection,
    )
    yield store
    store.delete_collection()


# ------------------------------------------------------------------ #
#  Document loader tests
# ------------------------------------------------------------------ #

class TestDocumentLoader:
    def test_load_txt(self):
        pages = load_document(os.path.join(TEST_DATA_DIR, "rag_overview.txt"))
        assert len(pages) >= 1
        assert "RAG" in pages[0]["text"]
        assert pages[0]["source"] == "rag_overview.txt"

    def test_load_md(self):
        pages = load_document(os.path.join(TEST_DATA_DIR, "graph_databases.md"))
        assert len(pages) >= 1
        assert "graph" in pages[0]["text"].lower()

    def test_load_json(self):
        pages = load_document(os.path.join(TEST_DATA_DIR, "tools.json"))
        assert len(pages) == 3  # 3 items in the JSON array

    def test_load_csv(self):
        pages = load_document(os.path.join(TEST_DATA_DIR, "tools_comparison.csv"))
        assert len(pages) == 4  # 4 data rows
        assert "Qdrant" in pages[0]["text"]

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/file.txt")

    def test_load_unsupported_type(self):
        # Create a temp file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_document(path)
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Chunker tests
# ------------------------------------------------------------------ #

class TestChunker:
    def test_basic_chunking(self):
        pages = [{"text": "A" * 1200, "source": "test.txt", "page": None}]
        chunks = chunk_document(pages, chunk_size=500, chunk_overlap=100)
        assert len(chunks) >= 2
        # Each chunk should have required metadata
        for c in chunks:
            assert "chunk_id" in c
            assert "source" in c
            assert "text" in c
            assert "chunk_index" in c

    def test_chunk_metadata_preserved(self):
        pages = [{"text": "Hello world " * 100, "source": "doc.pdf", "page": 3}]
        chunks = chunk_document(pages, chunk_size=200, chunk_overlap=50)
        for c in chunks:
            assert c["source"] == "doc.pdf"
            assert c["page"] == 3

    def test_small_text_single_chunk(self):
        pages = [{"text": "Short text.", "source": "s.txt", "page": None}]
        chunks = chunk_document(pages, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 1

    def test_empty_text_skipped(self):
        pages = [{"text": "   ", "source": "empty.txt", "page": None}]
        chunks = chunk_document(pages, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 0

    def test_deterministic_chunk_ids(self):
        pages = [{"text": "Deterministic test " * 50, "source": "det.txt", "page": 1}]
        c1 = chunk_document(pages, chunk_size=200, chunk_overlap=50)
        c2 = chunk_document(pages, chunk_size=200, chunk_overlap=50)
        assert [c["chunk_id"] for c in c1] == [c["chunk_id"] for c in c2]


# ------------------------------------------------------------------ #
#  Qdrant vector store tests
# ------------------------------------------------------------------ #

class TestQdrantVectorStore:
    def test_collection_created(self, store):
        status = store.get_status()
        assert status["collection_name"].startswith("test_collection_")
        assert status["vector_dimension"] == embedder.dimension

    def test_upsert_and_count(self, store):
        chunks = [
            {"chunk_id": "t1", "source": "test.txt", "text": "Hello world"},
            {"chunk_id": "t2", "source": "test.txt", "text": "Goodbye world"},
        ]
        n = store.upsert(chunks)
        assert n == 2
        assert store.count() == 2

    def test_search(self, store):
        chunks = [
            {"chunk_id": "rag1", "source": "rag.txt", "text": "Retrieval augmented generation improves LLM accuracy"},
            {"chunk_id": "cook1", "source": "cook.txt", "text": "How to bake a chocolate cake recipe"},
            {"chunk_id": "rag2", "source": "rag.txt", "text": "RAG reduces hallucinations in language models"},
        ]
        store.upsert(chunks)
        results = store.search("What is RAG?", top_k=2)
        assert len(results) == 2
        # Top result should be RAG-related
        top_payload, top_score = results[0]
        assert "rag" in top_payload["chunk_id"].lower()
        assert top_score > 0

    def test_metadata_preserved(self, store):
        chunks = [
            {"chunk_id": "m1", "source": "meta.pdf", "page": 5, "chunk_index": 0,
             "text": "Test metadata preservation"},
        ]
        store.upsert(chunks)
        results = store.search("metadata", top_k=1)
        payload, _ = results[0]
        assert payload["source"] == "meta.pdf"
        assert payload["page"] == 5
        assert payload["chunk_id"] == "m1"

    def test_upsert_idempotent(self, store):
        chunk = [{"chunk_id": "idem1", "source": "test.txt", "text": "Idempotent upsert"}]
        store.upsert(chunk)
        store.upsert(chunk)  # Same chunk_id → should not duplicate
        assert store.count() == 1

    def test_empty_upsert(self, store):
        n = store.upsert([])
        assert n == 0
        assert store.count() == 0

    def test_empty_collection_search(self, store):
        results = store.search("anything", top_k=3)
        assert results == []

    def test_source_filter(self, store):
        chunks = [
            {"chunk_id": "f1", "source": "alpha.txt", "text": "Alpha document about AI"},
            {"chunk_id": "f2", "source": "beta.txt", "text": "Beta document about AI"},
        ]
        store.upsert(chunks)
        results = store.search("AI", top_k=2, source_filter="alpha.txt")
        assert len(results) == 1
        assert results[0][0]["source"] == "alpha.txt"

    def test_retrieve_interface(self, store):
        """Verify that .retrieve() works as a drop-in for the pipeline."""
        chunks = [
            {"chunk_id": "r1", "source": "r.txt", "text": "Retrieve interface test"},
        ]
        store.upsert(chunks)
        results = store.retrieve("interface", top_k=1)
        assert len(results) == 1
        payload, score = results[0]
        assert payload["chunk_id"] == "r1"

    def test_delete_collection(self, store):
        chunks = [{"chunk_id": "d1", "source": "d.txt", "text": "Delete me"}]
        store.upsert(chunks)
        assert store.count() == 1
        store.delete_collection()
        # Collection should no longer exist in the active client
        collections = [c.name for c in store.client.get_collections().collections]
        assert store.collection_name not in collections


# ------------------------------------------------------------------ #
#  Ingestion service tests
# ------------------------------------------------------------------ #

class TestIngestion:
    def test_ingest_single_file(self):
        col = f"ingest_test_{uuid.uuid4().hex[:8]}"
        store = QdrantVectorStore(url="http://localhost:6333", collection_name=col)
        try:
            result = ingest_file(
                os.path.join(TEST_DATA_DIR, "rag_overview.txt"),
                store,
            )
            assert result["status"] == "success"
            assert result["chunks"] > 0
            assert store.count() > 0
        finally:
            store.delete_collection()

    def test_ingest_directory(self):
        col = f"dir_test_{uuid.uuid4().hex[:8]}"
        store = QdrantVectorStore(url="http://localhost:6333", collection_name=col)
        try:
            results = ingest_directory(TEST_DATA_DIR, store)
            assert len(results) >= 3  # txt, md, json, csv
            total = sum(r["chunks"] for r in results)
            assert total > 0
            assert store.count() == total
        finally:
            store.delete_collection()

    def test_ingest_invalid_file(self):
        col = f"inv_test_{uuid.uuid4().hex[:8]}"
        store = QdrantVectorStore(url="http://localhost:6333", collection_name=col)
        try:
            with pytest.raises(FileNotFoundError):
                ingest_file("/does/not/exist.txt", store)
        finally:
            store.delete_collection()


# ------------------------------------------------------------------ #
#  Persistence test
# ------------------------------------------------------------------ #

class TestPersistence:
    def test_vectors_persist_across_instances(self):
        """Ingest data, close the store, reopen, and verify vectors remain."""
        col = f"persist_test_{uuid.uuid4().hex[:8]}"
        # 1. Ingest
        store1 = QdrantVectorStore(
            url="http://localhost:6333", collection_name=col
        )
        chunks = [
            {"chunk_id": "p1", "source": "p.txt", "text": "Persistence test chunk one"},
            {"chunk_id": "p2", "source": "p.txt", "text": "Persistence test chunk two"},
        ]
        store1.upsert(chunks)
        assert store1.count() == 2

        # 2. Close client
        del store1

        # 3. Reopen from the same path
        store2 = QdrantVectorStore(
            url="http://localhost:6333", collection_name=col
        )
        try:
            assert store2.count() == 2

            results = store2.search("persistence", top_k=1)
            assert len(results) == 1
            assert results[0][0]["chunk_id"] in ("p1", "p2")
        finally:
            store2.delete_collection()


# ------------------------------------------------------------------ #
#  RAG via Qdrant endpoint tests (mocked Ollama)
# ------------------------------------------------------------------ #

class TestRAGViaQdrant:
    def test_rag_with_qdrant_fallback(self):
        """When Qdrant is empty, /api/query/rag should still work via fallback."""
        with patch(
            "app.rag.pipeline.ollama_client.generate",
            new_callable=AsyncMock,
            return_value="Fallback answer.",
        ):
            resp = client.post(
                "/api/query/rag",
                json={"question": "What is RAG?"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["retrieved_chunks"]) > 0

    def test_ingest_status_endpoint(self):
        """GET /api/ingest/status should return collection info."""
        resp = client.get("/api/ingest/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "collection_name" in data
        assert "points_count" in data
