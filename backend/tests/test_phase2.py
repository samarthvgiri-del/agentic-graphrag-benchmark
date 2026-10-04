"""
Phase 2 tests for: Ollama client, embeddings, retriever, and RAG endpoint.

Tests that require a live Ollama server are marked with @pytest.mark.ollama
so they can be skipped in CI with: pytest -m "not ollama"
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.llm.ollama_client import OllamaClient
from app.embeddings.embedder import embedder
from app.rag.corpus import SYNTHETIC_CORPUS
from app.rag.retriever import InMemoryRetriever, build_default_retriever


client = TestClient(app)


# ------------------------------------------------------------------ #
#  Embedding tests
# ------------------------------------------------------------------ #

class TestEmbedder:
    def test_embed_returns_numpy(self):
        vecs = embedder.embed(["hello world"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape[0] == 1

    def test_embed_multiple(self):
        texts = ["first", "second", "third"]
        vecs = embedder.embed(texts)
        assert vecs.shape[0] == 3

    def test_embed_query_returns_1d(self):
        vec = embedder.embed_query("test query")
        assert vec.ndim == 1
        assert vec.shape[0] == embedder.dimension

    def test_similar_texts_have_high_score(self):
        v1 = embedder.embed_query("retrieval augmented generation")
        v2 = embedder.embed_query("RAG combines retrieval with generation")
        score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert score > 0.5  # Semantically similar


# ------------------------------------------------------------------ #
#  Retriever tests
# ------------------------------------------------------------------ #

class TestRetriever:
    def setup_method(self):
        self.retriever = build_default_retriever()

    def test_retriever_returns_results(self):
        results = self.retriever.retrieve("What is RAG?", top_k=3)
        assert len(results) == 3

    def test_retriever_top_result_is_relevant(self):
        results = self.retriever.retrieve("What is RAG?", top_k=1)
        chunk, score = results[0]
        # The top result should be from the RAG-related chunks
        assert "rag" in chunk["chunk_id"].lower() or "rag" in chunk["text"].lower()
        assert score > 0.3

    def test_retriever_respects_top_k(self):
        for k in [1, 2, 5]:
            results = self.retriever.retrieve("vector database", top_k=k)
            assert len(results) == min(k, len(SYNTHETIC_CORPUS))

    def test_empty_retriever(self):
        empty = InMemoryRetriever()
        results = empty.retrieve("anything", top_k=3)
        assert results == []


# ------------------------------------------------------------------ #
#  Ollama client tests (mocked — no live server required)
# ------------------------------------------------------------------ #

class TestOllamaClientMocked:
    def test_default_config(self):
        oc = OllamaClient()
        assert "localhost" in oc.base_url
        assert oc.model == "qwen3:4b"

    @pytest.mark.asyncio
    async def test_generate_success(self):
        oc = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "Hello from mock"}

        with patch("app.llm.ollama_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await oc.generate("test prompt")
            assert result == "Hello from mock"

    @pytest.mark.asyncio
    async def test_unavailable_ollama_raises(self):
        import httpx
        oc = OllamaClient(base_url="http://localhost:99999")

        with patch("app.llm.ollama_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                await oc.generate("test")

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        import httpx
        oc = OllamaClient(timeout=1)

        with patch("app.llm.ollama_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="timed out"):
                await oc.generate("test")


# ------------------------------------------------------------------ #
#  RAG endpoint tests (mocked Ollama)
# ------------------------------------------------------------------ #

class TestRAGEndpoint:
    def test_rag_endpoint_returns_structure(self):
        """POST /api/query/rag should return the correct shape even with mocked LLM."""
        with patch(
            "app.rag.pipeline.ollama_client.generate",
            new_callable=AsyncMock,
            return_value="This is a mocked answer based on the evidence.",
        ):
            resp = client.post(
                "/api/query/rag",
                json={"question": "What is RAG?"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "retrieved_chunks" in data
        assert "latency_ms" in data
        assert isinstance(data["citations"], list)
        assert len(data["citations"]) > 0
        assert all("chunk_id" in c and "source" in c for c in data["citations"])

    def test_rag_endpoint_custom_top_k(self):
        with patch(
            "app.rag.pipeline.ollama_client.generate",
            new_callable=AsyncMock,
            return_value="Mocked answer.",
        ):
            resp = client.post(
                "/api/query/rag",
                json={"question": "vector database", "top_k": 2},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["retrieved_chunks"]) == 2
