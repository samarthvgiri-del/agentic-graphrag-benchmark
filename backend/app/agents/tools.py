import time
from typing import Dict, Any, List
import logging
from app.models.schemas import NormalizedToolResult
from app.graph.entity_linker import EntityLinker
from app.graph.graph_traversal import GraphTraversal
from app.rag.vector_store import QdrantVectorStore
from app.embeddings.embedder import LocalEmbedder

logger = logging.getLogger(__name__)

class AgenticTools:
    def __init__(self, tigergraph_client, qdrant_client: QdrantVectorStore, embedding_model: LocalEmbedder):
        self.tg_client = tigergraph_client
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

        self.entity_linker = EntityLinker(self.tg_client, vector_store=self.qdrant_client)
        self.graph_traverser = GraphTraversal(self.tg_client)

    def entity_linking(self, text: str) -> NormalizedToolResult:
        start_t = time.time()
        try:
            # We assume text is just a comma-separated list or phrase of entities
            # We'll just link the whole text as one query for simplicity,
            # or split by comma.
            entities_to_link = [e.strip() for e in text.split(",") if e.strip()]
            if not entities_to_link:
                entities_to_link = [text]

            linked = self.entity_linker.link_entities([{"name": e, "type": "GenericEntity"} for e in entities_to_link])

            latency = int((time.time() - start_t) * 1000)

            # Formulate evidence
            evidence = []
            for e in linked:
                evidence.append(f"Entity found in graph: '{e['query_entity']}' -> {e['linked_id']} ({e['linked_type']}) with score {e.get('confidence', 0.0)}")

            return NormalizedToolResult(
                tool="entity_linking",
                success=True,
                results=linked,
                evidence=evidence,
                source_ids=[e["linked_id"] for e in linked],
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"entity_linking failed: {e}")
            return NormalizedToolResult(
                tool="entity_linking",
                success=False,
                results=[],
                evidence=[f"Failed to link entities: {str(e)}"],
                source_ids=[],
                latency_ms=int((time.time() - start_t) * 1000)
            )

    def graph_traversal(self, linked_id: str, linked_type: str = "GenericEntity") -> NormalizedToolResult:
        start_t = time.time()
        try:
            # Fake a linked entity list
            linked_entities = [{"linked_id": linked_id, "linked_type": linked_type}]
            traversal_results = self.graph_traverser.traverse(linked_entities)

            latency = int((time.time() - start_t) * 1000)

            chunks = traversal_results.get("retrieved_chunks", [])
            evidence = [c["text"] for c in chunks]
            source_ids = [c["chunk_id"] for c in chunks]

            # also add relationships as evidence
            rels = traversal_results.get("relationships", [])
            for r in rels:
                evidence.append(f"Relationship: {r['source']} {r['rel_type']} {r['target']}")
                source_ids.append(f"rel_{r['source']}_{r['target']}")

            return NormalizedToolResult(
                tool="graph_traversal",
                success=True,
                results=chunks,
                evidence=evidence,
                source_ids=source_ids,
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"graph_traversal failed: {e}")
            return NormalizedToolResult(
                tool="graph_traversal",
                success=False,
                results=[],
                evidence=[f"Failed to traverse graph: {str(e)}"],
                source_ids=[],
                latency_ms=int((time.time() - start_t) * 1000)
            )

    def vector_search(self, query: str, top_k: int = 3) -> NormalizedToolResult:
        start_t = time.time()
        try:
            results = self.qdrant_client.search(query, top_k=top_k)

            latency = int((time.time() - start_t) * 1000)

            evidence = []
            source_ids = []
            res_dicts = []

            for payload, score in results:
                text = payload.get("text", "")
                chunk_id = payload.get("chunk_id", "unknown")
                evidence.append(text)
                source_ids.append(chunk_id)
                res_dicts.append({"chunk_id": chunk_id, "text": text, "score": score})

            return NormalizedToolResult(
                tool="vector_search",
                success=True,
                results=res_dicts,
                evidence=evidence,
                source_ids=source_ids,
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"vector_search failed: {e}")
            return NormalizedToolResult(
                tool="vector_search",
                success=False,
                results=[],
                evidence=[f"Failed vector search: {str(e)}"],
                source_ids=[],
                latency_ms=int((time.time() - start_t) * 1000)
            )

    def document_retrieval(self, document_id: str) -> NormalizedToolResult:
        start_t = time.time()
        try:
            # We can use vector_search with a source filter, or just do a generic search with the doc ID as query
            results = self.qdrant_client.search(document_id, top_k=3, source_filter=document_id)

            latency = int((time.time() - start_t) * 1000)

            evidence = []
            source_ids = []
            res_dicts = []

            for payload, score in results:
                text = payload.get("text", "")
                chunk_id = payload.get("chunk_id", "unknown")
                evidence.append(text)
                source_ids.append(chunk_id)
                res_dicts.append({"chunk_id": chunk_id, "text": text, "score": score})

            return NormalizedToolResult(
                tool="document_retrieval",
                success=True,
                results=res_dicts,
                evidence=evidence,
                source_ids=source_ids,
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"document_retrieval failed: {e}")
            return NormalizedToolResult(
                tool="document_retrieval",
                success=False,
                results=[],
                evidence=[f"Failed document retrieval: {str(e)}"],
                source_ids=[],
                latency_ms=int((time.time() - start_t) * 1000)
            )
