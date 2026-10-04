import logging
from typing import Dict, Any, List
from app.graph.tigergraph_client import TigerGraphClient
from app.rag.vector_store import build_vector_store

logger = logging.getLogger(__name__)

def normalize_entity_name(name: str) -> str:
    """Normalize a name to match the graph representation."""
    return name.strip().lower().replace(" ", "_").replace("'", "").replace('"', '').replace(",", "")

class EntityCandidateSearch:
    def __init__(self, tg_client: TigerGraphClient, vector_store=None):
        self.tg_client = tg_client
        self.vector_store = vector_store or build_vector_store()
        self.supported_types = ["Athlete", "Country", "Event", "Games", "GenericEntity"]

    def search(self, name: str, entity_type: str = "GenericEntity") -> List[Dict[str, Any]]:
        """
        Search for an entity candidate.
        Priority:
        1. Exact/normalized match via TigerGraph
        2. Qdrant semantic search (simulated via vector store search if needed, but GraphRAG
           typically relies on exact match unless an entity index is built).
        """
        if not name:
            return []

        if entity_type not in self.supported_types:
            entity_type = "GenericEntity"

        norm_name = normalize_entity_name(name)
        expected_id = f"{entity_type.lower()}:{norm_name}"

        candidates = []

        # 1. Exact match via TigerGraph
        try:
            # If type is GenericEntity, we don't know the true type. Try all types!
            types_to_try = [entity_type] if entity_type != "GenericEntity" else self.supported_types

            for t in types_to_try:
                expected_id = f"{t.lower()}:{norm_name}"
                try:
                    res = self.tg_client.client.getVerticesById(t, expected_id)
                    if res:
                        candidates.append({
                            "id": expected_id,
                            "name": res[0].get("attributes", {}).get("name", name),
                            "type": t,
                            "confidence": 1.0 if t == entity_type else 0.9,
                            "match_type": "exact"
                        })
                        return candidates
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Exact match failed for {norm_name}: {str(e)}")

        # 2. Qdrant Semantic Candidate Search
        # We can search the vector store for chunks mentioning the entity, and extract existing entities
        # from those chunks if they were linked. But since we need real TG IDs, and Qdrant stores text chunks,
        # we can't easily get *entity IDs* from Qdrant directly unless they were stored in Qdrant metadata.
        # However, the requirement is "Qdrant semantic candidate search -> return real TigerGraph IDs only"
        # If Qdrant returns chunks, we could query TigerGraph for entities attached to those chunks.
        try:
            results = self.vector_store.search(name, top_k=3)
            chunk_ids = [res[0]["chunk_id"] for res in results]

            # Fetch entities connected to these chunks
            # A chunk has edge MENTIONS_ATHLETE, MENTIONS_COUNTRY, etc.
            # But the edge is FROM Chunk TO Entity
            for chunk_id in chunk_ids:
                for target_type in self.supported_types:
                    edge_type = f"MENTIONS_{target_type.upper().replace('GENERICENTITY', 'GENERIC')}"
                    try:
                        edges = self.tg_client.client.getEdges("Chunk", chunk_id, edge_type)
                        for edge in edges:
                            tgt_id = edge.get("target_id", edge.get("to_id"))
                            if tgt_id:
                                # Ensure we don't add duplicates
                                if not any(c["id"] == tgt_id for c in candidates):
                                    logger.debug(f"[GraphRAG] Qdrant semantic search found chunk {chunk_id} with edge to {tgt_id}")
                                    candidates.append({
                                        "id": tgt_id,
                                        "name": tgt_id.split(":", 1)[-1].replace("_", " "),  # crude fallback for name
                                        "type": target_type,
                                        "confidence": 0.7,  # Semantic fallback confidence
                                        "match_type": "semantic"
                                    })
                    except Exception as e:
                        # Edge type might not exist or other error
                        pass

        except Exception as e:
            logger.error(f"Semantic fallback failed for {name}: {str(e)}")

        return candidates
