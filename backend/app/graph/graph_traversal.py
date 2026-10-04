import logging
from typing import Dict, Any, List
from app.graph.tigergraph_client import TigerGraphClient

logger = logging.getLogger(__name__)

class GraphTraversal:
    def __init__(self, client: TigerGraphClient):
        self.client = client
        self.edge_types = ["REPRESENTS", "WON_MEDAL", "COMPETED_IN", "PART_OF", "RELATES_TO"]

    def traverse(self, linked_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform a 1-hop traversal from the linked entities.
        Returns relationships and fetches the supporting chunks.
        """
        results = {
            "relationships": [],
            "retrieved_chunks": []
        }

        if not self.client.client:
            return results

        chunk_ids_to_fetch = set()

        # Define incoming edge requirements: { TargetVertexType: [(IncomingEdgeType, SourceVertexType)] }
        incoming_edge_map = {
            "Games": [("PART_OF", "Event"), ("MENTIONS_GAMES", "Chunk")],
            "Event": [("MENTIONS_EVENT", "Chunk")],
            "Country": [("REPRESENTS", "Athlete"), ("MENTIONS_COUNTRY", "Chunk")],
            "Athlete": [("MENTIONS_ATHLETE", "Chunk")],
            "GenericEntity": [("MENTIONS_GENERIC", "Chunk"), ("RELATES_TO", "GenericEntity")]
        }

        for ent in linked_entities:
            v_type = ent["linked_type"]
            v_id = ent["linked_id"]

            # 1-hop outgoing edges
            for e_type in self.edge_types:
                try:
                    edges = self.client.get_edges(v_type, v_id, e_type)
                    for edge in edges:
                        attrs = edge.get("attributes", {})
                        rel = {
                            "source": v_id,
                            "target": edge.get("target_id", edge.get("to_id")),
                            "rel_type": e_type,
                            "attributes": attrs
                        }
                        results["relationships"].append(rel)

                        # Provenance
                        chunk_id = attrs.get("source_chunk_id")
                        if chunk_id:
                            chunk_ids_to_fetch.add(chunk_id)

                except Exception as e:
                    logger.warning(f"Error traversing {e_type} for {v_id}: {str(e)}")

            # 1-hop incoming edges (using interpreted GSQL)
            if v_type in incoming_edge_map:
                for inc_edge_type, source_v_type in incoming_edge_map[v_type]:
                    try:
                        inc_edges = self.client.get_incoming_edges(v_type, v_id, inc_edge_type, source_v_type)
                        for edge in inc_edges:
                            attrs = edge.get("attributes", {})
                            rel = {
                                "source": edge.get("from_id"),
                                "target": v_id,
                                "rel_type": inc_edge_type,
                                "attributes": attrs
                            }
                            results["relationships"].append(rel)

                            # For Chunk sources, the chunk is the source!
                            if source_v_type == "Chunk":
                                chunk_ids_to_fetch.add(edge.get("from_id"))

                            # Provenance on the edge itself
                            chunk_id = attrs.get("source_chunk_id")
                            if chunk_id:
                                chunk_ids_to_fetch.add(chunk_id)

                    except Exception as e:
                        logger.warning(f"Error traversing incoming {inc_edge_type} for {v_id}: {str(e)}")

        # Fetch actual chunks for provenance
        chunk_ids_list = list(chunk_ids_to_fetch)[:100]  # Limit to prevent hanging on massive hubs
        for chunk_id in chunk_ids_list:
            try:
                # Strip prefix if present
                if chunk_id.startswith("chunk:"):
                    chunk_id = chunk_id.replace("chunk:", "")

                chunk_vertex = self.client.client.getVerticesById("Chunk", chunk_id)
                if chunk_vertex:
                    chunk_text = chunk_vertex[0].get("attributes", {}).get("text", "")
                    results["retrieved_chunks"].append({
                        "chunk_id": chunk_id,
                        "text": chunk_text
                    })
            except Exception as e:
                logger.warning(f"Error fetching chunk {chunk_id}: {str(e)}")

        return results
