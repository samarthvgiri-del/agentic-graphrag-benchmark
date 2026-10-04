import logging
from typing import Dict, Any, List, Optional
from app.graph.tigergraph_client import TigerGraphClient
from app.graph.entity_search import EntityCandidateSearch, normalize_entity_name

logger = logging.getLogger(__name__)

class EntityLinker:
    def __init__(self, client: TigerGraphClient, vector_store=None):
        self.client = client
        self.searcher = EntityCandidateSearch(client, vector_store=vector_store)
        self.supported_types = ["Athlete", "Country", "Event", "Games", "GenericEntity"]

    def link_entities(self, extracted_entities: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Takes a list of extracted entities (with 'name' and 'type') from the user query
        and attempts to match them to actual vertices in TigerGraph.
        """
        linked = []
        for ent in extracted_entities:
            name = ent.get("name", "")
            ent_type = ent.get("type", "GenericEntity")

            if not name:
                continue

            candidates = self.searcher.search(name, ent_type)
            if candidates:
                logger.debug(f"Candidates for {name}: {candidates}")
                # Take the highest confidence candidate (already ordered exact -> semantic)
                best = candidates[0]
                linked.append({
                    "query_entity": name,
                    "query_type": ent_type,
                    "linked_id": best["id"],
                    "linked_type": best["type"],
                    "confidence": best["confidence"],
                    "match_type": best["match_type"]
                })
            else:
                logger.warning(f"Entity not found: {name}")

        return linked
