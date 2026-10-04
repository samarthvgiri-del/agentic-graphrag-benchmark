import logging
import pyTigerGraph as tg
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class TigerGraphClient:
    def __init__(self):
        self.host = settings.TIGERGRAPH_HOST
        self.secret = settings.TIGERGRAPH_SECRET
        self.graphname = settings.TIGERGRAPH_GRAPHNAME
        self.client = None

        self.configured = bool(self.host and self.secret and self.graphname)
        self.connected = False

        if self.configured:
            try:
                # Try to connect using the secret directly
                self.client = tg.TigerGraphConnection(
                    host=self.host,
                    graphname=self.graphname,
                    gsqlSecret=self.secret
                )
                try:
                    self.client.getToken(self.secret)
                except Exception:
                    # If getting token with secret fails, maybe the secret IS the token
                    self.client = tg.TigerGraphConnection(
                        host=self.host,
                        graphname=self.graphname,
                        apiToken=self.secret
                    )
            except Exception as e:
                logger.error(f"Failed to initialize TigerGraph client: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        if not self.configured:
            return {"configured": False, "status": "unavailable/not configured"}

        if not self.client:
            return {"configured": True, "status": "connection_failed"}

        try:
            # Make a lightweight request to verify connection
            self.client.getEndpoints()
            self.connected = True
            return {"configured": True, "status": "connected"}
        except Exception as e:
            logger.error(f"TigerGraph connection test failed: {str(e)}")
            self.connected = False
            return {"configured": True, "status": "connection_failed"}

    def upsert_vertex(self, vertex_type: str, vertex_id: str, attributes: Dict[str, Any]) -> bool:
        if not self.client:
            return False
        try:
            self.client.upsertVertex(vertexType=vertex_type, vertexId=vertex_id, attributes=attributes)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vertex {vertex_type} {vertex_id}: {str(e)}")
            return False

    def upsert_edge(self, edge_type: str, source_type: str, source_id: str, target_type: str, target_id: str, attributes: Dict[str, Any] = None) -> bool:
        if not self.client:
            return False
        try:
            self.client.upsertEdge(
                sourceVertexType=source_type,
                sourceVertexId=source_id,
                edgeType=edge_type,
                targetVertexType=target_type,
                targetVertexId=target_id,
                attributes=attributes or {}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert edge {edge_type}: {str(e)}")
            return False

    def get_vertices(self, vertex_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        try:
            return self.client.getVertices(vertex_type, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get vertices for {vertex_type}: {str(e)}")
            return []

    def get_edges(self, vertex_type: str, vertex_id: str, edge_type: str = "") -> List[Dict[str, Any]]:
        if not self.client:
            return []
        try:
            # Retrieve edges for a specific vertex
            res = self.client.getEdges(sourceVertexType=vertex_type, sourceVertexId=vertex_id, edgeType=edge_type)
            return res
        except Exception as e:
            logger.error(f"Failed to get edges for {vertex_id}: {str(e)}")
            return []

    def get_incoming_edges(self, vertex_type: str, vertex_id: str, edge_type: str, source_type: str) -> List[Dict[str, Any]]:
        """Get incoming edges using an interpreted query."""
        if not self.client:
            return []
        try:
            query = f'''
            INTERPRET QUERY (VERTEX<{vertex_type}> v_id) FOR GRAPH {self.graphname} {{
                SetAccum<EDGE> @@edges;
                start = {{v_id}};
                res = SELECT t FROM start:s -(<{edge_type}:e)- {source_type}:t ACCUM @@edges += e;
                PRINT @@edges;
            }}
            '''
            result = self.client.runInterpretedQuery(query, params={"v_id": vertex_id})

            # Extract edges from the result
            if result and isinstance(result, list) and len(result) > 0:
                edges_list = result[0].get("@@edges", [])

                # Format to match get_edges style for drop-in replacement
                formatted_edges = []
                for edge in edges_list:
                    formatted_edges.append({
                        "from_type": edge.get("from_type"),
                        "from_id": edge.get("from_id"),
                        "to_type": edge.get("to_type"),
                        "to_id": edge.get("to_id"),
                        "e_type": edge.get("e_type"),
                        "attributes": edge.get("attributes", {})
                    })
                return formatted_edges
            return []
        except Exception as e:
            logger.error(f"Failed to get incoming edges for {vertex_id} ({edge_type}): {str(e)}")
            return []

    def query(self, query_name: str, params: Dict[str, Any]) -> List[Any]:
        if not self.client:
            return []
        try:
            return self.client.runInstalledQuery(query_name, params=params)
        except Exception as e:
            logger.error(f"Failed to run query {query_name}: {str(e)}")
            return []

    def upsert_vertices_batch(self, vertex_type: str, vertices: List[tuple]) -> int:
        """Batch upsert vertices. Each tuple is (vertex_id, attributes_dict).
        Returns count of upserted vertices."""
        if not self.client or not vertices:
            return 0
        try:
            return self.client.upsertVertices(
                vertexType=vertex_type,
                vertices=vertices,
            )
        except Exception as e:
            logger.error(f"Batch upsert vertices {vertex_type} failed: {str(e)}")
            return 0

    def upsert_edges_batch(self, source_type: str, edge_type: str, target_type: str, edges: List[tuple]) -> int:
        """Batch upsert edges. Each tuple is (source_id, target_id, attributes_dict).
        Returns count of upserted edges."""
        if not self.client or not edges:
            return 0
        try:
            return self.client.upsertEdges(
                sourceVertexType=source_type,
                edgeType=edge_type,
                targetVertexType=target_type,
                edges=edges,
            )
        except Exception as e:
            logger.error(f"Batch upsert edges {edge_type} failed: {str(e)}")
            return 0

    def init_schema(self):
        """Creates the graph schema via GSQL."""
        if not self.client:
            return False

        schema_gsql = f"""
        USE GRAPH {self.graphname}
        CREATE SCHEMA_CHANGE JOB add_phase4_schema FOR GRAPH {self.graphname} {{
            ADD VERTEX Document (PRIMARY_ID id STRING, title STRING, url STRING);
            ADD VERTEX Chunk (PRIMARY_ID id STRING, text STRING);
            ADD VERTEX Athlete (PRIMARY_ID id STRING, name STRING);
            ADD VERTEX Country (PRIMARY_ID id STRING, name STRING);
            ADD VERTEX Event (PRIMARY_ID id STRING, name STRING);
            ADD VERTEX Games (PRIMARY_ID id STRING, name STRING);
            ADD VERTEX GenericEntity (PRIMARY_ID id STRING, name STRING, entity_type STRING);

            ADD DIRECTED EDGE HAS_CHUNK (FROM Document, TO Chunk);
            ADD DIRECTED EDGE MENTIONS_ATHLETE (FROM Chunk, TO Athlete);
            ADD DIRECTED EDGE MENTIONS_COUNTRY (FROM Chunk, TO Country);
            ADD DIRECTED EDGE MENTIONS_EVENT (FROM Chunk, TO Event);
            ADD DIRECTED EDGE MENTIONS_GAMES (FROM Chunk, TO Games);
            ADD DIRECTED EDGE MENTIONS_GENERIC (FROM Chunk, TO GenericEntity);

            ADD DIRECTED EDGE REPRESENTS (FROM Athlete, TO Country, source_chunk_id STRING);
            ADD DIRECTED EDGE WON_MEDAL (FROM Athlete, TO Event, medal_type STRING, source_chunk_id STRING);
            ADD DIRECTED EDGE COMPETED_IN (FROM Athlete, TO Event, source_chunk_id STRING);
            ADD DIRECTED EDGE PART_OF (FROM Event, TO Games, source_chunk_id STRING);
            ADD DIRECTED EDGE RELATES_TO (FROM GenericEntity, TO GenericEntity, description STRING, source_chunk_id STRING);
        }}
        RUN SCHEMA_CHANGE JOB add_phase4_schema
        DROP JOB add_phase4_schema
        """
        try:
            logger.info("Initializing schema...")
            self.client.gsql(schema_gsql)
            return True
        except Exception as e:
            logger.error(f"Schema initialization failed: {str(e)}")
            return False
