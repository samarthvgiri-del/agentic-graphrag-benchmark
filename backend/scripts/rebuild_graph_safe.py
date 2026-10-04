import sys
import os
import time
import logging
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.graph.tigergraph_client import TigerGraphClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def safe_clear():
    client = TigerGraphClient()

    logger.info("--- INSPECTING CURRENT GRAPH ---")
    v_counts = client.client.getVertexCount('*')
    e_counts = client.client.getEdgeCount('*')

    logger.info("Current Vertex Counts:")
    for vt, count in v_counts.items():
        logger.info(f"  {vt}: {count}")

    logger.info("Current Edge Counts:")
    for et, count in e_counts.items():
        logger.info(f"  {et}: {count}")

    logger.info("--- THE FOLLOWING WILL BE DELETED ---")
    logger.info("Vertices of types: Chunk, Document, Athlete, Country, Event, Games, GenericEntity")
    logger.info("(Edges connected to these vertices will be automatically cascaded/deleted by TigerGraph)")

    # Actually delete them
    types_to_delete = ["Chunk", "Document", "Athlete", "Country", "Event", "Games", "GenericEntity"]
    for vt in types_to_delete:
        logger.info(f"Deleting all {vt} vertices...")
        try:
            res = client.client.delVertices(vt)
            logger.info(f"Deleted {res} vertices of type {vt}")
        except Exception as e:
            logger.error(f"Failed to delete {vt}: {e}")

    logger.info("--- VERIFYING DELETION ---")
    time.sleep(2)
    v_counts = client.client.getVertexCount('*')
    logger.info("Post-deletion Vertex Counts:")
    for vt, count in v_counts.items():
        logger.info(f"  {vt}: {count}")

if __name__ == "__main__":
    asyncio.run(safe_clear())
