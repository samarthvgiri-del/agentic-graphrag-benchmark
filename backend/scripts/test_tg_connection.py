import sys
import os
import json

# Ensure we can import backend packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graph.tigergraph_client import TigerGraphClient
from dotenv import load_dotenv

def test_connection():
    # Force load .env from backend
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    client = TigerGraphClient()

    report = {
        "configured": client.configured,
        "connected": False,
        "graph_reachable": False,
        "vertices_count": None,
        "edges_count": None
    }

    if client.configured:
        status = client.get_status()
        report["connected"] = status["status"] == "connected"

        if report["connected"] and client.client:
            try:
                # ping the graph directly
                v_counts = client.client.getVertexCount('*')
                e_counts = client.client.getEdgeCount('*')
                report["graph_reachable"] = True
                report["vertices_count"] = v_counts
                report["edges_count"] = e_counts
            except Exception as e:
                print(f"Error fetching counts: {e}")

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    test_connection()
