import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graph.tigergraph_client import TigerGraphClient
from app.graph.entity_linker import EntityLinker
from app.graph.graph_traversal import GraphTraversal

class MockTigerGraphClient:
    def __init__(self):
        self.configured = True
        self.client = self

    def getVerticesById(self, v_type, v_id):
        if v_id == "athlete:test":
            return [{"attributes": {"name": "Test Athlete"}}]
        if v_id == "chunk:1":
            return [{"attributes": {"text": "Test Chunk Text"}}]
        return []

    def get_edges(self, v_type, v_id, e_type):
        if v_id == "athlete:test" and e_type == "WON_MEDAL":
            return [{"target_id": "event:test", "attributes": {"medal_type": "Gold", "source_chunk_id": "chunk:1"}}]
        return []

def test_entity_linker():
    mock_client = TigerGraphClient()
    mock_client.client = MockTigerGraphClient()
    linker = EntityLinker(mock_client)

    extracted = [{"name": "Test", "type": "Athlete"}]
    linked = linker.link_entities(extracted)
    assert len(linked) == 1
    assert linked[0]["linked_id"] == "athlete:test"

def test_graph_traversal():
    mock_client = TigerGraphClient()
    mock_client.client = MockTigerGraphClient()

    # Overwrite the get_edges method to use the mock
    mock_client.get_edges = mock_client.client.get_edges

    traverser = GraphTraversal(mock_client)
    linked = [{"linked_id": "athlete:test", "linked_type": "Athlete"}]

    results = traverser.traverse(linked)
    assert len(results["relationships"]) == 1
    assert results["relationships"][0]["target"] == "event:test"
    assert len(results["retrieved_chunks"]) == 1
    assert results["retrieved_chunks"][0]["text"] == "Test Chunk Text"
