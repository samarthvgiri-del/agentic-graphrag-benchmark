import unittest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.graph.tigergraph_client import TigerGraphClient
from app.graph.graph_traversal import GraphTraversal

class TestGraphTraversal(unittest.TestCase):
    def setUp(self):
        self.client = TigerGraphClient()
        self.traverser = GraphTraversal(self.client)

    def test_tigergraph_connection(self):
        """Test that the TigerGraph connection is available."""
        status = self.client.get_status()
        self.assertTrue(status.get("configured"), "TigerGraph client should be configured")
        self.assertEqual(status.get("status"), "connected", "TigerGraph client should be connected")

    def test_forward_edges(self):
        """Test that forward edges return correctly for Event -> Chunk (MENTIONS_EVENT reverse check) or similar."""
        # For testing forward edges, we can test Event -> Games (PART_OF)
        # Note: Depending on TigerGraph setup, if we query Event -> PART_OF, we should get Games.
        events = self.client.client.getVertices('Event', limit=1)
        if events:
            v_id = events[0]['v_id']
            edges = self.client.get_edges("Event", v_id, "PART_OF")
            # Just ensure the method executes without error and returns a list
            self.assertIsInstance(edges, list)

    def test_reverse_edges(self):
        """Test the new interpreted GSQL reverse edge functionality (Games <- PART_OF <- Event)"""
        # We test that get_incoming_edges successfully executes the interpreted GSQL.
        # We will use games:2018_winter which is known to have reverse edges.
        edges = self.client.get_incoming_edges("Games", "games:2018_winter", "PART_OF", "Event")
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 0, "Reverse edges should have been found for games:2018_winter")

        # Verify edge structure
        first_edge = edges[0]
        self.assertEqual(first_edge.get("to_type"), "Games")
        self.assertEqual(first_edge.get("from_type"), "Event")
        self.assertEqual(first_edge.get("e_type"), "PART_OF")
        self.assertIn("attributes", first_edge)
        self.assertIn("source_chunk_id", first_edge["attributes"])

    def test_graph_traversal(self):
        """Test that the traversal logic aggregates chunks from reverse edges"""
        linked_entities = [{"linked_type": "Games", "linked_id": "games:2018_winter"}]
        results = self.traverser.traverse(linked_entities)

        self.assertIn("relationships", results)
        self.assertIn("retrieved_chunks", results)
        self.assertGreater(len(results["relationships"]), 0, "Should have retrieved relationships via reverse edge")
        self.assertGreater(len(results["retrieved_chunks"]), 0, "Should have retrieved provenance chunks")

        # Check that we haven't exceeded the hard limit
        self.assertLessEqual(len(results["retrieved_chunks"]), 100, "Retrieved chunks should be limited to 100")

if __name__ == '__main__':
    unittest.main()
