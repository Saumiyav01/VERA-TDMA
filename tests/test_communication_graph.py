import unittest

import networkx as nx

from vera_tdma.communication_graph import (
    COMMUNICATION_RANGE_M,
    build_communication_graph,
)
from vera_tdma.coordinates import Node


class CommunicationGraphTests(unittest.TestCase):
    def test_default_range_is_500_meters(self):
        self.assertEqual(COMMUNICATION_RANGE_M, 500.0)
        graph = build_communication_graph([Node("a", 0, 0), Node("b", 500, 0)])
        self.assertTrue(graph.has_edge("a", "b"))

    def test_connects_nodes_100_meters_apart(self):
        graph = build_communication_graph([Node("a", 0, 0), Node("b", 100, 0)])
        self.assertEqual(graph["a"]["b"]["distance_m"], 100.0)

    def test_connects_nodes_499_meters_apart(self):
        graph = build_communication_graph([Node("a", 0, 0), Node("b", 499, 0)])
        self.assertEqual(graph["a"]["b"]["distance_m"], 499.0)

    def test_connects_nodes_exactly_at_inclusive_500_meter_boundary(self):
        graph = build_communication_graph([Node("a", 0, 0), Node("b", 300, 400)])
        self.assertTrue(graph.has_edge("a", "b"))
        self.assertEqual(graph["a"]["b"]["distance_m"], 500.0)

    def test_does_not_connect_nodes_501_meters_apart(self):
        graph = build_communication_graph([Node("a", 0, 0), Node("b", 501, 0)])
        self.assertFalse(graph.has_edge("a", "b"))

    def test_uses_diagonal_squared_distance(self):
        inside = build_communication_graph(
            [Node("a", 0, 0), Node("b", 300, 300)]
        )
        outside = build_communication_graph(
            [Node("a", 0, 0), Node("b", 400, 300.1)]
        )
        self.assertTrue(inside.has_edge("a", "b"))
        self.assertAlmostEqual(inside["a"]["b"]["distance_m"], (300**2 + 300**2) ** 0.5)
        self.assertFalse(outside.has_edge("a", "b"))

    def test_duplicate_coordinates_create_zero_distance_edge(self):
        graph = build_communication_graph([Node("a", 12, -8), Node("b", 12, -8)])
        self.assertTrue(graph.has_edge("a", "b"))
        self.assertEqual(graph["a"]["b"]["distance_m"], 0.0)

    def test_preserves_disconnected_nodes_and_coordinates_as_attributes(self):
        graph = build_communication_graph(
            [Node("Node10", 1000, 0), Node("Node2", 0, 0)]
        )
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(list(graph.nodes), ["Node2", "Node10"])
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(
            graph.nodes["Node2"],
            {"x_m": 0.0, "y_m": 0.0, "coordinates_m": (0, 0)},
        )

    def test_rejects_duplicate_node_identifiers(self):
        with self.assertRaisesRegex(ValueError, "duplicate node identifier"):
            build_communication_graph([Node("same", 0, 0), Node("same", 1, 1)])

    def test_rejects_invalid_communication_range(self):
        for value in (-1, float("inf"), float("nan"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    build_communication_graph([], communication_range_m=value)


if __name__ == "__main__":
    unittest.main()
