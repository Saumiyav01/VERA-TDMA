import unittest
from itertools import combinations

import networkx as nx

from vera_tdma.conflict_graph import build_conflict_graph
from vera_tdma.scheduling.dsatur import dsatur_color, validate_coloring


def graph_with(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def chromatic_number_brute_force(graph):
    """Small test oracle; deliberately independent of DSATUR."""
    nodes = list(graph.nodes)
    if not nodes:
        return 0
    for color_count in range(1, len(nodes) + 1):
        for assignment in range(color_count ** len(nodes)):
            remainder = assignment
            colors = {}
            for node in nodes:
                colors[node] = remainder % color_count
                remainder //= color_count
            if all(colors[first] != colors[second] for first, second in graph.edges):
                return color_count
    raise AssertionError("finite graph should have a coloring")


class DsaturTests(unittest.TestCase):
    def setUp(self):
        communication_topologies = {
            "two-node": graph_with(["A", "B"], [("A", "B")]),
            "A-B-C chain": graph_with(
                ["A", "B", "C"], [("A", "B"), ("B", "C")]
            ),
            "five-node chain": graph_with(
                ["A", "B", "C", "D", "E"],
                [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
            ),
            "star": graph_with(
                ["center", "leaf1", "leaf2", "leaf3"],
                [
                    ("center", "leaf1"),
                    ("center", "leaf2"),
                    ("center", "leaf3"),
                ],
            ),
            "disconnected components": graph_with(
                ["A", "B", "C", "D", "E"],
                [("A", "B"), ("B", "C"), ("D", "E")],
            ),
            "isolated": graph_with(["A", "isolated"], []),
        }
        self.topologies = {
            name: build_conflict_graph(graph)
            for name, graph in communication_topologies.items()
        }
        self.topologies["triangle"] = graph_with(
            ["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")]
        )

    def assert_valid_and_edge_safe(self, graph, colors, number_of_colors):
        self.assertEqual(set(colors), set(graph.nodes))
        self.assertEqual(number_of_colors, len(set(colors.values())))
        validate_coloring(graph, colors)
        for first, second in graph.edges:
            self.assertNotEqual(colors[first], colors[second])

    def test_expected_color_counts_for_required_topologies(self):
        expected_counts = {
            "two-node": 2,
            "triangle": 3,
            "A-B-C chain": 3,
            "five-node chain": 3,
            "star": 4,
            "disconnected components": 3,
            "isolated": 1,
        }
        for name, expected in expected_counts.items():
            with self.subTest(topology=name):
                graph = self.topologies[name]
                colors, count = dsatur_color(graph)
                self.assertEqual(count, expected)
                self.assert_valid_and_edge_safe(graph, colors, count)

    def test_isolated_only_graph_uses_color_zero(self):
        colors, count = dsatur_color(graph_with(["isolated"], []))
        self.assertEqual(colors, {"isolated": 0})
        self.assertEqual(count, 1)

    def test_deterministic_repeated_execution(self):
        graph = self.topologies["five-node chain"]
        expected = dsatur_color(graph)
        for _ in range(5):
            self.assertEqual(dsatur_color(graph), expected)

    def test_different_node_insertion_orders_have_same_logical_coloring(self):
        nodes = ["Node10", "Node2", "Node1", "Node3", "isolated"]
        edges = [("Node10", "Node3"), ("Node2", "Node3"), ("Node1", "Node2")]
        first_graph = graph_with(nodes, edges)
        second_graph = graph_with(reversed(nodes), reversed(edges))

        first = dsatur_color(first_graph)
        second = dsatur_color(second_graph)

        self.assertEqual(first, second)

    def test_compares_with_brute_force_chromatic_number_on_tiny_graphs(self):
        tiny_graphs = [
            graph_with([], []),
            graph_with(["A"], []),
            graph_with(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")]),
            graph_with(["A", "B", "C", "D"], list(combinations(["A", "B", "C", "D"], 2))),
            graph_with(["A", "B", "C", "D", "E"], [
                ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")
            ]),
            graph_with(["A", "B", "C", "D", "E"], [
                ("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")
            ]),
        ]
        for graph in tiny_graphs:
            with self.subTest(nodes=list(graph.nodes), edges=list(graph.edges)):
                colors, count = dsatur_color(graph)
                self.assert_valid_and_edge_safe(graph, colors, count)
                self.assertGreaterEqual(count, chromatic_number_brute_force(graph))

    def test_validator_rejects_incomplete_invalid_and_noncontiguous_colorings(self):
        graph = graph_with(["A", "B"], [("A", "B")])
        invalid_colorings = [
            ({"A": 0}, "every graph node"),
            ({"A": 0, "B": 0}, "share a color"),
            ({"A": -1, "B": 0}, "non-negative integers"),
            ({"A": True, "B": 0}, "non-negative integers"),
            ({"A": 0, "B": 2}, "contiguous"),
        ]
        for colors, error in invalid_colorings:
            with self.subTest(colors=colors):
                with self.assertRaisesRegex(ValueError, error):
                    validate_coloring(graph, colors)

    def test_does_not_modify_conflict_graph(self):
        graph = self.topologies["star"]
        before_nodes = list(graph.nodes(data=True))
        before_edges = list(graph.edges(data=True))
        before_graph_attributes = dict(graph.graph)

        dsatur_color(graph)

        self.assertEqual(list(graph.nodes(data=True)), before_nodes)
        self.assertEqual(list(graph.edges(data=True)), before_edges)
        self.assertEqual(dict(graph.graph), before_graph_attributes)


if __name__ == "__main__":
    unittest.main()
