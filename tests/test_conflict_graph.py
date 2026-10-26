import unittest
from itertools import combinations

import networkx as nx

from vera_tdma.conflict_graph import build_conflict_graph
from vera_tdma.ordering import natural_sort_key


def topology_graph(nodes, edges):
    graph = nx.Graph(network="test topology")
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


class ConflictGraphTests(unittest.TestCase):
    def setUp(self):
        self.topologies = {
            "direct pair": topology_graph(["A", "B"], [("A", "B")]),
            "A-B-C chain": topology_graph(
                ["A", "B", "C"], [("A", "B"), ("B", "C")]
            ),
            "five-node chain": topology_graph(
                ["A", "B", "C", "D", "E"],
                [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
            ),
            "triangle": topology_graph(
                ["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")]
            ),
            "direct and two-hop": topology_graph(
                ["A", "B", "C", "D"],
                [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")],
            ),
            "star": topology_graph(
                ["center", "leaf1", "leaf2", "leaf3"],
                [
                    ("center", "leaf1"),
                    ("center", "leaf2"),
                    ("center", "leaf3"),
                ],
            ),
            "disconnected components": topology_graph(
                ["A", "B", "C", "D", "E"],
                [("A", "B"), ("B", "C"), ("D", "E")],
            ),
            "isolated node": topology_graph(
                ["A", "B", "isolated"], [("A", "B")]
            ),
            "duplicate coordinates": topology_graph(
                [
                    ("A", {"x_m": 4.0, "y_m": -3.0}),
                    ("B", {"x_m": 4.0, "y_m": -3.0}),
                    ("C", {"x_m": 20.0, "y_m": 20.0}),
                ],
                [("A", "B"), ("B", "C")],
            ),
            "4x4 grid": topology_graph(
                [f"n{row}{column}" for row in range(4) for column in range(4)],
                [
                    (f"n{row}{column}", f"n{row + 1}{column}")
                    for row in range(3)
                    for column in range(4)
                ]
                + [
                    (f"n{row}{column}", f"n{row}{column + 1}")
                    for row in range(4)
                    for column in range(3)
                ],
            ),
        }

    def assert_matches_shortest_path_oracle(self, graph):
        conflict = build_conflict_graph(graph)
        self.assertEqual(set(conflict.nodes), set(graph.nodes))
        self.assertFalse(nx.number_of_selfloops(conflict))
        self.assertFalse(conflict.is_directed())

        for first, second in combinations(graph.nodes, 2):
            try:
                distance = nx.shortest_path_length(graph, first, second)
            except nx.NetworkXNoPath:
                distance = float("inf")
            expected = distance <= 2
            self.assertEqual(
                conflict.has_edge(first, second),
                expected,
                f"conflict mismatch for {first!r}, {second!r} at distance {distance}",
            )

            if not expected:
                continue
            direct = graph.has_edge(first, second)
            common_neighbors = set(graph.neighbors(first)) & set(graph.neighbors(second))
            edge_data = conflict[first][second]
            expected_type = (
                "direct_and_two_hop"
                if direct and common_neighbors
                else "direct"
                if direct
                else "two_hop"
            )
            self.assertEqual(edge_data["conflict_type"], expected_type)
            if common_neighbors:
                self.assertEqual(
                    edge_data["witnesses"],
                    tuple(sorted(common_neighbors, key=natural_sort_key)),
                )
            else:
                self.assertNotIn("witnesses", edge_data)
        return conflict

    def test_all_requested_topologies_against_shortest_path_oracle(self):
        for name, graph in self.topologies.items():
            with self.subTest(topology=name):
                self.assert_matches_shortest_path_oracle(graph)

    def test_required_specific_edge_metadata(self):
        chain = self.assert_matches_shortest_path_oracle(self.topologies["A-B-C chain"])
        self.assertEqual(chain["A"]["C"]["conflict_type"], "two_hop")
        self.assertEqual(chain["A"]["C"]["witnesses"], ("B",))

        triangle = self.assert_matches_shortest_path_oracle(self.topologies["triangle"])
        self.assertEqual(triangle["A"]["B"]["conflict_type"], "direct_and_two_hop")
        self.assertEqual(triangle["A"]["B"]["witnesses"], ("C",))

    def test_preserves_node_and_graph_attributes_without_modifying_input(self):
        graph = topology_graph(
            [("Node10", {"x_m": 10, "y_m": 20, "extra": ["value"]}),
             ("Node2", {"x_m": 2, "y_m": 3, "label": "kept"}),
             ("isolated", {"x_m": 1000, "y_m": 1000})],
            [("Node10", "Node2", {"weight": 7})],
        )
        original_nodes = [
            (node, dict(attributes)) for node, attributes in graph.nodes(data=True)
        ]
        original_edges = [
            (first, second, dict(attributes))
            for first, second, attributes in graph.edges(data=True)
        ]
        original_graph_attributes = dict(graph.graph)

        conflict = build_conflict_graph(graph)

        self.assertEqual(list(conflict.nodes), ["Node2", "Node10", "isolated"])
        self.assertEqual(
            [(node, dict(attributes)) for node, attributes in graph.nodes(data=True)],
            original_nodes,
        )
        self.assertEqual(
            [
                (first, second, dict(attributes))
                for first, second, attributes in graph.edges(data=True)
            ],
            original_edges,
        )
        self.assertEqual(dict(graph.graph), original_graph_attributes)
        self.assertEqual(
            conflict.nodes["Node2"],
            {"x_m": 2, "y_m": 3, "label": "kept"},
        )
        self.assertEqual(conflict.graph["network"], "test topology")
        self.assertEqual(conflict["Node2"]["Node10"]["conflict_type"], "direct")

    def test_output_is_deterministic_across_input_insertion_order(self):
        nodes = ["Node10", "Node2", "Node1", "Node3"]
        edges = [("Node10", "Node3"), ("Node2", "Node3"), ("Node1", "Node2")]
        first = topology_graph(nodes, edges)
        second = topology_graph(reversed(nodes), reversed(edges))

        first_conflict = build_conflict_graph(first)
        second_conflict = build_conflict_graph(second)

        self.assertEqual(list(first_conflict.nodes), list(second_conflict.nodes))
        self.assertEqual(
            [
                (u, v, dict(data))
                for u, v, data in first_conflict.edges(data=True)
            ],
            [
                (u, v, dict(data))
                for u, v, data in second_conflict.edges(data=True)
            ],
        )
if __name__ == "__main__":
    unittest.main()
