import itertools
import unittest

import networkx as nx

from vera_tdma.communication_graph import build_communication_graph
from vera_tdma.conflict_graph import build_conflict_graph
from vera_tdma.coordinates import Node
from vera_tdma.optimization.class_elimination import eliminate_classes
from vera_tdma.optimization.exact_solver import solve_exact
from vera_tdma.optimization.kempe import kempe_refine
from vera_tdma.scheduling.dsatur import dsatur_color


def graph_with(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


class ExactSolverTests(unittest.TestCase):
    def setUp(self):
        self.known_graphs = {
            "empty": (graph_with([], []), 0),
            "single node": (graph_with(["A"], []), 1),
            "two-node edge": (graph_with(["A", "B"], [("A", "B")]), 2),
            "triangle": (
                graph_with(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")]),
                3,
            ),
            "path": (
                graph_with(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")]),
                2,
            ),
            "four-cycle": (
                graph_with(["A", "B", "C", "D"], [
                    ("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")
                ]),
                2,
            ),
            "five-cycle": (
                graph_with(["A", "B", "C", "D", "E"], [
                    ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")
                ]),
                3,
            ),
            "K4": (
                graph_with(
                    ["A", "B", "C", "D"],
                    list(itertools.combinations(["A", "B", "C", "D"], 2)),
                ),
                4,
            ),
            "bipartite": (
                graph_with(
                    ["A", "B", "C", "D", "E", "F"],
                    [
                        ("A", "D"), ("A", "E"), ("B", "D"), ("B", "F"),
                        ("C", "E"), ("C", "F"),
                    ],
                ),
                2,
            ),
            "disconnected": (
                graph_with(
                    ["A", "B", "C", "D", "E"],
                    [("A", "B"), ("B", "C"), ("D", "E")],
                ),
                2,
            ),
        }

    def test_known_chromatic_numbers_and_coloring_validity(self):
        for name, (graph, expected) in self.known_graphs.items():
            with self.subTest(graph=name):
                result = solve_exact(graph)
                self.assertTrue(result.is_optimal)
                self.assertEqual(result.chromatic_number, expected)
                self.assertEqual(set(result.optimal_coloring), set(graph.nodes))
                self.assertEqual(
                    set(result.optimal_coloring.values()),
                    set(range(expected)),
                )
                for first, second in graph.edges:
                    self.assertNotEqual(
                        result.optimal_coloring[first],
                        result.optimal_coloring[second],
                    )

    def test_search_proves_every_smaller_color_count_infeasible(self):
        result = solve_exact(self.known_graphs["five-cycle"][0])
        self.assertEqual(result.infeasible_color_counts, (1, 2))
        self.assertTrue(result.is_optimal)

    def test_deterministic_repeated_execution(self):
        graph = self.known_graphs["five-cycle"][0]
        first = solve_exact(graph)
        second = solve_exact(graph)
        self.assertEqual(first.optimal_coloring, second.optimal_coloring)
        self.assertEqual(first.chromatic_number, second.chromatic_number)
        self.assertEqual(first.nodes_explored, second.nodes_explored)

    def test_input_graph_is_not_modified(self):
        graph = self.known_graphs["disconnected"][0]
        before_nodes = list(graph.nodes(data=True))
        before_edges = list(graph.edges(data=True))
        before_attributes = dict(graph.graph)

        solve_exact(graph)

        self.assertEqual(list(graph.nodes(data=True)), before_nodes)
        self.assertEqual(list(graph.edges(data=True)), before_edges)
        self.assertEqual(dict(graph.graph), before_attributes)

    def test_greedy_dsatur_can_be_suboptimal(self):
        graph = graph_with(
            [f"N{index}" for index in range(7)],
            [
                ("N0", "N2"), ("N0", "N3"), ("N0", "N5"),
                ("N1", "N4"), ("N1", "N5"), ("N1", "N6"),
                ("N2", "N5"), ("N2", "N6"), ("N3", "N4"),
                ("N3", "N5"), ("N4", "N6"),
            ],
        )
        _, heuristic_count = dsatur_color(graph)
        exact = solve_exact(graph)

        self.assertEqual(exact.chromatic_number, 3)
        self.assertEqual(heuristic_count, 4)
        self.assertGreater(heuristic_count, exact.chromatic_number)

    def test_heuristic_pipeline_comparison_on_spatial_fixtures(self):
        spatial_cases = {
            "triangle": [
                Node("A", 0, 0),
                Node("B", 100, 0),
                Node("C", 50, 80),
            ],
            "three-node distance-two chain": [
                Node("A", 0, 0),
                Node("B", 400, 0),
                Node("C", 800, 0),
            ],
            "disjoint direct pairs": [
                Node("A", 0, 0),
                Node("B", 100, 0),
                Node("C", 2000, 0),
                Node("D", 2100, 0),
            ],
        }
        comparisons = {}
        for name, nodes in spatial_cases.items():
            with self.subTest(case=name):
                communication = build_communication_graph(nodes)
                conflict = build_conflict_graph(communication)
                dsatur, dsatur_slots = dsatur_color(conflict)
                class_result = eliminate_classes(conflict, dsatur)
                kempe_result = kempe_refine(conflict, class_result.coloring)
                exact = solve_exact(conflict)
                heuristic_slots = kempe_result.slots_after
                absolute_gap = heuristic_slots - exact.chromatic_number
                relative_gap = (
                    absolute_gap / exact.chromatic_number
                    if exact.chromatic_number
                    else 0.0
                )
                comparisons[name] = (
                    dsatur_slots,
                    heuristic_slots,
                    exact.chromatic_number,
                    absolute_gap,
                    relative_gap,
                )
                self.assertGreaterEqual(absolute_gap, 0)
                self.assertTrue(kempe_result.final_verification.valid)
                self.assertTrue(exact.is_optimal)
        self.assertEqual(comparisons["triangle"][2], 3)
        self.assertEqual(comparisons["three-node distance-two chain"][2], 3)
        self.assertEqual(comparisons["disjoint direct pairs"][2], 2)

    def test_rejects_invalid_graph_shapes(self):
        with self.assertRaisesRegex(ValueError, "undirected simple graph"):
            solve_exact(nx.DiGraph())
        graph = nx.Graph()
        graph.add_edge("A", "A")
        with self.assertRaisesRegex(ValueError, "self-loops"):
            solve_exact(graph)

    def test_reports_search_size_and_runtime(self):
        result = solve_exact(self.known_graphs["K4"][0])
        self.assertGreater(result.nodes_explored, 0)
        self.assertGreaterEqual(result.runtime_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
