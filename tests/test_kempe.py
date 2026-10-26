import unittest
from unittest.mock import patch

import networkx as nx

from vera_tdma.coordinates import Node
from vera_tdma.optimization.kempe import _kempe_components, kempe_refine
from vera_tdma.verification import verify_schedule


def two_disjoint_edges():
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            ("A", {"x_m": 0, "y_m": 0}),
            ("B", {"x_m": 100, "y_m": 0}),
            ("C", {"x_m": 2000, "y_m": 0}),
            ("D", {"x_m": 2100, "y_m": 0}),
        ]
    )
    graph.add_edges_from([("A", "B"), ("C", "D")])
    return graph


class KempeRefinementTests(unittest.TestCase):
    def test_two_color_induced_components_and_deterministic_component_order(self):
        graph = two_disjoint_edges()
        components = _kempe_components(
            graph,
            {"D": 1, "C": 0, "B": 1, "A": 0},
            0,
            1,
        )
        self.assertEqual(components, [("A", "B"), ("C", "D")])

    def test_swap_changes_one_complete_component_only_and_remains_valid(self):
        graph = two_disjoint_edges()
        coloring = {"A": 0, "B": 1, "C": 0, "D": 1}
        component = _kempe_components(graph, coloring, 0, 1)[0]
        swapped = dict(coloring)
        for node in component:
            swapped[node] = 1 if coloring[node] == 0 else 0

        self.assertEqual(swapped, {"A": 1, "B": 0, "C": 0, "D": 1})
        self.assertEqual(set(swapped.values()), set(coloring.values()))
        self.assertTrue(all(swapped[u] != swapped[v] for u, v in graph.edges))
        self.assertEqual(coloring, {"A": 0, "B": 1, "C": 0, "D": 1})

    def test_multiple_disconnected_components_are_swapped_separately(self):
        graph = two_disjoint_edges()
        coloring = {"A": 0, "B": 1, "C": 0, "D": 1}
        components = _kempe_components(graph, coloring, 0, 1)
        self.assertEqual(len(components), 2)
        self.assertTrue(set(components[0]).isdisjoint(components[1]))

    def test_phase_six_receives_each_candidate_after_swap_and_equal_slot_is_rejected(self):
        graph = two_disjoint_edges()
        coloring = {"A": 0, "B": 1, "C": 0, "D": 1}
        captured = []

        class EqualSlotResult:
            def __init__(self, candidate):
                self.coloring = candidate

        def observe_phase_six(_graph, candidate):
            captured.append(dict(candidate))
            return EqualSlotResult(dict(candidate))

        with patch(
            "vera_tdma.optimization.kempe.eliminate_classes",
            side_effect=observe_phase_six,
        ):
            result = kempe_refine(graph, coloring)

        self.assertGreater(result.attempted_swaps, 0)
        self.assertGreaterEqual(len(captured), 1)
        self.assertNotEqual(captured[0], coloring)
        self.assertEqual(set(captured[0].values()), set(coloring.values()))
        self.assertEqual(result.accepted_improvements, 0)
        self.assertEqual(result.coloring, coloring)
        self.assertEqual(result.slots_after, result.slots_before)

    def test_no_improvement_keeps_original_coloring_and_verifies_it(self):
        graph = two_disjoint_edges()
        coloring = {"A": 0, "B": 1, "C": 0, "D": 1}
        original = dict(coloring)
        result = kempe_refine(graph, coloring)

        self.assertEqual(coloring, original)
        self.assertEqual(result.coloring, original)
        self.assertEqual(result.slots_after, result.slots_before)
        self.assertEqual(result.accepted_improvements, 0)
        self.assertTrue(result.final_verification.valid)
        self.assertTrue(
            verify_schedule(
                [
                    Node(node, data["x_m"], data["y_m"])
                    for node, data in graph.nodes(data=True)
                ],
                result.coloring,
            ).valid
        )

    def test_repeated_execution_and_different_insertion_orders_are_deterministic(self):
        graph = two_disjoint_edges()
        reordered = nx.Graph()
        reordered.add_nodes_from(reversed(list(graph.nodes(data=True))))
        reordered.add_edges_from(reversed(list(graph.edges)))
        coloring = {"A": 0, "B": 1, "C": 0, "D": 1}

        first = kempe_refine(graph, coloring)
        second = kempe_refine(graph, coloring)
        reordered_result = kempe_refine(reordered, coloring)

        self.assertEqual(first, second)
        self.assertEqual(first.coloring, reordered_result.coloring)
        self.assertEqual(first.attempted_swaps, reordered_result.attempted_swaps)

    def test_color_pair_order_is_ascending(self):
        graph = nx.Graph()
        graph.add_nodes_from(
            (node, {"x_m": index * 10, "y_m": 0})
            for index, node in enumerate(["A", "B", "C", "D"])
        )
        nodes = list(graph.nodes)
        graph.add_edges_from(
            (first, second)
            for index, first in enumerate(nodes)
            for second in nodes[index + 1 :]
        )
        coloring = {"A": 0, "B": 1, "C": 2, "D": 3}
        observed_pairs = []
        original_components = _kempe_components

        def observe_components(graph_arg, colors, first, second):
            observed_pairs.append((first, second))
            return original_components(graph_arg, colors, first, second)

        with patch(
            "vera_tdma.optimization.kempe._kempe_components",
            side_effect=observe_components,
        ):
            kempe_refine(graph, coloring)

        self.assertTrue(observed_pairs)
        self.assertEqual(observed_pairs[:6], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    def test_final_slot_count_never_exceeds_input(self):
        graph = two_disjoint_edges()
        result = kempe_refine(graph, {"A": 0, "B": 1, "C": 0, "D": 1})
        self.assertLessEqual(result.slots_after, result.slots_before)
        self.assertTrue(all(item.valid for item in result.accepted_verifications))


if __name__ == "__main__":
    unittest.main()
