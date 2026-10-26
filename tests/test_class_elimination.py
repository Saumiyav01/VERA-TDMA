import unittest

import networkx as nx

from vera_tdma.coordinates import Node
from vera_tdma.optimization.class_elimination import eliminate_classes
from vera_tdma.verification import verify_schedule


def graph_with_coordinates(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node_id, {"x_m": coordinates[0], "y_m": coordinates[1]})
        for node_id, coordinates in nodes.items()
    )
    graph.add_edges_from(edges)
    return graph


class ClassEliminationTests(unittest.TestCase):
    def setUp(self):
        far_coordinates = {
            "A": (0, 0),
            "B": (1000, 0),
            "C": (2000, 0),
            "D": (3000, 0),
            "E": (4000, 0),
            "F": (5000, 0),
        }
        triangle_nodes = {
            "A": (0, 0),
            "B": (100, 0),
            "C": (50, 80),
            "I1": (5000, 0),
        }
        self.eliminable = graph_with_coordinates(
            triangle_nodes,
            [("A", "B"), ("B", "C"), ("C", "A")],
        )
        self.impossible = graph_with_coordinates(
            {"A": (0, 0), "B": (100, 0), "C": (50, 80)},
            [("A", "B"), ("B", "C"), ("C", "A")],
        )
        self.chain_distance_two = graph_with_coordinates(
            {"A": (0, 0), "B": (400, 0), "C": (800, 0)},
            [("A", "B"), ("B", "C"), ("A", "C")],
        )
        self.disconnected = graph_with_coordinates(
            {"A": (0, 0), "B": (100, 0), "C": (1000, 0), "D": (1100, 0)},
            [("A", "B"), ("C", "D")],
        )
        self.isolated = graph_with_coordinates(
            {"A": (0, 0), "isolated": (2000, 0)},
            [],
        )
        self.multiple_eliminable = graph_with_coordinates(
            {
                "A": (0, 0),
                "B": (100, 0),
                "C": (50, 80),
                "I1": (5000, 0),
                "I2": (6000, 0),
            },
            [("A", "B"), ("B", "C"), ("C", "A")],
        )

    def assert_safe_result(self, graph, initial, result):
        self.assertLessEqual(result.slots_after, result.slots_before)
        self.assertLessEqual(result.slots_after, len(set(initial.values())))
        self.assertTrue(result.final_verification.valid)
        self.assertTrue(verify_schedule(
            [
                Node(node, attributes["x_m"], attributes["y_m"])
                for node, attributes in graph.nodes(data=True)
            ],
            result.coloring,
        ).valid)
        for first, second in graph.edges:
            self.assertNotEqual(result.coloring[first], result.coloring[second])
        self.assertEqual(
            result.successful_eliminations,
            result.slots_before - result.slots_after,
        )
        prior_count = result.slots_before
        for after in result.slot_counts_after_reductions:
            self.assertLess(after, prior_count)
            prior_count = after
        self.assertEqual(prior_count, result.slots_after)

    def test_eliminates_a_clearable_class(self):
        initial = {"A": 0, "B": 1, "C": 2, "I1": 3}
        result = eliminate_classes(self.eliminable, initial)
        self.assertEqual(result.slots_before, 4)
        self.assertEqual(result.slots_after, 3)
        self.assertEqual(result.successful_eliminations, 1)
        self.assert_safe_result(self.eliminable, initial, result)

    def test_does_not_reduce_triangle(self):
        initial = {"A": 0, "B": 1, "C": 2}
        result = eliminate_classes(self.impossible, initial)
        self.assertEqual(result.slots_before, 3)
        self.assertEqual(result.slots_after, 3)
        self.assertEqual(result.successful_eliminations, 0)
        self.assert_safe_result(self.impossible, initial, result)

    def test_a_b_c_distance_two_topology_cannot_be_reduced(self):
        initial = {"A": 0, "B": 1, "C": 2}
        result = eliminate_classes(self.chain_distance_two, initial)
        self.assertEqual(result.slots_after, 3)
        self.assert_safe_result(self.chain_distance_two, initial, result)

    def test_disconnected_components_with_isolates_allow_reduction(self):
        initial = {"A": 0, "B": 1, "C": 2, "D": 3}
        result = eliminate_classes(self.disconnected, initial)
        self.assertEqual(result.slots_after, 2)
        self.assertEqual(result.successful_eliminations, 2)
        self.assert_safe_result(self.disconnected, initial, result)

    def test_isolated_node_class_can_be_eliminated(self):
        initial = {"A": 0, "isolated": 1}
        result = eliminate_classes(self.isolated, initial)
        self.assertEqual(result.slots_after, 1)
        self.assertEqual(result.coloring["isolated"], 0)
        self.assert_safe_result(self.isolated, initial, result)

    def test_multiple_classes_can_be_eliminated(self):
        initial = {"A": 0, "B": 1, "C": 2, "I1": 3, "I2": 4}
        result = eliminate_classes(self.multiple_eliminable, initial)
        self.assertEqual(result.slots_before, 5)
        self.assertEqual(result.slots_after, 3)
        self.assertEqual(result.successful_eliminations, 2)
        self.assert_safe_result(self.multiple_eliminable, initial, result)

    def test_repeated_execution_is_deterministic(self):
        initial = {"A": 0, "B": 1, "C": 2, "I1": 3, "I2": 4}
        first = eliminate_classes(self.multiple_eliminable, initial)
        second = eliminate_classes(self.multiple_eliminable, initial)
        self.assertEqual(first, second)

    def test_original_coloring_remains_unchanged(self):
        initial = {"A": 0, "B": 1, "C": 2, "I1": 3}
        original = initial.copy()
        eliminate_classes(self.eliminable, initial)
        self.assertEqual(initial, original)

    def test_rejects_invalid_initial_coloring(self):
        with self.assertRaisesRegex(ValueError, "independent schedule verification"):
            eliminate_classes(self.impossible, {"A": 0, "B": 0, "C": 2})

    def test_each_accepted_iteration_has_fewer_slots_and_passes_verifier(self):
        initial = {"A": 0, "B": 1, "C": 2, "I1": 3, "I2": 4}
        result = eliminate_classes(self.multiple_eliminable, initial)
        self.assertEqual(len(result.accepted_verifications), 2)
        self.assertTrue(all(item.valid for item in result.accepted_verifications))
        self.assertEqual(
            [len(set(initial.values())) - index for index in range(1, 3)],
            [4, 3],
        )


if __name__ == "__main__":
    unittest.main()
