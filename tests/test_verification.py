import unittest

from vera_tdma.coordinates import Node
from vera_tdma.verification import verify_schedule


class IndependentScheduleVerificationTests(unittest.TestCase):
    def test_valid_two_node_schedule(self):
        result = verify_schedule([Node("A", 0, 0), Node("B", 100, 0)], {"A": 0, "B": 1})
        self.assertTrue(result.valid)
        self.assertEqual(result.number_of_nodes, 2)
        self.assertEqual(result.direct_conflicts_checked, 1)
        self.assertEqual(result.two_hop_conflicts_checked, 0)
        self.assertEqual(result.number_of_violations, 0)

    def test_invalid_two_node_schedule_reports_direct_violation(self):
        result = verify_schedule([Node("A", 0, 0), Node("B", 100, 0)], {"A": 2, "B": 2})
        self.assertFalse(result.valid)
        self.assertEqual(result.number_of_violations, 1)
        violation = result.violations[0]
        self.assertEqual((violation.node_1, violation.node_2), ("A", "B"))
        self.assertEqual((violation.color_1, violation.color_2), (2, 2))
        self.assertEqual(violation.conflict_type, "direct")
        self.assertEqual(violation.distance_m, 100.0)
        self.assertEqual(violation.path, ("A", "B"))

    def test_valid_a_b_c_distance_two_schedule(self):
        nodes = [Node("A", 0, 0), Node("B", 400, 0), Node("C", 800, 0)]
        result = verify_schedule(nodes, {"A": 0, "B": 1, "C": 2})
        self.assertTrue(result.valid)
        self.assertEqual(result.direct_conflicts_checked, 2)
        self.assertEqual(result.two_hop_conflicts_checked, 1)

    def test_invalid_a_b_c_when_endpoints_share_color(self):
        nodes = [Node("A", 0, 0), Node("B", 400, 0), Node("C", 800, 0)]
        result = verify_schedule(nodes, {"A": 0, "B": 1, "C": 0})
        self.assertFalse(result.valid)
        self.assertEqual(result.number_of_violations, 1)
        violation = result.violations[0]
        self.assertEqual((violation.node_1, violation.node_2), ("A", "C"))
        self.assertEqual(violation.conflict_type, "two_hop")
        self.assertEqual(violation.path, ("A", "B", "C"))

    def test_triangle_valid_three_color_schedule(self):
        nodes = [Node("A", 0, 0), Node("B", 100, 0), Node("C", 50, 80)]
        result = verify_schedule(nodes, {"A": 0, "B": 1, "C": 2})
        self.assertTrue(result.valid)
        self.assertEqual(result.direct_conflicts_checked, 3)
        self.assertEqual(result.two_hop_conflicts_checked, 3)

    def test_triangle_invalid_coloring(self):
        nodes = [Node("A", 0, 0), Node("B", 100, 0), Node("C", 50, 80)]
        result = verify_schedule(nodes, {"A": 0, "B": 1, "C": 0})
        self.assertFalse(result.valid)
        self.assertEqual(result.number_of_violations, 1)
        self.assertEqual(result.violations[0].conflict_type, "direct_and_two_hop")

    def test_isolated_nodes_allow_same_color_and_have_no_self_conflicts(self):
        result = verify_schedule(
            [Node("A", 0, 0), Node("isolated", 1000, 1000)],
            {"A": 0, "isolated": 0},
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.direct_conflicts_checked, 0)
        self.assertEqual(result.two_hop_conflicts_checked, 0)
        self.assertFalse(
            any(v.node_1 == v.node_2 for v in result.violations)
        )

    def test_disconnected_components_only_check_within_range_relationships(self):
        nodes = [
            Node("A", 0, 0),
            Node("B", 400, 0),
            Node("C", 800, 0),
            Node("D", 3000, 0),
            Node("E", 3400, 0),
        ]
        result = verify_schedule(nodes, {"A": 0, "B": 1, "C": 2, "D": 0, "E": 1})
        self.assertTrue(result.valid)
        self.assertEqual(result.direct_conflicts_checked, 3)
        self.assertEqual(result.two_hop_conflicts_checked, 1)

    def test_exact_500_meter_boundary_is_inclusive(self):
        result = verify_schedule([Node("A", 0, 0), Node("B", 300, 400)], {"A": 4, "B": 4})
        self.assertFalse(result.valid)
        self.assertEqual(result.direct_conflicts_checked, 1)
        self.assertEqual(result.violations[0].distance_m, 500.0)

    def test_500_meters_plus_epsilon_is_not_direct(self):
        result = verify_schedule(
            [Node("A", 0, 0), Node("B", 500.0 + 1e-9, 0)],
            {"A": 0, "B": 0},
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.direct_conflicts_checked, 0)

    def test_missing_node_color_is_reported(self):
        result = verify_schedule([Node("A", 0, 0), Node("B", 10, 0)], {"A": 0})
        self.assertFalse(result.valid)
        self.assertEqual(result.number_of_violations, 1)
        self.assertEqual(result.violations[0].conflict_type, "missing_color")
        self.assertEqual(result.violations[0].node_1, "B")

    def test_unexpected_color_mapping_node_is_reported(self):
        result = verify_schedule([Node("A", 0, 0)], {"A": 0, "ghost": 1})
        self.assertFalse(result.valid)
        self.assertEqual(result.violations[0].conflict_type, "unexpected_node")
        self.assertEqual(result.violations[0].node_1, "ghost")

    def test_invalid_color_values_are_reported(self):
        for value in (-1, True, 1.5, "1"):
            with self.subTest(color=value):
                result = verify_schedule([Node("A", 0, 0)], {"A": value})
                self.assertFalse(result.valid)
                self.assertEqual(result.violations[0].conflict_type, "invalid_color")

    def test_input_coordinates_and_color_mapping_are_unchanged(self):
        nodes = [Node("B", -3.5, 8.25), Node("A", 1000, -200)]
        colors = {"A": 1, "B": 0}
        original_coordinates = [(node.node_id, node.x_m, node.y_m) for node in nodes]
        original_colors = colors.copy()

        result = verify_schedule(nodes, colors)

        self.assertTrue(result.valid)
        self.assertTrue(result.coordinates_unchanged)
        self.assertEqual(
            [(node.node_id, node.x_m, node.y_m) for node in nodes],
            original_coordinates,
        )
        self.assertEqual(colors, original_colors)

    def test_rejects_invalid_range(self):
        result = verify_schedule([Node("A", 0, 0)], {"A": 0}, communication_range=-1)
        self.assertFalse(result.valid)
        self.assertEqual(result.violations[0].conflict_type, "invalid_range")


if __name__ == "__main__":
    unittest.main()
