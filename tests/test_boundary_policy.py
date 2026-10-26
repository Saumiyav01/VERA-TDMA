import math
import random
import unittest

from vera_tdma.communication_graph import build_communication_graph, within_range
from vera_tdma.conflict_graph import build_conflict_graph
from vera_tdma.coordinates import Node
from vera_tdma.optimization.class_elimination import eliminate_classes
from vera_tdma.optimization.kempe import kempe_refine
from vera_tdma.scheduling.dsatur import dsatur_color
from vera_tdma.verification import verify_schedule


class BoundaryPolicyTests(unittest.TestCase):
    def test_reported_fractional_case_is_consistent_between_graph_and_verifier(self):
        a = Node("A", 0.0, 0.0)
        b = Node("B", 467.17233131436933, 178.1853328820786)
        linked = build_communication_graph([a, b]).number_of_edges() == 1
        verifier_conflict = not verify_schedule([a, b], {"A": 0, "B": 0}).valid
        self.assertEqual(linked, verifier_conflict)
        self.assertTrue(verify_schedule([a, b], {"A": 0, "B": 1}).valid)

    def test_graph_and_verifier_agree_on_random_boundary_cases(self):
        rng = random.Random(1)
        origin = Node("A", 0.0, 0.0)
        for _ in range(3000):
            angle = rng.random() * 2 * math.pi
            radius = 500 * (1 + rng.choice([-1, 1]) * rng.choice([0, 1e-16, 1e-15, 1e-12, 1e-9]))
            other = Node("B", radius * math.cos(angle), radius * math.sin(angle))
            linked = build_communication_graph([origin, other]).number_of_edges() == 1
            conflict = not verify_schedule([origin, other], {"A": 0, "B": 0}).valid
            self.assertEqual(linked, conflict, other)

    def test_exact_500_is_inclusive_and_strict_flag_excludes_it(self):
        a, b = Node("A", 0, 0), Node("B", 300, 400)
        self.assertEqual(build_communication_graph([a, b]).number_of_edges(), 1)
        self.assertEqual(build_communication_graph([a, b], inclusive=False).number_of_edges(), 0)
        self.assertFalse(verify_schedule([a, b], {"A": 0, "B": 0}).valid)
        self.assertTrue(verify_schedule([a, b], {"A": 0, "B": 0}, inclusive_range=False).valid)

    def test_within_range_uses_exact_arithmetic_just_outside_the_range(self):
        a = Node("A", 0.0, 0.0)
        self.assertTrue(within_range(a, Node("B", 500.0, 0.0), 500.0))
        self.assertFalse(within_range(a, Node("B", math.nextafter(500.0, 600.0), 0.0), 500.0))
        self.assertTrue(within_range(a, Node("B", math.nextafter(500.0, 0.0), 0.0), 500.0))

    def test_huge_coordinates_do_not_overflow(self):
        a, b = Node("A", 1e200, 0.0), Node("B", -1e200, 0.0)
        self.assertFalse(within_range(a, b, 500.0))
        self.assertEqual(build_communication_graph([a, b]).number_of_edges(), 0)

    def test_non_default_range_flows_through_every_stage(self):
        nodes = [Node("A", 0, 0), Node("B", 200, 0), Node("C", 400, 0), Node("D", 600, 0)]
        graph = build_communication_graph(nodes, 250.0)
        conflict = build_conflict_graph(graph)
        coloring, _ = dsatur_color(conflict)
        # These used to raise "initial coloring failed verification" because the
        # optimizers always verified with the default 500 m range.
        eliminated = eliminate_classes(conflict, coloring)
        refined = kempe_refine(conflict, eliminated.coloring)
        self.assertTrue(refined.final_verification.valid)
        self.assertTrue(verify_schedule(nodes, refined.coloring, 250.0).valid)

    def test_explicit_range_overrides_graph_metadata(self):
        nodes = [Node("A", 0, 0), Node("B", 200, 0)]
        conflict = build_conflict_graph(build_communication_graph(nodes, 250.0))
        coloring, _ = dsatur_color(conflict)
        result = eliminate_classes(conflict, coloring, communication_range_m=250.0)
        self.assertTrue(result.final_verification.valid)


if __name__ == "__main__":
    unittest.main()
