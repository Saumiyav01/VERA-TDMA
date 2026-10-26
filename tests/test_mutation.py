"""Mutation tests: deliberately corrupted schedules must be rejected."""

import unittest
from unittest.mock import patch

import networkx as nx

from vera_tdma import verification
from vera_tdma.coordinates import Node
from vera_tdma.verification import verify_matrix_consistency, verify_schedule


CHAIN = [Node("A", 0, 0), Node("B", 400, 0), Node("C", 800, 0), Node("D", 1200, 0)]
GOOD = {"A": 0, "B": 1, "C": 2, "D": 0}


def matrix_for(colors):
    slots = sorted(set(colors.values()))
    return {
        str(slot): {node: int(color == slot) for node, color in colors.items()}
        for slot in slots
    }


class VerifierMutationTests(unittest.TestCase):
    def test_unmutated_schedule_passes(self):
        self.assertTrue(verify_schedule(CHAIN, GOOD).valid)

    def test_direct_neighbor_sharing_a_slot_is_detected(self):
        result = verify_schedule(CHAIN, {**GOOD, "B": GOOD["A"]})
        self.assertFalse(result.valid)
        self.assertIn("direct", {v.conflict_type for v in result.violations})

    def test_two_hop_neighbor_sharing_a_slot_names_shared_receiver(self):
        result = verify_schedule(CHAIN, {**GOOD, "C": GOOD["A"]})
        self.assertFalse(result.valid)
        violation = [v for v in result.violations if v.node_2 == "C"][0]
        self.assertEqual(violation.conflict_type, "two_hop")
        self.assertEqual(violation.path, ("A", "B", "C"))

    def test_moving_a_coordinate_across_500_meters_changes_adjacency(self):
        colors = {"A": 0, "B": 0}
        far = verify_schedule([Node("A", 0, 0), Node("B", 501, 0)], colors)
        near = verify_schedule([Node("A", 0, 0), Node("B", 500, 0)], colors)
        self.assertTrue(far.valid)
        self.assertFalse(near.valid)

    def test_invalid_slot_indices_are_rejected(self):
        for bad in (-1, True, 2.0, "0", None):
            with self.subTest(slot=bad):
                result = verify_schedule(CHAIN, {**GOOD, "D": bad})
                self.assertFalse(result.valid)
                self.assertEqual(result.violations[0].conflict_type, "invalid_color")

    def test_missing_and_unknown_nodes_are_rejected(self):
        missing = {k: v for k, v in GOOD.items() if k != "C"}
        self.assertIn(
            "missing_color",
            {v.conflict_type for v in verify_schedule(CHAIN, missing).violations},
        )
        self.assertIn(
            "unexpected_node",
            {v.conflict_type for v in verify_schedule(CHAIN, {**GOOD, "Z": 1}).violations},
        )

    def test_disagreement_with_networkx_power_is_reported(self):
        verification._derive_geometry.cache_clear()
        try:
            with patch.object(verification.nx, "power", return_value=nx.Graph()):
                result = verify_schedule(CHAIN, GOOD)
        finally:
            verification._derive_geometry.cache_clear()
        self.assertFalse(result.valid)
        self.assertEqual(result.violations[0].conflict_type, "crosscheck_mismatch")


class MatrixMutationTests(unittest.TestCase):
    def test_consistent_matrix_passes(self):
        self.assertEqual(verify_matrix_consistency(matrix_for(GOOD), list(GOOD), GOOD), ())

    def test_flipped_entry_is_detected(self):
        matrix = matrix_for(GOOD)
        matrix["1"]["B"] = 0
        problems = verify_matrix_consistency(matrix, list(GOOD), GOOD)
        self.assertTrue(problems)
        self.assertEqual(problems[0].conflict_type, "matrix_mismatch")

    def test_node_in_two_slots_is_detected(self):
        matrix = matrix_for(GOOD)
        matrix["2"]["A"] = 1
        self.assertTrue(verify_matrix_consistency(matrix, list(GOOD), GOOD))

    def test_non_boolean_entry_is_detected(self):
        matrix = matrix_for(GOOD)
        matrix["0"]["A"] = 2
        self.assertTrue(verify_matrix_consistency(matrix, list(GOOD), GOOD))

    def test_slot_index_outside_matrix_is_detected(self):
        problems = verify_matrix_consistency(matrix_for(GOOD), list(GOOD), {**GOOD, "D": 9})
        self.assertTrue(problems)

    def test_negative_slot_index_is_detected(self):
        problems = verify_matrix_consistency(matrix_for(GOOD), list(GOOD), {**GOOD, "D": -1})
        self.assertIn("invalid_slot_index", {p.conflict_type for p in problems})


if __name__ == "__main__":
    unittest.main()
