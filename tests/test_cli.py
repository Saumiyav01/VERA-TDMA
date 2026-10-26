import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vera_tdma.cli import build_result, main
from vera_tdma.coordinates import Node
from vera_tdma.verification import ScheduleViolation, VerificationResult


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.input_path = Path(self.tempdir.name) / "nodes.json"
        self.input_path.write_text(
            json.dumps(
                {
                    "node10": {"x": 300.0, "y": 0.0},
                    "node2": {"x": 0.0, "y": 0.0},
                    "node1": {"x": 100.0, "y": 0.0},
                }
            ),
            encoding="utf-8",
        )

    def run_cli(self, *args):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main([*args])
        return code, output.getvalue(), errors.getvalue()

    def test_valid_json_input_and_network_counts(self):
        code, report, _ = self.run_cli("--input", str(self.input_path))
        self.assertEqual(code, 0)
        result = build_result(
            [
                Node("node1", 100, 0),
                Node("node2", 0, 0),
                Node("node10", 300, 0),
            ]
        )
        self.assertEqual(result["metrics"]["total_nodes"], 3)
        self.assertEqual(result["metrics"]["communication_links"], 3)
        self.assertEqual(result["metrics"]["distance_2_conflict_edges"], 3)
        self.assertIn("Total Nodes: 3", report)
        self.assertIn("Communication Links: 3", report)
        self.assertIn("Distance-2 Conflict Edges: 3", report)

    def test_dsatur_frame_and_optimized_frame_lengths(self):
        result = build_result([Node("A", 0, 0), Node("B", 100, 0)])
        self.assertEqual(result["metrics"]["initial_dsatur_frame_length"], 2)
        self.assertEqual(result["metrics"]["optimized_frame_length"], 2)

    def test_node_mapping_and_boolean_slot_matrix(self):
        result = build_result(
            [Node("Node10", 300, 0), Node("Node2", 0, 0), Node("Node1", 100, 0)]
        )
        self.assertEqual(list(result["node_to_slot"]), ["Node1", "Node2", "Node10"])
        self.assertEqual(result["slot_matrix"]["node_order"], ["Node1", "Node2", "Node10"])
        rows = result["slot_matrix"]["rows"]
        self.assertEqual(len(rows), 3)
        for node_id, slot in result["node_to_slot"].items():
            self.assertEqual(rows[str(slot)][node_id], 1)
            self.assertEqual(sum(rows[str(row_slot)][node_id] for row_slot in rows), 1)
        for row in rows.values():
            self.assertTrue(all(value in (0, 1) for value in row.values()))
        self.assertEqual(result["slot_matrix"]["legend"]["1"], "assigned to slot")

    def test_independent_verification_result_is_reported(self):
        code, report, _ = self.run_cli("--input", str(self.input_path))
        self.assertEqual(code, 0)
        self.assertIn("Verification Status: PASS", report)
        self.assertIn("Schedule verified conflict-free", report)
        result = build_result([Node("A", 0, 0), Node("B", 100, 0)])
        self.assertTrue(result["verification"]["valid"])
        self.assertEqual(result["verification"]["status"], "PASS")

    def test_exact_solver_runs_for_small_graph(self):
        result = build_result(
            [Node("A", 0, 0), Node("B", 100, 0)],
            run_exact=True,
            exact_max_nodes=2,
        )
        self.assertEqual(result["exact_solver"]["status"], "EXACT")
        self.assertEqual(result["exact_solver"]["chromatic_number"], 2)
        self.assertIsInstance(result["exact_solver"]["nodes_explored"], int)
        self.assertGreaterEqual(result["exact_solver"]["runtime_seconds"], 0)

    def test_exact_solver_is_bounded_above_node_limit(self):
        result = build_result(
            [Node("A", 0, 0), Node("B", 100, 0), Node("C", 2000, 0)],
            run_exact=True,
            exact_max_nodes=2,
        )
        self.assertEqual(result["exact_solver"]["status"], "BOUNDED/SKIPPED")
        self.assertIsNone(result["exact_solver"]["chromatic_number"])

    def test_input_hash_is_canonical_across_node_order(self):
        first = build_result([Node("B", 100, 0), Node("A", 0, 0)])
        second = build_result([Node("A", 0.0, 0.0), Node("B", 100.0, 0.0)])
        self.assertEqual(first["input_hash"], second["input_hash"])

    def test_schedule_hash_is_canonical_and_repeatable(self):
        nodes = [Node("B", 100, 0), Node("A", 0, 0)]
        first = build_result(nodes)
        second = build_result(nodes)
        self.assertEqual(first["schedule_hash"], second["schedule_hash"])
        self.assertEqual(first["node_to_slot"], second["node_to_slot"])

    def test_repeated_execution_has_identical_deterministic_result_fields(self):
        nodes = [Node("A", 0, 0), Node("B", 100, 0), Node("C", 1000, 0)]
        first = build_result(nodes)
        second = build_result(nodes)
        for key in (
            "input_hash",
            "schedule_hash",
            "metrics",
            "verification",
            "node_to_slot",
            "slot_matrix",
            "optimization_audit",
            "exact_solver",
        ):
            self.assertEqual(first[key], second[key])

    def test_class_elimination_attempts_are_measured_not_fabricated(self):
        # Two directly connected nodes need two slots; each of the two slot
        # classes is tried once and neither can be removed.
        result = build_result([Node("A", 0, 0), Node("B", 100, 0)])
        self.assertEqual(result["optimization_audit"]["class_elimination_attempts"], 2)
        self.assertEqual(result["optimization_audit"]["accepted_eliminations"], 0)
        self.assertNotIn("class_elimination_attempts_note", result["optimization_audit"])

    def test_malformed_json_returns_nonzero_and_useful_error(self):
        self.input_path.write_text("{", encoding="utf-8")
        code, _, errors = self.run_cli("--input", str(self.input_path))
        self.assertNotEqual(code, 0)
        self.assertIn("could not parse JSON", errors)

    def test_invalid_coordinates_return_nonzero(self):
        self.input_path.write_text('{"A":{"x":"bad","y":0}}', encoding="utf-8")
        code, _, errors = self.run_cli("--input", str(self.input_path))
        self.assertNotEqual(code, 0)
        self.assertIn("finite number", errors)

    def test_missing_fields_duplicate_nodes_and_empty_input_are_rejected(self):
        self.input_path.write_text('{"A":{"x":0}}', encoding="utf-8")
        self.assertNotEqual(self.run_cli("--input", str(self.input_path))[0], 0)

        self.input_path.write_text('{"A":{"x":0,"y":0},"A":{"x":1,"y":1}}', encoding="utf-8")
        code, _, errors = self.run_cli("--input", str(self.input_path))
        self.assertNotEqual(code, 0)
        self.assertIn("duplicate node identifier", errors)

        self.input_path.write_text("{}", encoding="utf-8")
        code, _, errors = self.run_cli("--input", str(self.input_path))
        self.assertNotEqual(code, 0)
        self.assertIn("at least one node", errors)

    def test_output_json_contains_required_machine_fields(self):
        output_path = Path(self.tempdir.name) / "result.json"
        code, _, _ = self.run_cli(
            "--input", str(self.input_path), "--output", str(output_path), "--exact"
        )
        self.assertEqual(code, 0)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        for field in (
            "input_hash",
            "schedule_hash",
            "metrics",
            "verification",
            "violations",
            "node_to_slot",
            "slot_matrix",
            "optimization_audit",
            "exact_solver",
        ):
            self.assertIn(field, result)

    def test_baseline_verification_failure_withholds_schedule_and_never_claims_success(self):
        violation = ScheduleViolation(
            "A", "B", 0, 0, "direct", 100.0, ("A", "B"), "same slot conflict"
        )
        failing = VerificationResult(2, 1, 0, (violation,), True)
        output_path = Path(self.tempdir.name) / "withheld.json"
        with patch("vera_tdma.cli.verify_schedule", side_effect=[failing]):
            code, report, errors = self.run_cli(
                "--input",
                str(self.input_path),
                "--output",
                str(output_path),
            )
        self.assertNotEqual(code, 0)
        self.assertIn("VERIFICATION: FAIL", report)
        self.assertIn("same slot conflict", report)
        self.assertIn("SCHEDULE WITHHELD", report)
        self.assertNotIn("Schedule verified conflict-free", report)
        self.assertNotIn("Node -> Slot", report)
        self.assertFalse(output_path.exists())
        self.assertIn("no verified schedule exists", errors)

    def test_failed_optimized_schedule_falls_back_to_verified_baseline(self):
        violation = ScheduleViolation(
            "A", "B", 0, 0, "direct", 100.0, ("A", "B"), "same slot conflict"
        )
        passing = VerificationResult(2, 1, 0, (), True)
        failing = VerificationResult(2, 1, 0, (violation,), True)
        with patch("vera_tdma.cli.verify_schedule", side_effect=[passing, failing]):
            result = build_result([Node("A", 0, 0), Node("B", 100, 0)])
        self.assertEqual(result["publication"]["status"], "PUBLISHED_BASELINE")
        self.assertEqual(result["verification"]["status"], "PASS")
        self.assertEqual(result["violations"], [])
        self.assertEqual(
            result["rejected_optimization"]["violations"][0]["message"],
            "same slot conflict",
        )
        self.assertEqual(set(result["node_to_slot"]), {"A", "B"})

    def test_verification_unavailable_is_explicit_and_never_claims_success(self):
        with patch(
            "vera_tdma.cli.verify_schedule",
            side_effect=ValueError("verifier unavailable"),
        ):
            code, report, _ = self.run_cli("--input", str(self.input_path))
        self.assertNotEqual(code, 0)
        self.assertIn("Verification Status: NOT RUN", report)
        self.assertIn("VERIFICATION: NOT RUN", report)
        self.assertIn("SCHEDULE WITHHELD", report)
        self.assertNotIn("Schedule verified conflict-free", report)

    def test_exact_solver_skip_is_explicit_without_exact_flag(self):
        result = build_result([Node("A", 0, 0)])
        self.assertEqual(result["exact_solver"]["status"], "SKIPPED")
        self.assertFalse(result["exact_solver"]["requested"])

    def test_invalid_exact_limit_returns_nonzero(self):
        code, _, errors = self.run_cli(
            "--input", str(self.input_path), "--exact-max-nodes", "0"
        )
        self.assertNotEqual(code, 0)
        self.assertIn("positive integer", errors)


if __name__ == "__main__":
    unittest.main()
