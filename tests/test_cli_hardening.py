import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vera_tdma.cli import build_result, load_cli_coordinates, main
from vera_tdma.config import AppConfig, ConfigError
from vera_tdma.coordinates import CoordinateLoadError, ImplausibleCoordinateWarning, Node
from vera_tdma.optimization.exact_solver import solve_exact as real_solve_exact


class CliHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.directory = Path(self.tempdir.name)

    def write(self, name, content):
        path = self.directory / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def run_cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main([*map(str, args)])
        return code, output.getvalue(), errors.getvalue()

    # --- input format -------------------------------------------------------
    def test_specification_array_format_is_accepted(self):
        path = self.write("spec.json", {"Node_01": [0.0, 0.0], "Node_02": [300.0, 0.0]})
        code, report, errors = self.run_cli("--input", path)
        self.assertEqual(code, 0, errors)
        self.assertIn("Total Nodes: 2", report)

    def test_array_and_object_formats_give_identical_results(self):
        array = self.write("a.json", {"A": [0, 0], "B": [300, 0], "C": [600, 0]})
        mapping = self.write(
            "o.json",
            {"A": {"x": 0, "y": 0}, "B": {"x": 300, "y": 0}, "C": {"x": 600, "y": 0}},
        )
        first = build_result(load_cli_coordinates(array))
        second = build_result(load_cli_coordinates(mapping))
        self.assertEqual(first["input_hash"], second["input_hash"])
        self.assertEqual(first["schedule_hash"], second["schedule_hash"])

    def test_malformed_coordinate_shapes_are_rejected(self):
        for index, document in enumerate(
            ['{"A": [1]}', '{"A": [1, 2, 3]}', '{"A": "1,2"}', '{"A": {"x": 1, "y": 2, "z": 3}}',
             '{"A": [true, 1]}', '{"A": ["1", 2]}']
        ):
            with self.subTest(document=document):
                path = self.write(f"bad{index}.json", document)
                with self.assertRaises(CoordinateLoadError):
                    load_cli_coordinates(path)

    # --- robustness ---------------------------------------------------------
    def test_astronomically_large_integer_is_a_clean_error_not_a_traceback(self):
        path = self.write("big.json", '{"A": [1' + "0" * 400 + ', 0]}')
        code, _, errors = self.run_cli("--input", path)
        self.assertEqual(code, 2)
        self.assertIn("finite number", errors)
        self.assertNotIn("Traceback", errors)

    def test_identifiers_with_surrounding_whitespace_are_rejected(self):
        for name in (" A", "A ", "A\n", "\tA"):
            with self.subTest(identifier=name):
                path = self.write("ws.json", {name: [0, 0]})
                with self.assertRaisesRegex(CoordinateLoadError, "whitespace"):
                    load_cli_coordinates(path)

    def test_implausibly_large_coordinates_warn(self):
        path = self.write("far.json", {"A": [0, 0], "B": [1e200, 0]})
        with self.assertWarns(ImplausibleCoordinateWarning):
            load_cli_coordinates(path)
        code, _, errors = self.run_cli("--input", path)
        self.assertEqual(code, 0)
        self.assertIn("implausibly large", errors)

    # --- publication gate ---------------------------------------------------
    def test_optimizer_crash_falls_back_to_verified_baseline(self):
        nodes = [Node("A", 0, 0), Node("B", 100, 0)]
        with patch("vera_tdma.cli.eliminate_classes", side_effect=RuntimeError("boom")):
            result = build_result(nodes)
        self.assertEqual(result["publication"]["status"], "PUBLISHED_BASELINE")
        self.assertEqual(result["verification"]["status"], "PASS")
        self.assertIn("boom", result["rejected_optimization"]["reason"])

    def test_normal_run_is_published_without_rejections(self):
        result = build_result([Node("A", 0, 0), Node("B", 100, 0)])
        self.assertEqual(result["publication"]["status"], "PUBLISHED")
        self.assertIsNone(result["rejected_optimization"])

    # --- report -------------------------------------------------------------
    def test_report_contains_required_conflict_check_lines(self):
        path = self.write("n.json", {"A": [0, 0], "B": [300, 0], "C": [600, 0]})
        code, report, _ = self.run_cli("--input", path)
        self.assertEqual(code, 0)
        for line in (
            "Schedule verified conflict-free.",
            "Direct-link conflicts checked: 0 violations.",
            "Two-hop hidden-terminal conflicts checked: 0 violations.",
            "Execution finalized cleanly.",
            "Golden Self-Tests: PASS (10/10)",
        ):
            self.assertIn(line, report)

    def test_report_shows_lower_bound_gap_when_exact_solver_is_skipped(self):
        path = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        code, report, _ = self.run_cli("--input", path, "--no-exact")
        self.assertEqual(code, 0)
        self.assertIn("Proven Lower Bound: 2", report)
        self.assertIn("Lower-Bound Gap (not an exact gap): 0", report)

    def test_exact_runtime_is_formatted_compactly(self):
        path = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        _, report, _ = self.run_cli("--input", path)
        runtime_line = [l for l in report.splitlines() if l.strip().startswith("Runtime:")][0]
        self.assertRegex(runtime_line, r"Runtime: \d+\.\d{4} s$")

    def test_bounded_exact_status_never_claims_an_optimum(self):
        import random

        def one_node_budget(graph, **_options):
            return real_solve_exact(graph, max_search_nodes=1)

        found = False
        for seed in range(60):
            rng = random.Random(seed)
            nodes = [Node(f"N{i}", rng.uniform(0, 1200), rng.uniform(0, 1200)) for i in range(14)]
            with patch("vera_tdma.cli.solve_exact", side_effect=one_node_budget):
                exact = build_result(nodes, run_exact=True)["exact_solver"]
            if exact["status"] != "BOUNDED":
                continue
            found = True
            self.assertIsNone(exact["chromatic_number"])
            self.assertIsNone(exact["optimality_gap"])
            self.assertIsNone(exact["relative_optimality_gap"])
            self.assertLess(exact["lower_bound"], exact["upper_bound"])
            break
        self.assertTrue(found, "no instance needed more than one search node")

    # --- configuration wiring -----------------------------------------------
    def config_file(self, text):
        return self.write("cfg.yaml", text)

    def test_config_range_is_honoured(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        config = self.config_file("network:\n  communication_range_m: 250\n")
        _, report, _ = self.run_cli("--input", nodes, "--config", config)
        self.assertIn("Communication Links: 0", report)
        _, default_report, _ = self.run_cli("--input", nodes)
        self.assertIn("Communication Links: 1", default_report)

    def test_config_non_default_range_runs_the_whole_pipeline(self):
        nodes = self.write(
            "n.json", {"A": [0, 0], "B": [200, 0], "C": [400, 0], "D": [600, 0]}
        )
        config = self.config_file("network:\n  communication_range_m: 250\n")
        code, report, errors = self.run_cli("--input", nodes, "--config", config)
        self.assertEqual(code, 0, errors)
        self.assertIn("Verification Status: PASS", report)

    def test_config_strict_boundary_policy(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 400]})
        config = self.config_file("network:\n  inclusive_range: false\n")
        _, strict, _ = self.run_cli("--input", nodes, "--config", config)
        _, inclusive, _ = self.run_cli("--input", nodes)
        self.assertIn("Communication Links: 0", strict)
        self.assertIn("Communication Links: 1", inclusive)

    def test_config_can_disable_refinement_stages_and_exact_solver(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        config = self.config_file(
            "scheduling:\n  enable_class_elimination: false\n  enable_kempe_refinement: false\n"
            "exact_solver:\n  enabled: false\n"
        )
        code, report, _ = self.run_cli("--input", nodes, "--config", config)
        self.assertEqual(code, 0)
        self.assertIn("Class Elimination Attempts: N/A (did not run)", report)
        self.assertIn("Status: SKIPPED", report)

    def test_config_exact_enabled_runs_exact_without_flag_and_flag_overrides(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        _, default_report, _ = self.run_cli("--input", nodes)
        self.assertIn("Status: EXACT", default_report)
        _, off_report, _ = self.run_cli("--input", nodes, "--no-exact")
        self.assertIn("Status: SKIPPED", off_report)

    def test_config_exact_max_nodes_applies_unless_overridden(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0], "C": [600, 0]})
        config = self.config_file("exact_solver:\n  max_nodes: 2\n")
        _, limited, _ = self.run_cli("--input", nodes, "--config", config)
        self.assertIn("Status: BOUNDED/SKIPPED", limited)
        _, overridden, _ = self.run_cli(
            "--input", nodes, "--config", config, "--exact-max-nodes", "3"
        )
        self.assertIn("Status: EXACT", overridden)

    def test_config_output_flags_remove_sections(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        config = self.config_file(
            "output:\n  include_slot_node_matrix: false\n  include_exact_solver_metrics: false\n"
            "  include_optimization_metrics: false\n"
        )
        out = self.directory / "out.json"
        code, report, _ = self.run_cli("--input", nodes, "--config", config, "--output", out)
        self.assertEqual(code, 0)
        self.assertNotIn("Boolean Matrix", report)
        self.assertNotIn("Exact Solver", report)
        self.assertNotIn("Optimization Audit", report)
        saved = json.loads(out.read_text(encoding="utf-8"))
        for removed in ("slot_matrix", "exact_solver", "optimization_audit"):
            self.assertNotIn(removed, saved)
        self.assertIn("node_to_slot", saved)

    def test_config_can_skip_golden_self_tests(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        config = self.config_file("validation:\n  run_golden_tests: false\n")
        _, report, _ = self.run_cli("--input", nodes, "--config", config)
        self.assertNotIn("Golden Self-Tests", report)

    def test_failing_golden_self_test_halts_with_exit_code_3(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        from vera_tdma.golden import GoldenResult

        failing = (GoldenResult("synthetic", False, 1, "expected 2 slots, got 1"),)
        with patch("vera_tdma.cli.run_golden_tests", return_value=failing):
            code, report, errors = self.run_cli("--input", nodes)
        self.assertEqual(code, 3)
        self.assertEqual(report, "")
        self.assertIn("golden self-test failed: synthetic", errors)

    def test_fail_on_verification_error_only_selects_exit_status(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        config = self.config_file("validation:\n  fail_on_verification_error: false\n")
        from vera_tdma.verification import ScheduleViolation, VerificationResult

        failing = VerificationResult(
            2, 1, 0, (ScheduleViolation("A", "B", 0, 0, "direct", message="x"),), True
        )
        with patch("vera_tdma.cli.verify_schedule", side_effect=[failing]):
            code, report, _ = self.run_cli("--input", nodes, "--config", config)
        self.assertEqual(code, 0)
        self.assertIn("SCHEDULE WITHHELD", report)
        self.assertNotIn("Node -> Slot", report)

    def test_input_falls_back_to_configured_path(self):
        nodes = self.write("n.json", {"A": [0, 0], "B": [300, 0]})
        config = self.config_file(f"paths:\n  input_nodes: '{nodes.as_posix()}'\n")
        code, report, _ = self.run_cli("--config", config)
        self.assertEqual(code, 0)
        self.assertIn("Total Nodes: 2", report)

    def test_invalid_config_is_a_clean_error(self):
        nodes = self.write("n.json", {"A": [0, 0]})
        config = self.config_file("network:\n  communication_range_m: -5\n")
        code, _, errors = self.run_cli("--input", nodes, "--config", config)
        self.assertEqual(code, 2)
        self.assertIn("non-negative", errors)

    def test_check_conflicts_cannot_be_disabled(self):
        with self.assertRaisesRegex(ConfigError, "check_conflicts cannot be disabled"):
            AppConfig.from_mapping({"validation": {"check_conflicts": False}})

    def test_optimization_time_budget_is_validated_and_applied(self):
        with self.assertRaisesRegex(ConfigError, "greater than zero"):
            AppConfig.from_mapping({"scheduling": {"optimization_time_budget_seconds": 0}})
        config = AppConfig.from_mapping({"scheduling": {"optimization_time_budget_seconds": 5}})
        self.assertEqual(config.scheduling.optimization_time_budget_seconds, 5.0)
        result = build_result([Node("A", 0, 0), Node("B", 300, 0)], config=config)
        self.assertFalse(result["optimization_audit"]["optimization_budget_exhausted"])
        self.assertEqual(result["verification"]["status"], "PASS")

    def test_matrix_inconsistency_withholds_the_schedule(self):
        with patch("vera_tdma.cli.verify_matrix_consistency") as check:
            from vera_tdma.verification import ScheduleViolation

            check.return_value = (
                ScheduleViolation(None, None, None, None, "matrix_mismatch", message="corrupt"),
            )
            result = build_result([Node("A", 0, 0), Node("B", 300, 0)])
        self.assertEqual(result["publication"]["status"], "WITHHELD")
        self.assertEqual(result["verification"]["status"], "FAIL")
        self.assertEqual(result["node_to_slot"], {})
        self.assertIsNone(result["schedule_hash"])


if __name__ == "__main__":
    unittest.main()
