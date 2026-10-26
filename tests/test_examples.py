import importlib.util
import unittest
from pathlib import Path

from vera_tdma.cli import build_result, load_cli_coordinates

ROOT = Path(__file__).resolve().parents[1]


class ShippedDataTests(unittest.TestCase):
    def test_optimizer_example_shows_a_measured_improvement_proven_optimal(self):
        nodes = load_cli_coordinates(ROOT / "data" / "example_optimizer_helps.json")
        result = build_result(nodes, run_exact=True)
        metrics = result["metrics"]
        self.assertEqual(metrics["initial_dsatur_frame_length"], 8)
        self.assertEqual(metrics["optimized_frame_length"], 7)
        self.assertEqual(result["exact_solver"]["chromatic_number"], 7)
        self.assertEqual(result["exact_solver"]["optimality_gap"], 0)
        self.assertEqual(metrics["lower_bound"], 7)  # optimality provable without the solver
        self.assertGreaterEqual(result["optimization_audit"]["accepted_eliminations"], 1)
        self.assertEqual(result["verification"]["status"], "PASS")

    def test_sample_sixteen_node_grid_runs_end_to_end(self):
        nodes = load_cli_coordinates(ROOT / "data" / "sample_16_nodes_grid.json")
        self.assertEqual(len(nodes), 16)
        result = build_result(nodes, run_exact=True)
        self.assertEqual(result["exact_solver"]["status"], "EXACT")
        self.assertEqual(result["metrics"]["optimized_frame_length"], 9)
        self.assertEqual(result["exact_solver"]["optimality_gap"], 0)

    def test_benchmark_runner_smoke(self):
        spec = importlib.util.spec_from_file_location(
            "run_benchmarks", ROOT / "experiments" / "run_benchmarks.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = module.run([8], 1, 16)
        self.assertEqual(len(rows), len(module.FAMILIES))
        self.assertTrue(all(row["verification"] == "PASS" for row in rows))
        self.assertIn("DSATUR", module.summarize(rows))


if __name__ == "__main__":
    unittest.main()
