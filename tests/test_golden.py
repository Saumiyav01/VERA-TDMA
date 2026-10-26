import json
import tempfile
import unittest
import warnings
from pathlib import Path

from vera_tdma.cli import build_result, load_cli_coordinates
from vera_tdma.coordinates import CoLocatedNodesWarning
from vera_tdma.golden import golden_cases, run_golden_tests


class GoldenTests(unittest.TestCase):
    def test_every_golden_case_passes(self):
        results = run_golden_tests()
        self.assertEqual(len(results), len(golden_cases()))
        self.assertEqual(len(results), 10)
        for result in results:
            with self.subTest(case=result.name):
                self.assertTrue(result.passed, result.detail)

    def test_known_slot_counts(self):
        expected = {
            "two nodes 100 m apart": 2,
            "A-B-C at 300 m spacing": 3,
            "A-B-C-D-E at 300 m spacing": 3,
            "two nodes at 499 m": 2,
            "two nodes at exactly 500 m (inclusive)": 2,
            "two nodes at 501 m": 1,
            "star with 5 leaves": 6,
            "two disconnected pairs more than 1000 m apart": 2,
            "duplicate coordinates are connected": 2,
        }
        slots = {result.name: result.slots for result in run_golden_tests()}
        for name, count in expected.items():
            with self.subTest(case=name):
                self.assertEqual(slots[name], count)

    def test_grid_is_computed_not_hard_coded(self):
        grid_case = [c for c in golden_cases() if c.name.startswith("4x4 grid")][0]
        self.assertIsNone(grid_case.expected_slots)

    def test_duplicate_coordinates_warn_and_are_scheduled_in_different_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup.json"
            path.write_text(json.dumps({"A": [5, 5], "B": [5, 5]}), encoding="utf-8")
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                nodes = load_cli_coordinates(path)
        self.assertTrue(any(w.category is CoLocatedNodesWarning for w in emitted))
        result = build_result(nodes)
        self.assertEqual(result["metrics"]["communication_links"], 1)
        self.assertEqual(result["metrics"]["optimized_frame_length"], 2)


if __name__ == "__main__":
    unittest.main()
