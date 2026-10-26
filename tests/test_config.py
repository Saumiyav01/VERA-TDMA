import tempfile
import unittest
from pathlib import Path

from vera_tdma.config import AppConfig, ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_match_phase_one_configuration(self):
        config = AppConfig()
        self.assertEqual(config.paths.input_nodes, Path("data/nodes.json"))
        self.assertEqual(config.network.communication_range_m, 500.0)
        self.assertTrue(config.network.inclusive_range)
        self.assertEqual(config.scheduling.algorithm, "vera-tdma")
        self.assertEqual(config.scheduling.initial_coloring, "dsatur")
        self.assertTrue(config.scheduling.enable_class_elimination)
        self.assertTrue(config.scheduling.enable_kempe_refinement)
        self.assertTrue(config.scheduling.deterministic)
        self.assertEqual(config.exact_solver.timeout_seconds, 10.0)
        self.assertEqual(config.exact_solver.max_nodes, 16)
        self.assertTrue(config.validation.independent_verification)
        self.assertTrue(config.output.include_node_slot_mapping)

    def test_load_yaml_with_partial_values_and_default_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "custom.yaml"
            config_path.write_text(
                "network:\n  communication_range_m: 250\n"
                "paths:\n  input_nodes: custom/nodes.csv\n",
                encoding="utf-8",
            )
            config = load_config(config_path)

        self.assertEqual(config.network.communication_range_m, 250.0)
        self.assertEqual(config.paths.input_nodes, Path("custom/nodes.csv"))
        self.assertEqual(config.exact_solver.timeout_seconds, 10.0)

    def test_loads_project_configuration_file(self):
        config = load_config(Path(__file__).parents[1] / "config.yaml")

        self.assertEqual(config.paths.output_dir, Path("outputs"))
        self.assertEqual(config.paths.report_file, Path("outputs/report.txt"))
        self.assertEqual(config.network.communication_range_m, 500.0)
        self.assertTrue(config.network.inclusive_range)
        self.assertEqual(config.scheduling.algorithm, "vera-tdma")
        self.assertEqual(config.scheduling.initial_coloring, "dsatur")
        self.assertTrue(config.scheduling.enable_class_elimination)
        self.assertTrue(config.scheduling.enable_kempe_refinement)
        self.assertTrue(config.scheduling.deterministic)
        self.assertTrue(config.exact_solver.enabled)
        self.assertEqual(config.exact_solver.timeout_seconds, 10.0)
        self.assertEqual(config.exact_solver.max_nodes, 16)
        self.assertTrue(config.validation.independent_verification)
        self.assertTrue(config.validation.check_matrix_consistency)
        self.assertTrue(config.validation.check_conflicts)
        self.assertTrue(config.validation.run_golden_tests)
        self.assertTrue(config.validation.fail_on_verification_error)
        self.assertTrue(config.output.include_node_slot_mapping)
        self.assertTrue(config.output.include_slot_node_matrix)
        self.assertTrue(config.output.include_conflict_details)
        self.assertTrue(config.output.include_optimization_metrics)
        self.assertTrue(config.output.include_exact_solver_metrics)

    def test_rejects_invalid_values(self):
        invalid = [
            ({"network": {"communication_range_m": -1}}, "non-negative"),
            ({"network": {"communication_range_m": float("inf")}}, "finite"),
            ({"network": {"communication_range_m": "500"}}, "must be a number"),
            ({"network": {"inclusive_range": "yes"}}, "must be a boolean"),
            ({"scheduling": {"algorithm": "unknown"}}, "vera-tdma"),
            ({"scheduling": {"initial_coloring": "random"}}, "dsatur"),
            ({"exact_solver": {"timeout_seconds": 0}}, "greater than zero"),
            ({"exact_solver": {"timeout_seconds": True}}, "must be a number"),
            ({"exact_solver": {"max_nodes": 0}}, "positive integer"),
            ({"exact_solver": {"max_nodes": 1.5}}, "positive integer"),
            ({"validation": {"independent_verification": False}}, "cannot be disabled"),
        ]
        for mapping, error_text in invalid:
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(ConfigError, error_text):
                    AppConfig.from_mapping(mapping)

    def test_rejects_unknown_fields_and_non_mapping_sections(self):
        with self.assertRaisesRegex(ConfigError, "unknown field"):
            AppConfig.from_mapping({"network": {"range": 500}})
        with self.assertRaisesRegex(ConfigError, "must be a mapping"):
            AppConfig.from_mapping({"output": []})
        with self.assertRaisesRegex(ConfigError, "unknown configuration section"):
            AppConfig.from_mapping({"unrecognized": {}})

    def test_rejects_invalid_yaml_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.yaml"
            config_path.write_text("network: [", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "could not load configuration"):
                load_config(config_path)
            with self.assertRaisesRegex(ConfigError, "could not load configuration"):
                load_config(Path(directory) / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
