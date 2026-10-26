"""Typed application configuration and startup validation."""

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when configuration is malformed or contains invalid values."""


@dataclass(frozen=True)
class PathsConfig:
    input_nodes: Path = Path("data/nodes.json")
    output_dir: Path = Path("outputs/")
    report_file: Path = Path("outputs/report.txt")


@dataclass(frozen=True)
class NetworkConfig:
    communication_range_m: float = 500.0
    inclusive_range: bool = True


@dataclass(frozen=True)
class SchedulingConfig:
    algorithm: str = "vera-tdma"
    initial_coloring: str = "dsatur"
    enable_class_elimination: bool = True
    enable_kempe_refinement: bool = True
    deterministic: bool = True
    # Wall-clock budget for class elimination + Kempe refinement; None = unlimited
    # (deterministic). On expiry the best verified schedule so far is kept.
    optimization_time_budget_seconds: float | None = None


@dataclass(frozen=True)
class ExactSolverConfig:
    enabled: bool = True
    timeout_seconds: float = 10.0
    max_nodes: int = 16


@dataclass(frozen=True)
class ValidationConfig:
    independent_verification: bool = True
    check_matrix_consistency: bool = True
    check_conflicts: bool = True
    run_golden_tests: bool = True
    fail_on_verification_error: bool = True


@dataclass(frozen=True)
class OutputConfig:
    include_node_slot_mapping: bool = True
    include_slot_node_matrix: bool = True
    include_conflict_details: bool = True
    include_optimization_metrics: bool = True
    include_exact_solver_metrics: bool = True


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    exact_solver: ExactSolverConfig = field(default_factory=ExactSolverConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        distance = self.network.communication_range_m
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ConfigError("network.communication_range_m must be a number")
        if not math.isfinite(distance):
            raise ConfigError("network.communication_range_m must be finite")
        if distance < 0:
            raise ConfigError("network.communication_range_m must be non-negative")
        if type(self.network.inclusive_range) is not bool:
            raise ConfigError("network.inclusive_range must be a boolean")
        if self.scheduling.algorithm != "vera-tdma":
            raise ConfigError("scheduling.algorithm must be 'vera-tdma'")
        if self.scheduling.initial_coloring != "dsatur":
            raise ConfigError("scheduling.initial_coloring must be 'dsatur'")
        for section_name in ("scheduling", "exact_solver", "validation", "output"):
            section = getattr(self, section_name)
            for item in fields(section):
                if type(item.default) is bool and type(getattr(section, item.name)) is not bool:
                    raise ConfigError(f"{section_name}.{item.name} must be a boolean")
        if not self.validation.independent_verification:
            raise ConfigError("validation.independent_verification cannot be disabled")
        if not self.validation.check_conflicts:
            raise ConfigError("validation.check_conflicts cannot be disabled")
        budget = self.scheduling.optimization_time_budget_seconds
        if budget is not None:
            if isinstance(budget, bool) or not isinstance(budget, (int, float)):
                raise ConfigError("scheduling.optimization_time_budget_seconds must be a number")
            if not math.isfinite(budget) or budget <= 0:
                raise ConfigError(
                    "scheduling.optimization_time_budget_seconds must be greater than zero"
                )
        timeout = self.exact_solver.timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ConfigError("exact_solver.timeout_seconds must be a number")
        if not math.isfinite(timeout):
            raise ConfigError("exact_solver.timeout_seconds must be finite")
        if self.exact_solver.timeout_seconds <= 0:
            raise ConfigError("exact_solver.timeout_seconds must be greater than zero")
        if type(self.exact_solver.max_nodes) is not int or self.exact_solver.max_nodes <= 0:
            raise ConfigError("exact_solver.max_nodes must be a positive integer")
        for item in fields(self.paths):
            value = getattr(self.paths, item.name)
            if not isinstance(value, Path) or not str(value).strip():
                raise ConfigError(f"paths.{item.name} must be a non-empty path")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AppConfig":
        """Build configuration from a partial mapping, using documented defaults."""
        if not isinstance(mapping, Mapping):
            raise ConfigError("configuration root must be a mapping")

        section_types = {
            "paths": PathsConfig,
            "network": NetworkConfig,
            "scheduling": SchedulingConfig,
            "exact_solver": ExactSolverConfig,
            "validation": ValidationConfig,
            "output": OutputConfig,
        }
        unknown_sections = set(mapping) - set(section_types)
        if unknown_sections:
            raise ConfigError(f"unknown configuration section(s): {', '.join(sorted(unknown_sections))}")

        parsed: dict[str, Any] = {}
        for section_name, section_type in section_types.items():
            values = mapping.get(section_name, {})
            if not isinstance(values, Mapping):
                raise ConfigError(f"{section_name} must be a mapping")
            known_fields = {item.name for item in fields(section_type)}
            unknown_fields = set(values) - known_fields
            if unknown_fields:
                raise ConfigError(
                    f"unknown field(s) in {section_name}: {', '.join(sorted(unknown_fields))}"
                )
            converted = dict(values)
            if section_name == "paths":
                for field_name, value in converted.items():
                    if not isinstance(value, (str, Path)) or not str(value).strip():
                        raise ConfigError(f"paths.{field_name} must be a non-empty path")
                    converted[field_name] = Path(value)
            if section_name == "network" and "communication_range_m" in converted:
                converted["communication_range_m"] = _number(
                    converted["communication_range_m"], "network.communication_range_m"
                )
            if (
                section_name == "scheduling"
                and converted.get("optimization_time_budget_seconds") is not None
            ):
                converted["optimization_time_budget_seconds"] = _number(
                    converted["optimization_time_budget_seconds"],
                    "scheduling.optimization_time_budget_seconds",
                )
            if section_name == "exact_solver":
                if "timeout_seconds" in converted:
                    converted["timeout_seconds"] = _number(
                        converted["timeout_seconds"], "exact_solver.timeout_seconds"
                    )
                if "max_nodes" in converted and type(converted["max_nodes"]) is not int:
                    raise ConfigError("exact_solver.max_nodes must be a positive integer")
            parsed[section_name] = section_type(**converted)
        return cls(**parsed)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not load configuration '{config_path}': {exc}") from exc
    if raw_config is None:
        raw_config = {}
    return AppConfig.from_mapping(raw_config)
