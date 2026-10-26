"""Phase 1 foundations for VERA-TDMA."""

from .communication_graph import COMMUNICATION_RANGE_M, build_communication_graph
from .config import AppConfig, ConfigError, load_config
from .conflict_graph import build_conflict_graph
from .coordinates import (
    CoLocatedNodesWarning,
    CoordinateLoadError,
    Node,
    load_coordinates,
)
from .ordering import natural_sort_key, natural_sorted
from .verification import ScheduleViolation, VerificationResult, verify_schedule

__all__ = [
    "AppConfig",
    "COMMUNICATION_RANGE_M",
    "CoLocatedNodesWarning",
    "ConfigError",
    "CoordinateLoadError",
    "Node",
    "load_config",
    "load_coordinates",
    "build_communication_graph",
    "build_conflict_graph",
    "natural_sort_key",
    "natural_sorted",
    "ScheduleViolation",
    "VerificationResult",
    "verify_schedule",
]
