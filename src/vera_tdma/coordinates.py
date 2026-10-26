"""Coordinate input loading and validation for JSON and CSV files."""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Any

from .ordering import natural_sort_key


class CoordinateLoadError(ValueError):
    """Raised when a coordinate file is malformed or contains invalid nodes."""


class CoLocatedNodesWarning(UserWarning):
    """Warns that distinct nodes have identical coordinates."""


class ImplausibleCoordinateWarning(UserWarning):
    """Warns that a coordinate is implausibly large for a planar layout in meters."""


MAX_PLAUSIBLE_COORDINATE_M = 1e7


@dataclass(frozen=True)
class Node:
    node_id: str
    x_m: float
    y_m: float

    @property
    def coordinates_m(self) -> tuple[float, float]:
        return self.x_m, self.y_m


def load_coordinates(path: str | Path, file_format: str | None = None) -> list[Node]:
    """Load nodes from JSON or CSV and return them in natural identifier order."""
    coordinate_path = Path(path)
    selected_format = (file_format or coordinate_path.suffix.lstrip(".")).lower()
    if selected_format not in {"json", "csv"}:
        raise CoordinateLoadError("coordinate format must be 'json' or 'csv'")

    try:
        if selected_format == "json":
            records = _read_json(coordinate_path)
        else:
            records = _read_csv(coordinate_path)
    except CoordinateLoadError:
        raise
    except OSError as exc:
        raise CoordinateLoadError(f"could not read coordinate file '{coordinate_path}': {exc}") from exc

    nodes = [
        _make_node(node_id, coordinates, allow_numeric_strings=selected_format == "csv")
        for node_id, coordinates in records
    ]
    identifiers: set[str] = set()
    for node in nodes:
        if node.node_id in identifiers:
            raise CoordinateLoadError(f"duplicate node identifier: {node.node_id!r}")
        identifiers.add(node.node_id)
    _warn_for_colocated_nodes(nodes)
    return sorted(nodes, key=lambda node: natural_sort_key(node.node_id))


def _read_json(path: Path) -> list[tuple[str, Any]]:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CoordinateLoadError(f"duplicate node identifier in JSON: {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as coordinate_file:
            data = json.load(coordinate_file, object_pairs_hook=object_from_pairs)
    except OSError as exc:
        raise CoordinateLoadError(f"could not read coordinate file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CoordinateLoadError(f"could not parse JSON coordinate file '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise CoordinateLoadError("JSON coordinate input must be an object mapping node IDs to [x, y]")
    return list(data.items())


def _read_csv(path: Path) -> list[tuple[str, Any]]:
    expected_header = ["node_id", "x_coordinate", "y_coordinate"]
    records: list[tuple[str, tuple[str, str]]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as coordinate_file:
            reader = csv.DictReader(coordinate_file)
            if reader.fieldnames != expected_header:
                raise CoordinateLoadError(
                    "CSV header must be exactly: node_id,x_coordinate,y_coordinate"
                )
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise CoordinateLoadError(f"invalid CSV row at line {line_number}")
                records.append((row["node_id"], (row["x_coordinate"], row["y_coordinate"])))
    except OSError as exc:
        raise CoordinateLoadError(f"could not read CSV coordinate file '{path}': {exc}") from exc
    return records


def _make_node(node_id: Any, coordinates: Any, *, allow_numeric_strings: bool) -> Node:
    if not isinstance(node_id, str) or not node_id.strip():
        raise CoordinateLoadError("node identifiers must be non-empty strings")
    if node_id != node_id.strip():
        raise CoordinateLoadError(
            f"node identifier {node_id!r} must not have leading or trailing whitespace"
        )
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, (list, tuple)):
        raise CoordinateLoadError(f"coordinates for node {node_id!r} must be a two-value [x, y] pair")
    if len(coordinates) != 2:
        raise CoordinateLoadError(f"coordinates for node {node_id!r} must contain exactly two values")
    x_m = _coordinate_number(coordinates[0], node_id, "x", allow_numeric_strings)
    y_m = _coordinate_number(coordinates[1], node_id, "y", allow_numeric_strings)
    return Node(node_id=node_id, x_m=x_m, y_m=y_m)


def make_node(node_id: Any, coordinates: Any, *, allow_numeric_strings: bool = False) -> Node:
    """Validate one ``[x, y]`` pair and return a :class:`Node` (public wrapper)."""
    return _make_node(node_id, coordinates, allow_numeric_strings=allow_numeric_strings)


def _coordinate_number(
    value: Any, node_id: str, axis: str, allow_numeric_strings: bool
) -> float:
    if isinstance(value, bool):
        raise CoordinateLoadError(f"{axis} coordinate for node {node_id!r} must be a finite number")
    if not isinstance(value, (int, float)) and not (
        allow_numeric_strings and isinstance(value, str)
    ):
        raise CoordinateLoadError(f"{axis} coordinate for node {node_id!r} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CoordinateLoadError(
            f"{axis} coordinate for node {node_id!r} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise CoordinateLoadError(f"{axis} coordinate for node {node_id!r} must be a finite number")
    return number


def warn_for_implausible_coordinates(nodes: list[Node]) -> None:
    """Warn about coordinates beyond ``MAX_PLAUSIBLE_COORDINATE_M`` meters."""
    for node in nodes:
        if max(abs(node.x_m), abs(node.y_m)) > MAX_PLAUSIBLE_COORDINATE_M:
            warnings.warn(
                f"node {node.node_id!r} has an implausibly large coordinate "
                f"{node.coordinates_m} meters; it will be isolated from other nodes",
                ImplausibleCoordinateWarning,
                stacklevel=2,
            )


def _warn_for_colocated_nodes(nodes: list[Node]) -> None:
    coordinate_nodes: dict[tuple[float, float], list[str]] = defaultdict(list)
    for node in nodes:
        coordinate_nodes[node.coordinates_m].append(node.node_id)
    for coordinates, node_ids in coordinate_nodes.items():
        if len(node_ids) > 1:
            identifiers = ", ".join(sorted(node_ids))
            warnings.warn(
                f"nodes {identifiers} share coordinates {coordinates} meters",
                CoLocatedNodesWarning,
                stacklevel=2,
            )
