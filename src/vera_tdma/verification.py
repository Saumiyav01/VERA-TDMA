"""Independent verification of TDMA colorings from node coordinates."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Any

import networkx as nx

from .coordinates import Node
from .ordering import natural_sort_key


@dataclass(frozen=True)
class ScheduleViolation:
    """One invalid schedule property discovered by the independent verifier."""

    node_1: str | None
    node_2: str | None
    color_1: Any
    color_2: Any
    conflict_type: str
    distance_m: float | None = None
    path: tuple[str, ...] | None = None
    message: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Summary and detailed violations from schedule verification."""

    number_of_nodes: int
    direct_conflicts_checked: int
    two_hop_conflicts_checked: int
    violations: tuple[ScheduleViolation, ...]
    coordinates_unchanged: bool

    @property
    def valid(self) -> bool:
        return not self.violations

    @property
    def number_of_violations(self) -> int:
        return len(self.violations)


def verify_schedule(
    nodes: Iterable[Node],
    colors: Mapping[str, Any],
    communication_range: float = 500.0,
    inclusive_range: bool = True,
) -> VerificationResult:
    """Independently verify a coloring using only the input node coordinates.

    Euclidean distance is calculated for every distinct node pair to
    reconstruct direct communication adjacency, using exact rational
    arithmetic near the range boundary (same inclusive/strict policy as the
    scheduler, see ``inclusive_range``). Hop distances up to two are then
    derived by expanding paths ``u - w - v`` from the adjacency, and the
    resulting conflict set is cross-checked against ``networkx.power(G, 2)``.
    No graph-construction or coloring module of this package is consulted.
    """
    violations: list[ScheduleViolation] = []
    node_list = list(nodes)
    coordinate_snapshot = [
        (node, getattr(node, "node_id", None), getattr(node, "x_m", None),
         getattr(node, "y_m", None))
        for node in node_list
    ]

    range_value = _finite_float(communication_range)
    if range_value is None or range_value < 0:
        violations.append(
            ScheduleViolation(
                None,
                None,
                None,
                None,
                "invalid_range",
                message="communication_range must be a finite non-negative number",
            )
        )
        return VerificationResult(len(node_list), 0, 0, tuple(violations), True)

    if not isinstance(colors, Mapping):
        violations.append(
            ScheduleViolation(
                None, None, None, None, "invalid_color_mapping",
                message="colors must be a mapping from node IDs to integer colors",
            )
        )
        return VerificationResult(len(node_list), 0, 0, tuple(violations), True)

    valid_nodes: list[Node] = []
    node_by_id: dict[str, Node] = {}
    coordinates_by_id: dict[str, tuple[float, float]] = {}
    for node in node_list:
        node_id = getattr(node, "node_id", None)
        x_m = getattr(node, "x_m", None)
        y_m = getattr(node, "y_m", None)
        if not isinstance(node_id, str) or not node_id.strip():
            violations.append(
                ScheduleViolation(
                    node_id if isinstance(node_id, str) else None,
                    None,
                    colors.get(node_id) if isinstance(node_id, str) else None,
                    None,
                    "invalid_node",
                    message="node identifier must be a non-empty string",
                )
            )
            continue
        if node_id in node_by_id:
            violations.append(
                ScheduleViolation(
                    node_id,
                    None,
                    colors.get(node_id),
                    None,
                    "duplicate_node",
                    message=f"duplicate input node identifier {node_id!r}",
                )
            )
            continue
        x_value = _finite_float(x_m)
        y_value = _finite_float(y_m)
        if x_value is None or y_value is None:
            violations.append(
                ScheduleViolation(
                    node_id,
                    None,
                    colors.get(node_id),
                    None,
                    "invalid_coordinates",
                    message=f"coordinates for node {node_id!r} must be finite numbers",
                )
            )
            continue
        node_by_id[node_id] = node
        coordinates_by_id[node_id] = (x_value, y_value)
        valid_nodes.append(node)

    ordered_nodes = sorted(valid_nodes, key=lambda node: natural_sort_key(node.node_id))
    node_ids = [node.node_id for node in ordered_nodes]
    expected_ids = set(node_ids)
    supplied_ids = set(colors)
    valid_color_ids: set[str] = set()

    for node_id in sorted(expected_ids - supplied_ids, key=natural_sort_key):
        violations.append(
            ScheduleViolation(
                node_id,
                None,
                None,
                None,
                "missing_color",
                message=f"node {node_id!r} has no assigned color",
            )
        )
    for node_id in sorted(supplied_ids - expected_ids, key=_identifier_sort_key):
        violations.append(
            ScheduleViolation(
                node_id if isinstance(node_id, str) else None,
                None,
                colors[node_id],
                None,
                "unexpected_node",
                message=f"color mapping contains unknown node {node_id!r}",
            )
        )
    for node_id in node_ids:
        if node_id not in colors:
            continue
        color = colors[node_id]
        if type(color) is not int or color < 0:
            violations.append(
                ScheduleViolation(
                    node_id,
                    None,
                    color,
                    None,
                    "invalid_color",
                    message=f"color for node {node_id!r} must be a non-negative integer",
                )
            )
        else:
            valid_color_ids.add(node_id)

    geometry = _derive_geometry(
        tuple((node.node_id, *coordinates_by_id[node.node_id]) for node in ordered_nodes),
        range_value,
        inclusive_range,
    )
    direct_neighbors = geometry.direct_neighbors
    witness = geometry.witness
    distance_by_pair = geometry.distance_by_pair
    direct_conflicts_checked = geometry.direct_pairs
    two_hop_conflicts_checked = len(witness)
    conflict_pairs = geometry.conflict_pairs
    if geometry.crosscheck_differences:
        violations.append(
            ScheduleViolation(
                None,
                None,
                None,
                None,
                "crosscheck_mismatch",
                message=(
                    "verifier conflict set disagrees with networkx.power(G, 2): "
                    f"{geometry.crosscheck_differences} differing pair(s)"
                ),
            )
        )

    for first_id, second_id in conflict_pairs:
        if first_id not in valid_color_ids or second_id not in valid_color_ids:
            continue
        color_1 = colors[first_id]
        color_2 = colors[second_id]
        if color_1 != color_2:
            continue
        direct = second_id in direct_neighbors[first_id]
        shared_receiver = witness.get((first_id, second_id))
        path = (
            (first_id, shared_receiver, second_id)
            if shared_receiver is not None
            else (first_id, second_id)
        )
        conflict_type = (
            "direct_and_two_hop"
            if direct and shared_receiver is not None
            else "direct"
            if direct
            else "two_hop"
        )
        violations.append(
            ScheduleViolation(
                first_id,
                second_id,
                color_1,
                color_2,
                conflict_type,
                distance_m=distance_by_pair[(first_id, second_id)] if direct else None,
                path=path,
                message=f"conflicting nodes {first_id!r} and {second_id!r} share color {color_1}",
            )
        )

    coordinates_unchanged = all(
        getattr(node, "node_id", None) == node_id
        and getattr(node, "x_m", None) == x_m
        and getattr(node, "y_m", None) == y_m
        for node, node_id, x_m, y_m in coordinate_snapshot
    )
    if not coordinates_unchanged:
        violations.append(
            ScheduleViolation(
                None,
                None,
                None,
                None,
                "coordinates_modified",
                message="input node coordinates changed during verification",
            )
        )

    return VerificationResult(
        len(node_list),
        direct_conflicts_checked,
        two_hop_conflicts_checked,
        tuple(violations),
        coordinates_unchanged,
    )


@dataclass(frozen=True)
class _Geometry:
    """Everything derived from coordinates alone (independent of any coloring)."""

    direct_neighbors: dict[str, frozenset[str]]
    witness: dict[tuple[str, str], str]
    distance_by_pair: dict[tuple[str, str], float]
    conflict_pairs: tuple[tuple[str, str], ...]  # natural order, deterministic
    direct_pairs: int
    crosscheck_differences: int


@lru_cache(maxsize=8)
def _derive_geometry(
    coordinates: tuple[tuple[str, float, float], ...],
    range_value: float,
    inclusive: bool,
) -> _Geometry:
    """Reconstruct adjacency, hop distance <= 2 and the conflict set from coordinates.

    A pure function of its arguments, so repeated verification of different
    colorings over the same coordinates (class elimination, Kempe refinement)
    reuses the result instead of repeating the O(n^2) geometry and the
    ``networkx.power`` cross-check.
    """
    node_ids = [item[0] for item in coordinates]
    xy = {item[0]: (item[1], item[2]) for item in coordinates}
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    distance_by_pair: dict[tuple[str, str], float] = {}
    direct_pairs = 0
    for index, first_id in enumerate(node_ids):
        first_x, first_y = xy[first_id]
        for second_id in node_ids[index + 1 :]:
            second_x, second_y = xy[second_id]
            distance_by_pair[(first_id, second_id)] = math.hypot(
                first_x - second_x, first_y - second_y
            )
            if _pair_within_range(
                first_x, first_y, second_x, second_y, range_value, inclusive
            ):
                neighbors[first_id].add(second_id)
                neighbors[second_id].add(first_id)
                direct_pairs += 1

    # Hop distance <= 2 by explicit path expansion u - w - v. ``witness`` maps
    # a pair (in natural order) to its smallest shared receiver.
    witness: dict[tuple[str, str], str] = {}
    for first_id in node_ids:
        for middle_id in sorted(neighbors[first_id], key=natural_sort_key):
            for second_id in neighbors[middle_id]:
                if second_id == first_id or position[first_id] > position[second_id]:
                    continue
                witness.setdefault((first_id, second_id), middle_id)

    pairs = {
        (first_id, second_id)
        for first_id in node_ids
        for second_id in neighbors[first_id]
        if position[first_id] < position[second_id]
    } | set(witness)

    # Cross-check against the library definition of the square graph.
    reference_graph = nx.Graph()
    reference_graph.add_nodes_from(node_ids)
    for first_id in node_ids:
        for second_id in neighbors[first_id]:
            reference_graph.add_edge(first_id, second_id)
    reference_pairs = {
        tuple(sorted(edge, key=position.__getitem__))
        for edge in nx.power(reference_graph, 2).edges
    }
    return _Geometry(
        direct_neighbors={node_id: frozenset(items) for node_id, items in neighbors.items()},
        witness=witness,
        distance_by_pair=distance_by_pair,
        conflict_pairs=tuple(
            sorted(pairs, key=lambda pair: (position[pair[0]], position[pair[1]]))
        ),
        direct_pairs=direct_pairs,
        crosscheck_differences=len(pairs ^ reference_pairs),
    )


def _pair_within_range(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
    range_value: float,
    inclusive: bool,
) -> bool:
    """Exact range decision (float fast path, rational arithmetic near the boundary)."""
    dx = first_x - second_x
    dy = first_y - second_y
    distance_squared = dx * dx + dy * dy
    range_squared = range_value * range_value
    if math.isfinite(distance_squared) and math.isfinite(range_squared):
        band = 1e-9 * max(range_squared, distance_squared)
        if distance_squared < range_squared - band:
            return True
        if distance_squared > range_squared + band:
            return False
    exact_dx = Fraction(first_x) - Fraction(second_x)
    exact_dy = Fraction(first_y) - Fraction(second_y)
    exact_squared = exact_dx * exact_dx + exact_dy * exact_dy
    exact_range_squared = Fraction(range_value) ** 2
    return exact_squared <= exact_range_squared if inclusive else exact_squared < exact_range_squared


def verify_matrix_consistency(
    rows: Mapping[Any, Mapping[str, Any]],
    node_order: Iterable[str],
    node_to_slot: Mapping[str, Any],
) -> tuple[ScheduleViolation, ...]:
    """Check the Slot x Node Boolean matrix against the node-to-slot mapping.

    Requires ``M[s(v)][v] == 1`` and ``M[k][v] == 0`` for every ``k != s(v)``,
    only 0/1 entries, one row per used slot, and no slot index outside the rows.
    """
    problems: list[ScheduleViolation] = []
    ordered = list(node_order)
    row_keys = {str(key): key for key in rows}
    used_slots = {slot for slot in node_to_slot.values()}
    for node_id in ordered:
        slot = node_to_slot.get(node_id)
        if type(slot) is not int or slot < 0:
            problems.append(
                ScheduleViolation(node_id, None, slot, None, "invalid_slot_index",
                                  message=f"node {node_id!r} has invalid slot index {slot!r}")
            )
            continue
        if str(slot) not in row_keys:
            problems.append(
                ScheduleViolation(node_id, None, slot, None, "matrix_mismatch",
                                  message=f"matrix has no row for slot {slot} of node {node_id!r}")
            )
    for key, row in rows.items():
        if str(key) not in {str(slot) for slot in used_slots}:
            problems.append(
                ScheduleViolation(None, None, None, None, "matrix_mismatch",
                                  message=f"matrix row {key!r} matches no assigned slot")
            )
        for node_id in ordered:
            value = row.get(node_id)
            if type(value) is not int or value not in (0, 1):
                problems.append(
                    ScheduleViolation(node_id, None, value, None, "matrix_mismatch",
                                      message=f"matrix entry [{key}][{node_id}] must be 0 or 1")
                )
                continue
            expected = int(str(node_to_slot.get(node_id)) == str(key))
            if value != expected:
                problems.append(
                    ScheduleViolation(node_id, None, key, node_to_slot.get(node_id),
                                      "matrix_mismatch",
                                      message=(
                                          f"matrix entry [{key}][{node_id}] is {value} but the "
                                          f"node-to-slot mapping says slot {node_to_slot.get(node_id)}"
                                      ))
                )
    return tuple(problems)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) else None


def _identifier_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, str):
        return 0, natural_sort_key(value)
    return 1, (type(value).__name__, repr(value))
