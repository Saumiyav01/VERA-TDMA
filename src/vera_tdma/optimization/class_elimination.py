"""Deterministic greedy class elimination for TDMA colorings."""

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import networkx as nx

from ..coordinates import Node
from ..ordering import natural_sort_key
from ..verification import VerificationResult, verify_schedule


@dataclass(frozen=True)
class ClassEliminationResult:
    """Optimized coloring and verification/slot-count information."""

    coloring: dict[str, int]
    slots_before: int
    slots_after: int
    successful_eliminations: int
    slot_counts_after_reductions: tuple[int, ...]
    initial_verification: VerificationResult
    accepted_verifications: tuple[VerificationResult, ...]
    attempted_eliminations: int = 0
    budget_exhausted: bool = False

    @property
    def final_verification(self) -> VerificationResult:
        if self.accepted_verifications:
            return self.accepted_verifications[-1]
        return self.initial_verification


def eliminate_classes(
    conflict_graph: nx.Graph,
    coloring: Mapping[str, int],
    *,
    communication_range_m: float | None = None,
    inclusive_range: bool | None = None,
    deadline: float | None = None,
) -> ClassEliminationResult:
    """Try removing color classes without introducing a new slot.

    ``communication_range_m`` / ``inclusive_range`` select the range policy used
    by the independent verifier; when omitted they are read from
    ``conflict_graph.graph`` (set by ``build_communication_graph``) and finally
    default to 500 m inclusive. ``deadline`` is an absolute ``perf_counter()``
    timestamp; once passed, refinement stops and the best verified coloring so
    far is returned with ``budget_exhausted=True``.

    Candidate classes are attempted in ascending class size (smallest first),
    then ascending numeric color. Nodes within a class are reassigned by
    descending conflict degree and natural identifier; candidate colors are
    tried in ascending order with backtracking.

    Each accepted reduction is independently checked using the Phase 5
    coordinate-based verifier. This is a heuristic improvement step and does
    not guarantee a globally minimum coloring.
    """
    if not isinstance(conflict_graph, nx.Graph):
        raise TypeError("conflict_graph must be a NetworkX graph")
    if conflict_graph.is_directed() or conflict_graph.is_multigraph():
        raise ValueError("conflict_graph must be an undirected simple graph")

    current = dict(coloring)
    _validate_coloring_shape(conflict_graph, current)
    nodes = _nodes_from_graph(conflict_graph)
    range_m, inclusive = range_policy(conflict_graph, communication_range_m, inclusive_range)
    initial_verification = verify_schedule(nodes, current, range_m, inclusive)
    if not initial_verification.valid:
        raise ValueError("initial coloring failed independent schedule verification")
    _validate_graph_edges(conflict_graph, current)

    slots_before = len(set(current.values()))
    accepted_results: list[VerificationResult] = []
    accepted_slot_counts: list[int] = []
    attempts = 0
    budget_exhausted = False

    while current:
        if deadline is not None and perf_counter() > deadline:
            budget_exhausted = True
            break
        slots = sorted(set(current.values()))
        candidate_classes = sorted(
            ((color, [node for node, assigned in current.items() if assigned == color])
             for color in slots),
            key=lambda item: (len(item[1]), item[0]),
        )
        accepted = False

        for eliminated_color, class_nodes in candidate_classes:
            if deadline is not None and perf_counter() > deadline:
                budget_exhausted = True
                break
            attempts += 1
            remaining_slots = [slot for slot in slots if slot != eliminated_color]
            reassigned = _reassign_class(
                conflict_graph,
                current,
                class_nodes,
                eliminated_color,
                remaining_slots,
            )
            if reassigned is None:
                continue

            candidate = dict(current)
            candidate.update(reassigned)
            candidate = _normalize_colors(candidate)
            if len(set(candidate.values())) >= len(set(current.values())):
                continue
            _validate_graph_edges(conflict_graph, candidate)

            verification = verify_schedule(nodes, candidate, range_m, inclusive)
            if not verification.valid:
                continue

            current = candidate
            accepted_results.append(verification)
            accepted_slot_counts.append(len(set(current.values())))
            accepted = True
            break

        if not accepted:
            break

    final_slots = len(set(current.values()))
    if final_slots > slots_before:
        raise RuntimeError("class elimination increased the number of slots")
    return ClassEliminationResult(
        coloring=current,
        slots_before=slots_before,
        slots_after=final_slots,
        successful_eliminations=len(accepted_results),
        slot_counts_after_reductions=tuple(accepted_slot_counts),
        initial_verification=initial_verification,
        accepted_verifications=tuple(accepted_results),
        attempted_eliminations=attempts,
        budget_exhausted=budget_exhausted,
    )


def range_policy(
    conflict_graph: nx.Graph,
    communication_range_m: float | None,
    inclusive_range: bool | None,
) -> tuple[float, bool]:
    """Resolve the verifier range policy: explicit argument, graph metadata, default."""
    range_m = (
        communication_range_m
        if communication_range_m is not None
        else conflict_graph.graph.get("communication_range_m", 500.0)
    )
    inclusive = (
        inclusive_range
        if inclusive_range is not None
        else conflict_graph.graph.get("inclusive_range", True)
    )
    return range_m, inclusive


def _validate_coloring_shape(
    conflict_graph: nx.Graph,
    coloring: Mapping[str, Any],
) -> None:
    graph_nodes = set(conflict_graph.nodes)
    if set(coloring) != graph_nodes:
        raise ValueError("coloring must contain exactly one entry for every graph node")
    if any(not isinstance(node, str) for node in graph_nodes):
        raise ValueError("conflict graph node identifiers must be strings")
    if any(type(color) is not int or color < 0 for color in coloring.values()):
        raise ValueError("colors must be non-negative integers")


def _nodes_from_graph(conflict_graph: nx.Graph) -> list[Node]:
    result: list[Node] = []
    for node_id in sorted(conflict_graph.nodes, key=natural_sort_key):
        attributes = conflict_graph.nodes[node_id]
        if "x_m" not in attributes or "y_m" not in attributes:
            raise ValueError(
                f"node {node_id!r} must have x_m and y_m coordinate attributes"
            )
        result.append(Node(node_id, attributes["x_m"], attributes["y_m"]))
    return result


def _validate_graph_edges(conflict_graph: nx.Graph, coloring: Mapping[str, int]) -> None:
    for first, second in conflict_graph.edges:
        if coloring[first] == coloring[second]:
            raise ValueError(f"conflicting nodes {first!r} and {second!r} share a color")


def _reassign_class(
    conflict_graph: nx.Graph,
    coloring: Mapping[str, int],
    class_nodes: list[str],
    eliminated_color: int,
    available_slots: list[int],
) -> dict[str, int] | None:
    if not available_slots:
        return None

    ordered_class = sorted(
        class_nodes,
        key=lambda node: (-conflict_graph.degree[node], natural_sort_key(node)),
    )
    working = dict(coloring)
    for node in class_nodes:
        del working[node]

    def assign(index: int) -> bool:
        if index == len(ordered_class):
            return True
        node = ordered_class[index]
        forbidden = {
            working[neighbor]
            for neighbor in conflict_graph.neighbors(node)
            if neighbor in working
        }
        for slot in available_slots:
            if slot in forbidden:
                continue
            working[node] = slot
            if assign(index + 1):
                return True
            del working[node]
        return False

    if not assign(0):
        return None
    return {node: working[node] for node in class_nodes}


def _normalize_colors(coloring: Mapping[str, int]) -> dict[str, int]:
    ordered_slots = sorted(set(coloring.values()))
    remapping = {old: new for new, old in enumerate(ordered_slots)}
    return {node: remapping[color] for node, color in coloring.items()}
