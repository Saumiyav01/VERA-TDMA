"""Deterministic Kempe-chain refinement for TDMA colorings."""

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import networkx as nx

from ..coordinates import Node
from ..ordering import natural_sort_key
from ..verification import VerificationResult, verify_schedule
from .class_elimination import eliminate_classes, range_policy


@dataclass(frozen=True)
class KempeRefinementResult:
    """Refined coloring and details about attempted and accepted swaps."""

    coloring: dict[str, int]
    slots_before: int
    slots_after: int
    attempted_swaps: int
    accepted_improvements: int
    initial_verification: VerificationResult
    final_verification: VerificationResult
    accepted_verifications: tuple[VerificationResult, ...]
    budget_exhausted: bool = False


def kempe_refine(
    conflict_graph: nx.Graph,
    coloring: Mapping[str, int],
    *,
    communication_range_m: float | None = None,
    inclusive_range: bool | None = None,
    deadline: float | None = None,
) -> KempeRefinementResult:
    """Use deterministic two-color component swaps to enable class reduction.

    Color pairs are considered in ascending numeric order. For each pair,
    induced connected components are considered by their smallest natural
    node identifier. Swaps are attempted one component at a time, in that
    order. A candidate is retained only if Phase 6 class elimination and the
    independent Phase 5 verifier together produce a strict slot reduction.

    The range policy and ``deadline`` (absolute ``perf_counter()`` timestamp)
    behave as in :func:`eliminate_classes`; on expiry the best verified
    coloring so far is returned with ``budget_exhausted=True``.
    """
    _validate_graph(conflict_graph)
    original = dict(coloring)
    _validate_coloring(conflict_graph, original)
    nodes = _nodes_from_conflict_graph(conflict_graph)
    range_m, inclusive = range_policy(conflict_graph, communication_range_m, inclusive_range)
    initial_verification = verify_schedule(nodes, original, range_m, inclusive)
    if not initial_verification.valid:
        raise ValueError("initial coloring failed independent schedule verification")

    # Only explicitly supplied options are forwarded; otherwise the range policy
    # is resolved from the graph metadata inside ``eliminate_classes``.
    elimination_options = {
        name: value
        for name, value in (
            ("communication_range_m", communication_range_m),
            ("inclusive_range", inclusive_range),
            ("deadline", deadline),
        )
        if value is not None
    }
    original_slot_count = len(set(original.values()))
    current = dict(original)
    attempted_swaps = 0
    accepted_improvements = 0
    current_verification = initial_verification
    accepted_verifications: list[VerificationResult] = []
    budget_exhausted = False

    while not budget_exhausted:
        improved = False
        current_slot_count = len(set(current.values()))
        slots = sorted(set(current.values()))
        color_pairs = [
            (first, second)
            for index, first in enumerate(slots)
            for second in slots[index + 1 :]
        ]
        for first_color, second_color in color_pairs:
            components = _kempe_components(
                conflict_graph, current, first_color, second_color
            )
            for component in components:
                if deadline is not None and perf_counter() > deadline:
                    budget_exhausted = True
                    break
                attempted_swaps += 1
                swapped = dict(current)
                for node in component:
                    swapped[node] = (
                        second_color if current[node] == first_color else first_color
                    )

                if len(set(swapped.values())) != len(slots):
                    continue
                if not _is_valid_coloring(conflict_graph, swapped):
                    continue

                elimination = eliminate_classes(
                    conflict_graph, swapped, **elimination_options
                )
                candidate = elimination.coloring
                candidate_slot_count = len(set(candidate.values()))
                if candidate_slot_count >= current_slot_count:
                    continue

                verification = verify_schedule(nodes, candidate, range_m, inclusive)
                if not verification.valid:
                    continue

                current = dict(candidate)
                current_slot_count = candidate_slot_count
                current_verification = verification
                accepted_verifications.append(verification)
                accepted_improvements += 1
                improved = True
                break
            if improved or budget_exhausted:
                break
        if not improved:
            break

    if len(set(current.values())) > original_slot_count:
        raise RuntimeError("Kempe refinement increased the number of slots")
    if not current_verification.valid:
        raise RuntimeError("Kempe refinement produced an invalid schedule")

    return KempeRefinementResult(
        coloring=current,
        slots_before=original_slot_count,
        slots_after=len(set(current.values())),
        attempted_swaps=attempted_swaps,
        accepted_improvements=accepted_improvements,
        initial_verification=initial_verification,
        final_verification=current_verification,
        accepted_verifications=tuple(accepted_verifications),
        budget_exhausted=budget_exhausted,
    )


def _validate_graph(conflict_graph: nx.Graph) -> None:
    if not isinstance(conflict_graph, nx.Graph):
        raise TypeError("conflict_graph must be a NetworkX graph")
    if conflict_graph.is_directed() or conflict_graph.is_multigraph():
        raise ValueError("conflict_graph must be an undirected simple graph")
    if any(not isinstance(node, str) for node in conflict_graph.nodes):
        raise ValueError("conflict graph node identifiers must be strings")


def _validate_coloring(conflict_graph: nx.Graph, coloring: Mapping[str, int]) -> None:
    if set(coloring) != set(conflict_graph.nodes):
        raise ValueError("coloring must contain exactly one entry for every graph node")
    if any(type(color) is not int or color < 0 for color in coloring.values()):
        raise ValueError("colors must be non-negative integers")
    if not _is_valid_coloring(conflict_graph, coloring):
        raise ValueError("initial coloring is invalid for the conflict graph")


def _is_valid_coloring(conflict_graph: nx.Graph, coloring: Mapping[str, int]) -> bool:
    return all(
        coloring[first] != coloring[second]
        for first, second in conflict_graph.edges
    )


def _nodes_from_conflict_graph(conflict_graph: nx.Graph) -> list[Node]:
    nodes: list[Node] = []
    for node_id in sorted(conflict_graph.nodes, key=natural_sort_key):
        attributes = conflict_graph.nodes[node_id]
        if "x_m" not in attributes or "y_m" not in attributes:
            raise ValueError(
                f"node {node_id!r} must have x_m and y_m coordinate attributes"
            )
        nodes.append(Node(node_id, attributes["x_m"], attributes["y_m"]))
    return nodes


def _kempe_components(
    conflict_graph: nx.Graph,
    coloring: Mapping[str, int],
    first_color: int,
    second_color: int,
) -> list[tuple[str, ...]]:
    selected_nodes = {
        node
        for node, color in coloring.items()
        if color == first_color or color == second_color
    }
    induced = conflict_graph.subgraph(selected_nodes)
    components = [
        tuple(sorted(component, key=natural_sort_key))
        for component in nx.connected_components(induced)
    ]
    return sorted(components, key=lambda component: natural_sort_key(component[0]))
