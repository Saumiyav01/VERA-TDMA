"""Construction of the geographic communication graph."""

import math
from collections.abc import Iterable
from fractions import Fraction

import networkx as nx

from .coordinates import Node
from .ordering import natural_sort_key

COMMUNICATION_RANGE_M = 500.0

# Relative band around the range in which float arithmetic cannot be trusted
# to decide the boundary; pairs inside the band are decided with exact
# rational arithmetic instead.
_BOUNDARY_RELATIVE_BAND = 1e-9


def within_range(
    first: Node,
    second: Node,
    communication_range_m: float,
    inclusive: bool = True,
) -> bool:
    """Decide ``distance <= range`` (or ``<`` when not inclusive) exactly.

    A float fast path decides clear cases; near the boundary (or on overflow)
    the decision uses exact ``Fraction`` arithmetic on the stored float values,
    so the result never depends on rounding.
    """
    dx = first.x_m - second.x_m
    dy = first.y_m - second.y_m
    distance_squared = dx * dx + dy * dy
    range_squared = communication_range_m * communication_range_m
    if math.isfinite(distance_squared) and math.isfinite(range_squared):
        band = _BOUNDARY_RELATIVE_BAND * max(range_squared, distance_squared)
        if distance_squared < range_squared - band:
            return True
        if distance_squared > range_squared + band:
            return False
    exact_dx = Fraction(first.x_m) - Fraction(second.x_m)
    exact_dy = Fraction(first.y_m) - Fraction(second.y_m)
    exact_squared = exact_dx * exact_dx + exact_dy * exact_dy
    exact_range_squared = Fraction(communication_range_m) ** 2
    if inclusive:
        return exact_squared <= exact_range_squared
    return exact_squared < exact_range_squared


def build_communication_graph(
    nodes: Iterable[Node],
    communication_range_m: float = COMMUNICATION_RANGE_M,
    inclusive: bool = True,
) -> nx.Graph:
    """Build an undirected graph using inclusive Euclidean range in meters.

    Pass ``inclusive=False`` to treat a distance of exactly the range as out of
    range. The range policy is recorded in ``graph.graph`` so downstream stages
    verify with the same policy.
    """
    if isinstance(communication_range_m, bool) or not isinstance(
        communication_range_m, (int, float)
    ):
        raise ValueError("communication_range_m must be a finite non-negative number")
    if not math.isfinite(communication_range_m) or communication_range_m < 0:
        raise ValueError("communication_range_m must be a finite non-negative number")

    ordered_nodes = sorted(nodes, key=lambda node: natural_sort_key(node.node_id))
    identifiers: set[str] = set()
    for node in ordered_nodes:
        if node.node_id in identifiers:
            raise ValueError(f"duplicate node identifier: {node.node_id!r}")
        identifiers.add(node.node_id)
        if not math.isfinite(node.x_m) or not math.isfinite(node.y_m):
            raise ValueError(f"coordinates for node {node.node_id!r} must be finite")

    graph = nx.Graph()
    graph.graph["communication_range_m"] = float(communication_range_m)
    graph.graph["inclusive_range"] = bool(inclusive)
    for node in ordered_nodes:
        graph.add_node(
            node.node_id,
            x_m=node.x_m,
            y_m=node.y_m,
            coordinates_m=node.coordinates_m,
        )

    for index, first in enumerate(ordered_nodes):
        for second in ordered_nodes[index + 1 :]:
            if within_range(first, second, communication_range_m, inclusive):
                dx = first.x_m - second.x_m
                dy = first.y_m - second.y_m
                graph.add_edge(
                    first.node_id,
                    second.node_id,
                    distance_m=math.sqrt(dx * dx + dy * dy),
                )
    return graph
