"""Deterministic DSATUR initial coloring for conflict graphs."""

import networkx as nx

from ..ordering import natural_sort_key


def dsatur_color(conflict_graph: nx.Graph) -> tuple[dict[str, int], int]:
    """Color a conflict graph with deterministic DSATUR.

    Nodes are selected by descending saturation degree, descending graph
    degree, then ascending natural node identifier. The smallest available
    color is assigned. This greedy coloring is feasible but is not guaranteed
    to be minimum.
    """
    _require_undirected_simple_graph(conflict_graph)
    nodes = sorted(conflict_graph.nodes, key=natural_sort_key)
    colors: dict[str, int] = {}

    while len(colors) < len(nodes):
        uncolored = (node for node in nodes if node not in colors)
        selected = min(
            uncolored,
            key=lambda node: (
                -len({colors[neighbor] for neighbor in conflict_graph.neighbors(node)
                      if neighbor in colors}),
                -conflict_graph.degree[node],
                natural_sort_key(node),
            ),
        )
        unavailable = {
            colors[neighbor]
            for neighbor in conflict_graph.neighbors(selected)
            if neighbor in colors
        }
        color = 0
        while color in unavailable:
            color += 1
        colors[selected] = color

    number_of_colors = len(set(colors.values()))
    validate_coloring(conflict_graph, colors)
    return colors, number_of_colors


def validate_coloring(conflict_graph: nx.Graph, colors: dict[str, int]) -> None:
    """Raise ValueError if a coloring is incomplete, invalid, or non-contiguous."""
    _require_undirected_simple_graph(conflict_graph)
    if set(colors) != set(conflict_graph.nodes):
        raise ValueError("color mapping must contain every graph node exactly once")
    if any(type(color) is not int or color < 0 for color in colors.values()):
        raise ValueError("colors must be non-negative integers")
    used_colors = set(colors.values())
    if used_colors != set(range(len(used_colors))):
        raise ValueError("colors must be contiguous starting from zero")
    for first, second in conflict_graph.edges:
        if colors[first] == colors[second]:
            raise ValueError(f"conflicting nodes {first!r} and {second!r} share a color")


def _require_undirected_simple_graph(conflict_graph: nx.Graph) -> None:
    if not isinstance(conflict_graph, nx.Graph):
        raise TypeError("conflict_graph must be a NetworkX graph")
    if conflict_graph.is_directed() or conflict_graph.is_multigraph():
        raise ValueError("conflict_graph must be an undirected simple graph")
