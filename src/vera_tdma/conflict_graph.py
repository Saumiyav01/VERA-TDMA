"""Construction of distance-2 conflict graphs."""

from itertools import combinations

import networkx as nx

from .ordering import natural_sort_key


def build_conflict_graph(communication_graph: nx.Graph) -> nx.Graph:
    """Build the square of an undirected communication graph.

    Conflict edges record whether the relationship is direct, two-hop, or both.
    For every edge with a two-hop relationship, ``witnesses`` contains its
    common neighbors in deterministic natural order.
    """
    if communication_graph.is_directed() or communication_graph.is_multigraph():
        raise ValueError("communication_graph must be an undirected simple graph")

    ordered_nodes = sorted(communication_graph.nodes, key=natural_sort_key)
    conflict_graph = nx.Graph()
    conflict_graph.graph.update(communication_graph.graph)
    conflict_graph.add_nodes_from(
        (node, dict(communication_graph.nodes[node])) for node in ordered_nodes
    )

    for first, second in combinations(ordered_nodes, 2):
        direct = communication_graph.has_edge(first, second)
        witnesses = sorted(
            set(communication_graph.neighbors(first))
            & set(communication_graph.neighbors(second)),
            key=natural_sort_key,
        )
        if direct or witnesses:
            if direct and witnesses:
                conflict_type = "direct_and_two_hop"
            elif direct:
                conflict_type = "direct"
            else:
                conflict_type = "two_hop"
            edge_attributes = {"conflict_type": conflict_type}
            if witnesses:
                edge_attributes["witnesses"] = tuple(witnesses)
            conflict_graph.add_edge(first, second, **edge_attributes)

    return conflict_graph
