"""Rigorous lower bounds on the number of TDMA slots.

Both bounds are valid for the Distance-2 conflict graph ``C = G^2``:

* closed-neighborhood bound: for every node ``v`` the closed neighborhood
  ``N[v]`` of the *communication* graph ``G`` is a clique in ``C`` (any two of
  its members are within two hops through ``v``), so ``K >= max(deg(v) + 1)``;
* clique bound: any clique found in ``C`` of size ``w`` forces ``K >= w``.
"""

import networkx as nx

from .ordering import natural_sort_key

_MAX_CLIQUE_SEEDS = 32


def closed_neighborhood_lower_bound(communication_graph: nx.Graph) -> int:
    """Return ``max(deg_G(v) + 1)``; zero for an empty graph."""
    if communication_graph.number_of_nodes() == 0:
        return 0
    return max(degree for _, degree in communication_graph.degree) + 1


def greedy_clique(conflict_graph: nx.Graph) -> tuple[tuple[str, ...], int]:
    """Deterministically grow cliques and return the best one with a step count.

    The step count is the number of vertices added across all attempts and is
    reported so callers can account for the bounding effort.
    """
    nodes = sorted(
        conflict_graph.nodes,
        key=lambda node: (-conflict_graph.degree[node], natural_sort_key(node)),
    )
    best: tuple[str, ...] = ()
    steps = 0
    for seed in nodes[:_MAX_CLIQUE_SEEDS]:
        clique = [seed]
        steps += 1
        candidates = set(conflict_graph.neighbors(seed))
        while candidates:
            chosen = min(
                candidates,
                key=lambda node: (
                    -len(candidates & set(conflict_graph.neighbors(node))),
                    natural_sort_key(node),
                ),
            )
            clique.append(chosen)
            steps += 1
            candidates &= set(conflict_graph.neighbors(chosen))
        if len(clique) > len(best):
            best = tuple(clique)
    return best, steps


def clique_lower_bound(conflict_graph: nx.Graph) -> int:
    """Return the size of a greedily found clique of the conflict graph."""
    return len(greedy_clique(conflict_graph)[0])


def frame_length_lower_bound(
    conflict_graph: nx.Graph,
    communication_graph: nx.Graph | None = None,
) -> int:
    """Return the strongest available proven lower bound on the frame length."""
    bound = clique_lower_bound(conflict_graph)
    if communication_graph is not None:
        bound = max(bound, closed_neighborhood_lower_bound(communication_graph))
    return bound
