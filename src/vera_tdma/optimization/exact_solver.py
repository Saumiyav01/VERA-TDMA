"""Deterministic exact chromatic-number solver for small conflict graphs.

Exact graph coloring is exponential in the worst case; this module is intended
as a reference solver for small graphs, not as a production scheduler. A wall
clock limit and/or a search-node limit can be supplied; when either is hit the
solver stops and reports ``BOUNDED`` with the best feasible coloring and the
largest lower bound proven so far, never an unproven optimum.
"""

from dataclasses import dataclass
from time import perf_counter

import networkx as nx

from ..bounds import greedy_clique
from ..ordering import natural_sort_key
from ..scheduling.dsatur import dsatur_color

STATUS_EXACT = "EXACT"
STATUS_BOUNDED = "BOUNDED"
_TIME_CHECK_INTERVAL = 256


class _BudgetExceeded(Exception):
    """Internal signal that the time or node budget was exhausted."""


@dataclass(frozen=True)
class ExactSolverResult:
    """Optimal coloring and search evidence returned by :func:`solve_exact`."""

    optimal_coloring: dict[str, int]
    chromatic_number: int
    nodes_explored: int
    runtime_seconds: float
    is_optimal: bool
    infeasible_color_counts: tuple[int, ...]
    lower_bound: int = 0
    status: str = STATUS_EXACT

    @property
    def upper_bound(self) -> int:
        """Number of colors in the returned (always valid) coloring."""
        return len(set(self.optimal_coloring.values()))


def solve_exact(
    conflict_graph: nx.Graph,
    *,
    time_limit_seconds: float | None = None,
    max_search_nodes: int | None = None,
) -> ExactSolverResult:
    """Return a minimum coloring, proving every smaller color count infeasible.

    The best of a degree-ordered greedy coloring and DSATUR supplies a feasible
    upper bound, and a greedy clique supplies a proven lower bound; every count
    below the clique size is infeasible by that bound. Feasibility is then
    tested upward from the lower bound using exhaustive backtracking with
    color-symmetry breaking. The first feasible count is optimal because every
    smaller positive count was disproved.

    ``nodes_explored`` counts clique-construction steps plus search nodes.
    If ``time_limit_seconds`` or ``max_search_nodes`` is exhausted, the result
    has ``is_optimal=False`` and ``status="BOUNDED"``: ``optimal_coloring`` is
    the best feasible coloring found (its size is the upper bound) and
    ``lower_bound`` is the largest count proven necessary. A ``max_search_nodes``
    limit is deterministic; a wall-clock limit is not.
    """
    started = perf_counter()
    _validate_graph(conflict_graph)
    if time_limit_seconds is not None and not time_limit_seconds > 0:
        raise ValueError("time_limit_seconds must be greater than zero")
    if max_search_nodes is not None and (
        type(max_search_nodes) is not int or max_search_nodes < 1
    ):
        raise ValueError("max_search_nodes must be a positive integer")
    deadline = None if time_limit_seconds is None else started + time_limit_seconds
    nodes = sorted(conflict_graph.nodes, key=natural_sort_key)
    if not nodes:
        return ExactSolverResult({}, 0, 0, perf_counter() - started, True, ())

    search_order = sorted(
        nodes,
        key=lambda node: (-conflict_graph.degree[node], natural_sort_key(node)),
    )
    upper_coloring = _greedy_coloring(conflict_graph, search_order)
    dsatur_coloring, _ = dsatur_color(conflict_graph)
    if len(set(dsatur_coloring.values())) < len(set(upper_coloring.values())):
        upper_coloring = dsatur_coloring
    upper_bound = len(set(upper_coloring.values()))
    clique, clique_steps = greedy_clique(conflict_graph)
    lower_bound = min(max(len(clique), 1), upper_bound)
    explored = clique_steps
    # Counts below the clique size are infeasible by the clique bound.
    infeasible_counts: list[int] = list(range(1, lower_bound))

    for color_limit in range(lower_bound, upper_bound):
        assigned: dict[str, int] = {}

        def search(index: int, used_colors: int) -> dict[str, int] | None:
            nonlocal explored
            explored += 1
            if (
                max_search_nodes is not None and explored - clique_steps > max_search_nodes
            ) or (
                deadline is not None
                and explored % _TIME_CHECK_INTERVAL == 0
                and perf_counter() > deadline
            ):
                raise _BudgetExceeded
            if index == len(search_order):
                return dict(assigned)

            node = search_order[index]
            forbidden = {
                assigned[neighbor]
                for neighbor in conflict_graph.neighbors(node)
                if neighbor in assigned
            }
            largest_color_to_try = min(used_colors, color_limit - 1)
            for color in range(largest_color_to_try + 1):
                new_color_count = max(used_colors, color + 1)
                if color in forbidden or new_color_count > color_limit:
                    continue
                assigned[node] = color
                if _remaining_nodes_have_available_colors(
                    conflict_graph, search_order[index + 1 :], assigned, color_limit
                ):
                    solution = search(index + 1, new_color_count)
                    if solution is not None:
                        return solution
                del assigned[node]
            return None

        try:
            solution = search(0, 0)
        except _BudgetExceeded:
            coloring = _normalize(upper_coloring)
            _validate_solution(conflict_graph, coloring, upper_bound)
            return ExactSolverResult(
                optimal_coloring=coloring,
                chromatic_number=upper_bound,
                nodes_explored=explored,
                runtime_seconds=perf_counter() - started,
                is_optimal=False,
                infeasible_color_counts=tuple(infeasible_counts),
                lower_bound=color_limit,
                status=STATUS_BOUNDED,
            )
        if solution is None:
            infeasible_counts.append(color_limit)
            continue

        coloring = _normalize(solution)
        _validate_solution(conflict_graph, coloring, color_limit)
        return ExactSolverResult(
            optimal_coloring=coloring,
            chromatic_number=len(set(coloring.values())),
            nodes_explored=explored,
            runtime_seconds=perf_counter() - started,
            is_optimal=True,
            infeasible_color_counts=tuple(infeasible_counts),
            lower_bound=color_limit,
            status=STATUS_EXACT,
        )

    coloring = _normalize(upper_coloring)
    _validate_solution(conflict_graph, coloring, upper_bound)
    return ExactSolverResult(
        optimal_coloring=coloring,
        chromatic_number=len(set(coloring.values())),
        nodes_explored=explored,
        runtime_seconds=perf_counter() - started,
        is_optimal=True,
        infeasible_color_counts=tuple(infeasible_counts),
        lower_bound=upper_bound,
        status=STATUS_EXACT,
    )


def _validate_graph(conflict_graph: nx.Graph) -> None:
    if not isinstance(conflict_graph, nx.Graph):
        raise TypeError("conflict_graph must be a NetworkX graph")
    if conflict_graph.is_directed() or conflict_graph.is_multigraph():
        raise ValueError("conflict_graph must be an undirected simple graph")
    if any(not isinstance(node, str) for node in conflict_graph.nodes):
        raise ValueError("conflict graph node identifiers must be strings")
    if nx.number_of_selfloops(conflict_graph):
        raise ValueError("conflict_graph must not contain self-loops")


def _greedy_coloring(
    conflict_graph: nx.Graph,
    search_order: list[str],
) -> dict[str, int]:
    coloring: dict[str, int] = {}
    for node in search_order:
        forbidden = {
            coloring[neighbor]
            for neighbor in conflict_graph.neighbors(node)
            if neighbor in coloring
        }
        color = 0
        while color in forbidden:
            color += 1
        coloring[node] = color
    return coloring


def _remaining_nodes_have_available_colors(
    conflict_graph: nx.Graph,
    remaining: list[str],
    assigned: dict[str, int],
    color_limit: int,
) -> bool:
    for node in remaining:
        forbidden = {
            assigned[neighbor]
            for neighbor in conflict_graph.neighbors(node)
            if neighbor in assigned
        }
        if len(forbidden) == color_limit:
            return False
    return True


def _normalize(coloring: dict[str, int]) -> dict[str, int]:
    remapping = {
        old_color: new_color
        for new_color, old_color in enumerate(sorted(set(coloring.values())))
    }
    return {
        node: remapping[color]
        for node, color in sorted(coloring.items(), key=lambda item: natural_sort_key(item[0]))
    }


def _validate_solution(
    conflict_graph: nx.Graph,
    coloring: dict[str, int],
    chromatic_number: int,
) -> None:
    if set(coloring) != set(conflict_graph.nodes):
        raise RuntimeError("exact solver returned an incomplete coloring")
    if any(type(color) is not int or color < 0 for color in coloring.values()):
        raise RuntimeError("exact solver returned an invalid color value")
    used_colors = set(coloring.values())
    if used_colors != set(range(chromatic_number)):
        raise RuntimeError("exact solver color count is not normalized")
    for first, second in conflict_graph.edges:
        if coloring[first] == coloring[second]:
            raise RuntimeError(
                f"exact solver returned a conflict on edge {first!r}-{second!r}"
            )
