import itertools
import random
import time
import unittest

import networkx as nx

from vera_tdma.bounds import (
    closed_neighborhood_lower_bound,
    clique_lower_bound,
    frame_length_lower_bound,
)
from vera_tdma.communication_graph import build_communication_graph
from vera_tdma.conflict_graph import build_conflict_graph
from vera_tdma.coordinates import Node
from vera_tdma.optimization.class_elimination import eliminate_classes
from vera_tdma.optimization.exact_solver import solve_exact
from vera_tdma.optimization.kempe import kempe_refine
from vera_tdma.scheduling.dsatur import dsatur_color


def random_conflict_graph(seed, count, size):
    rng = random.Random(seed)
    nodes = [Node(f"N{i}", rng.uniform(0, size), rng.uniform(0, size)) for i in range(count)]
    communication = build_communication_graph(nodes)
    return nodes, communication, build_conflict_graph(communication)


class ExactSolverBudgetTests(unittest.TestCase):
    def five_cycle(self):
        graph = nx.cycle_graph(5)
        return nx.relabel_nodes(graph, {i: f"N{i}" for i in range(5)})

    def test_finished_search_is_labelled_exact(self):
        result = solve_exact(self.five_cycle())
        self.assertEqual(result.status, "EXACT")
        self.assertTrue(result.is_optimal)
        self.assertEqual(result.lower_bound, result.chromatic_number)

    def test_node_budget_yields_bounded_never_an_unproven_optimum(self):
        result = solve_exact(self.five_cycle(), max_search_nodes=1)
        self.assertEqual(result.status, "BOUNDED")
        self.assertFalse(result.is_optimal)
        self.assertLess(result.lower_bound, result.upper_bound)
        graph = self.five_cycle()
        for first, second in graph.edges:
            self.assertNotEqual(result.optimal_coloring[first], result.optimal_coloring[second])

    def test_node_budget_is_deterministic(self):
        first = solve_exact(self.five_cycle(), max_search_nodes=1)
        second = solve_exact(self.five_cycle(), max_search_nodes=1)
        self.assertEqual(first.nodes_explored, second.nodes_explored)
        self.assertEqual(first.optimal_coloring, second.optimal_coloring)

    def test_wall_clock_limit_is_enforced_on_a_hard_instance(self):
        # 60 random nodes used to run for tens of seconds in some cases.
        hardest = None
        for seed in range(12):
            _, _, conflict = random_conflict_graph(seed, 60, 1500)
            started = time.perf_counter()
            result = solve_exact(conflict, time_limit_seconds=0.2)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 5.0)
            if result.status == "BOUNDED":
                hardest = result
        if hardest is not None:  # only meaningful when a case actually timed out
            self.assertFalse(hardest.is_optimal)
            self.assertLessEqual(hardest.lower_bound, hardest.upper_bound)

    def test_invalid_budgets_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_exact(self.five_cycle(), time_limit_seconds=0)
        with self.assertRaises(ValueError):
            solve_exact(self.five_cycle(), max_search_nodes=0)

    def test_matches_brute_force_chromatic_number_on_small_random_graphs(self):
        rng = random.Random(4)
        for _ in range(60):
            count = rng.randint(2, 7)
            names = [f"N{i}" for i in range(count)]
            graph = nx.Graph()
            graph.add_nodes_from(names)
            graph.add_edges_from(
                pair for pair in itertools.combinations(names, 2) if rng.random() < 0.5
            )
            expected = next(
                k
                for k in range(1, count + 1)
                if any(
                    all(assign[names.index(a)] != assign[names.index(b)] for a, b in graph.edges)
                    for assign in itertools.product(range(k), repeat=count)
                )
            )
            self.assertEqual(solve_exact(graph).chromatic_number, expected)

    def test_known_graphs(self):
        petersen = nx.relabel_nodes(nx.petersen_graph(), lambda i: f"P{i}")
        self.assertEqual(solve_exact(petersen).chromatic_number, 3)
        complete = nx.relabel_nodes(nx.complete_graph(6), lambda i: f"K{i}")
        self.assertEqual(solve_exact(complete).chromatic_number, 6)


class LowerBoundTests(unittest.TestCase):
    def test_star_with_five_leaves_needs_six_slots(self):
        nodes = [Node("Hub", 0, 0)] + [
            Node(f"L{i}", x, y)
            for i, (x, y) in enumerate([(500, 0), (-500, 0), (0, 500), (0, -500), (300, 400)])
        ]
        communication = build_communication_graph(nodes)
        conflict = build_conflict_graph(communication)
        self.assertEqual(closed_neighborhood_lower_bound(communication), 6)
        self.assertEqual(clique_lower_bound(conflict), 6)
        self.assertEqual(frame_length_lower_bound(conflict, communication), 6)

    def test_empty_graph_bound_is_zero(self):
        self.assertEqual(closed_neighborhood_lower_bound(nx.Graph()), 0)

    def test_bound_never_exceeds_the_exact_optimum(self):
        for seed in range(40):
            _, communication, conflict = random_conflict_graph(seed, 14, 1200)
            bound = frame_length_lower_bound(conflict, communication)
            self.assertLessEqual(bound, solve_exact(conflict).chromatic_number)
            self.assertGreaterEqual(bound, 1)


class OptimizerBudgetTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.communication, self.conflict = random_conflict_graph(2, 25, 1500)
        self.coloring, _ = dsatur_color(self.conflict)

    def test_class_elimination_reports_measured_attempts(self):
        result = eliminate_classes(self.conflict, self.coloring)
        slots = len(set(self.coloring.values()))
        self.assertGreaterEqual(result.attempted_eliminations, 1)
        self.assertGreaterEqual(result.attempted_eliminations, result.successful_eliminations)
        self.assertLessEqual(result.slots_after, slots)
        self.assertFalse(result.budget_exhausted)

    def test_expired_deadline_stops_refinement_but_keeps_a_valid_schedule(self):
        expired = time.perf_counter() - 1.0
        eliminated = eliminate_classes(self.conflict, self.coloring, deadline=expired)
        self.assertTrue(eliminated.budget_exhausted)
        self.assertEqual(eliminated.coloring, self.coloring)
        refined = kempe_refine(self.conflict, self.coloring, deadline=expired)
        self.assertTrue(refined.budget_exhausted)
        self.assertTrue(refined.final_verification.valid)
        self.assertEqual(refined.coloring, self.coloring)


if __name__ == "__main__":
    unittest.main()
