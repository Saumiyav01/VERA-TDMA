"""Built-in golden self-tests: hand-calculated topologies with known answers.

``run_golden_tests`` runs the full pipeline (graph, conflict graph, DSATUR,
class elimination, Kempe refinement, independent verification, exact solver) on
each case and checks the expected frame length. It is wired to
``validation.run_golden_tests`` in the configuration.
"""

from dataclasses import dataclass

from .bounds import frame_length_lower_bound
from .communication_graph import build_communication_graph
from .conflict_graph import build_conflict_graph
from .coordinates import Node
from .optimization.class_elimination import eliminate_classes
from .optimization.exact_solver import solve_exact
from .optimization.kempe import kempe_refine
from .scheduling.dsatur import dsatur_color
from .verification import verify_schedule


@dataclass(frozen=True)
class GoldenCase:
    name: str
    nodes: tuple[Node, ...]
    expected_slots: int | None  # None: computed, checked against exact optimum
    expected_pattern: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GoldenResult:
    name: str
    passed: bool
    slots: int | None
    detail: str


def _line(count: int, spacing: float) -> tuple[Node, ...]:
    return tuple(Node(chr(ord("A") + i), i * spacing, 0.0) for i in range(count))


def golden_cases() -> tuple[GoldenCase, ...]:
    star = (Node("Hub", 0, 0),) + (
        Node("L1", 500, 0),
        Node("L2", -500, 0),
        Node("L3", 0, 500),
        Node("L4", 0, -500),
        Node("L5", 300, 400),
    )
    grid = tuple(
        Node(f"G{row}{col}", col * 300.0, row * 300.0)
        for row in range(4)
        for col in range(4)
    )
    return (
        GoldenCase("two nodes 100 m apart", _line(2, 100), 2),
        GoldenCase("A-B-C at 300 m spacing", _line(3, 300), 3),
        GoldenCase("A-B-C-D-E at 300 m spacing", _line(5, 300), 3, (0, 1, 2, 0, 1)),
        GoldenCase("two nodes at 499 m", _line(2, 499), 2),
        GoldenCase("two nodes at exactly 500 m (inclusive)", _line(2, 500), 2),
        GoldenCase("two nodes at 501 m", _line(2, 501), 1),
        GoldenCase("star with 5 leaves", star, 6),
        GoldenCase(
            "two disconnected pairs more than 1000 m apart",
            (Node("A", 0, 0), Node("B", 100, 0), Node("C", 2000, 0), Node("D", 2100, 0)),
            2,
        ),
        GoldenCase(
            "duplicate coordinates are connected",
            (Node("A", 10, 10), Node("B", 10, 10)),
            2,
        ),
        GoldenCase("4x4 grid at 300 m (computed, compared with exact)", grid, None),
    )


def _canonical_pattern(pattern: tuple[int, ...]) -> tuple[int, ...]:
    """Relabel slots by first appearance so patterns compare up to slot renaming."""
    labels: dict[int, int] = {}
    return tuple(labels.setdefault(slot, len(labels)) for slot in pattern)


def run_golden_case(case: GoldenCase) -> GoldenResult:
    nodes = list(case.nodes)
    communication = build_communication_graph(nodes)
    conflict = build_conflict_graph(communication)
    initial, _ = dsatur_color(conflict)
    eliminated = eliminate_classes(conflict, initial)
    refined = kempe_refine(conflict, eliminated.coloring)
    coloring = refined.coloring
    slots = len(set(coloring.values()))
    verification = verify_schedule(nodes, coloring)
    exact = solve_exact(conflict, time_limit_seconds=10.0)
    lower = frame_length_lower_bound(conflict, communication)

    problems: list[str] = []
    if not verification.valid:
        problems.append("independent verifier rejected the schedule")
    if not exact.is_optimal:
        problems.append("exact solver did not finish")
    elif slots != exact.chromatic_number:
        problems.append(f"{slots} slots but exact optimum is {exact.chromatic_number}")
    if slots < lower:
        problems.append(f"{slots} slots is below the proven lower bound {lower}")
    if case.expected_slots is not None and slots != case.expected_slots:
        problems.append(f"expected {case.expected_slots} slots, got {slots}")
    if case.expected_pattern is not None:
        pattern = tuple(
            coloring[node.node_id]
            for node in sorted(nodes, key=lambda node: node.node_id)
        )
        if _canonical_pattern(pattern) != case.expected_pattern:
            problems.append(f"expected pattern {case.expected_pattern}, got {pattern}")
    detail = "; ".join(problems) if problems else "ok"
    return GoldenResult(case.name, not problems, slots, detail)


def run_golden_tests() -> tuple[GoldenResult, ...]:
    """Run every golden case; a raised exception is reported as a failed case."""
    results: list[GoldenResult] = []
    for case in golden_cases():
        try:
            results.append(run_golden_case(case))
        except Exception as exc:  # noqa: BLE001 - a crash is a golden failure
            results.append(GoldenResult(case.name, False, None, f"crashed: {exc!r}"))
    return tuple(results)
