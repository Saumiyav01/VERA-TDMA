"""Production command-line orchestration and reporting."""

from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings
from typing import Any

from .bounds import frame_length_lower_bound
from .communication_graph import build_communication_graph
from .config import AppConfig, ConfigError, OutputConfig, load_config
from .conflict_graph import build_conflict_graph
from .coordinates import (
    CoLocatedNodesWarning,
    CoordinateLoadError,
    ImplausibleCoordinateWarning,
    Node,
    make_node,
    warn_for_implausible_coordinates,
)
from .golden import run_golden_tests
from .optimization.class_elimination import eliminate_classes
from .optimization.exact_solver import solve_exact
from .optimization.kempe import kempe_refine
from .ordering import natural_sort_key
from .scheduling.dsatur import dsatur_color
from .verification import (
    VerificationResult,
    verify_matrix_consistency,
    verify_schedule,
)


DEFAULT_EXACT_MAX_NODES = 16

PUBLISHED = "PUBLISHED"
PUBLISHED_BASELINE = "PUBLISHED_BASELINE"
WITHHELD = "WITHHELD"


def load_cli_coordinates(path: str | Path) -> list[Node]:
    """Load the CLI JSON mapping ``node_id -> [x, y]`` or ``node_id -> {x, y}``."""
    input_path = Path(path)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CoordinateLoadError(f"duplicate node identifier in JSON: {key!r}")
            result[key] = value
        return result

    try:
        with input_path.open("r", encoding="utf-8-sig") as input_file:
            raw = json.load(input_file, object_pairs_hook=unique_object)
    except OSError as exc:
        raise CoordinateLoadError(f"could not read input file '{input_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CoordinateLoadError(
            f"could not parse JSON input '{input_path}': {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise CoordinateLoadError("input JSON must be an object mapping node IDs to coordinates")
    if not raw:
        raise CoordinateLoadError("input must contain at least one node")

    nodes: list[Node] = []
    for node_id, coordinate_value in raw.items():
        if isinstance(coordinate_value, dict):
            if set(coordinate_value) != {"x", "y"}:
                raise CoordinateLoadError(
                    f"coordinates for node {node_id!r} must contain exactly x and y fields"
                )
            pair: Any = (coordinate_value["x"], coordinate_value["y"])
        elif isinstance(coordinate_value, list):
            pair = coordinate_value
        else:
            raise CoordinateLoadError(
                f"coordinates for node {node_id!r} must be an [x, y] pair "
                "or an object with x and y fields"
            )
        nodes.append(make_node(node_id, pair, allow_numeric_strings=False))

    nodes.sort(key=lambda node: natural_sort_key(node.node_id))
    coordinate_ids: dict[tuple[float, float], list[str]] = {}
    for node in nodes:
        coordinate_ids.setdefault(node.coordinates_m, []).append(node.node_id)
    for coordinates, identifiers in coordinate_ids.items():
        if len(identifiers) > 1:
            warnings.warn(
                f"nodes {', '.join(identifiers)} share coordinates {coordinates} meters",
                CoLocatedNodesWarning,
                stacklevel=2,
            )
    warn_for_implausible_coordinates(nodes)
    return nodes


def _verify(
    nodes: Sequence[Node],
    coloring: dict[str, int],
    range_m: float,
    inclusive: bool,
) -> VerificationResult | None:
    """Run the independent verifier; ``None`` means it could not run."""
    try:
        return verify_schedule(nodes, coloring, range_m, inclusive)
    except (TypeError, ValueError, ArithmeticError):
        return None


def _status_of(result: VerificationResult | None) -> str:
    if result is None:
        return "NOT RUN"
    return "PASS" if result.valid else "FAIL"


def build_result(
    nodes: Sequence[Node],
    *,
    run_exact: bool = False,
    exact_max_nodes: int = DEFAULT_EXACT_MAX_NODES,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Run the pipeline and return the deterministic machine-readable result.

    Publication gate: the optimized schedule is published only if the
    independent verifier passes it; otherwise the verified DSATUR baseline is
    published instead. If the baseline itself cannot be verified, nothing is
    published (``publication.status == "WITHHELD"``).
    """
    if not nodes:
        raise ValueError("input must contain at least one node")
    if type(exact_max_nodes) is not int or exact_max_nodes < 1:
        raise ValueError("exact_max_nodes must be a positive integer")

    cfg = config if config is not None else AppConfig()
    range_m = cfg.network.communication_range_m
    inclusive = cfg.network.inclusive_range

    ordered_nodes = sorted(nodes, key=lambda node: natural_sort_key(node.node_id))
    canonical_input = [
        {"node_id": node.node_id, "x": float(node.x_m), "y": float(node.y_m)}
        for node in ordered_nodes
    ]
    input_hash = _sha256(canonical_input)

    communication = build_communication_graph(ordered_nodes, range_m, inclusive)
    conflict = build_conflict_graph(communication)
    initial_coloring, initial_slots = dsatur_color(conflict)
    initial_verification = _verify(ordered_nodes, initial_coloring, range_m, inclusive)
    lower_bound = frame_length_lower_bound(conflict, communication)

    class_result = None
    kempe_result = None
    rejected_optimization: dict[str, Any] | None = None
    published_coloring: dict[str, int] | None = None
    final_verification = initial_verification

    if initial_verification is None:
        publication_status = WITHHELD
        publication_detail = "baseline schedule could not be verified; nothing published"
    elif not initial_verification.valid:
        publication_status = WITHHELD
        publication_detail = "baseline schedule failed independent verification; nothing published"
    else:
        publication_status = PUBLISHED
        publication_detail = "optimized schedule passed independent verification"
        candidate = dict(initial_coloring)
        budget = cfg.scheduling.optimization_time_budget_seconds
        deadline = None if budget is None else perf_counter() + budget
        options: dict[str, Any] = {
            "communication_range_m": range_m,
            "inclusive_range": inclusive,
        }
        if deadline is not None:
            options["deadline"] = deadline
        candidate_verification: VerificationResult | None = None
        optimizer_error: str | None = None
        try:
            if cfg.scheduling.enable_class_elimination:
                class_result = eliminate_classes(conflict, candidate, **options)
                candidate = dict(class_result.coloring)
            if cfg.scheduling.enable_kempe_refinement:
                kempe_result = kempe_refine(conflict, candidate, **options)
                candidate = dict(kempe_result.coloring)
            candidate_verification = _verify(ordered_nodes, candidate, range_m, inclusive)
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
            optimizer_error = f"optimizer raised {type(exc).__name__}: {exc}"

        if candidate_verification is not None and candidate_verification.valid:
            published_coloring = candidate
            final_verification = candidate_verification
        else:
            published_coloring = dict(initial_coloring)
            final_verification = initial_verification
            publication_status = PUBLISHED_BASELINE
            publication_detail = (
                "optimized schedule was rejected; verified DSATUR baseline published"
            )
            rejected_optimization = {
                "reason": optimizer_error
                or (
                    "independent verifier could not run on the optimized schedule"
                    if candidate_verification is None
                    else "independent verifier rejected the optimized schedule"
                ),
                "violations": [
                    _violation_to_dict(violation)
                    for violation in (
                        candidate_verification.violations
                        if candidate_verification is not None
                        else ()
                    )
                ],
            }

    ordered_ids = [node.node_id for node in ordered_nodes]
    slot_matrix: dict[str, dict[str, int]] = {}
    schedule_hash: str | None = None
    if published_coloring is not None:
        slot_matrix = {
            str(slot): {
                node_id: int(published_coloring[node_id] == slot)
                for node_id in ordered_ids
            }
            for slot in sorted(set(published_coloring.values()))
        }
        if cfg.validation.check_matrix_consistency:
            matrix_problems = verify_matrix_consistency(
                slot_matrix, ordered_ids, published_coloring
            )
            if matrix_problems:
                final_verification = dataclasses.replace(
                    final_verification,
                    violations=final_verification.violations + matrix_problems,
                )
                published_coloring = None
                slot_matrix = {}
                publication_status = WITHHELD
                publication_detail = "slot matrix is inconsistent with the node-to-slot mapping"
    if published_coloring is not None:
        schedule_hash = _sha256([
            {"node_id": node_id, "slot": published_coloring[node_id]}
            for node_id in ordered_ids
        ])
    published_slots = (
        len(set(published_coloring.values())) if published_coloring is not None else None
    )

    exact_block = _exact_block(
        conflict,
        published_slots,
        run_exact=run_exact,
        exact_max_nodes=exact_max_nodes,
        timeout_seconds=cfg.exact_solver.timeout_seconds,
        lower_bound=lower_bound,
    )
    best_lower_bound = max(lower_bound, exact_block.pop("_search_lower_bound", 0))

    class_attempts = class_result.attempted_eliminations if class_result is not None else None
    audit: dict[str, Any] = {
        "class_elimination_attempts": class_attempts,
        "accepted_eliminations": (
            class_result.successful_eliminations if class_result is not None else None
        ),
        "kempe_attempts": (
            kempe_result.attempted_swaps if kempe_result is not None else None
        ),
        "accepted_kempe_moves": (
            kempe_result.accepted_improvements if kempe_result is not None else None
        ),
        "optimization_budget_exhausted": bool(
            (class_result is not None and class_result.budget_exhausted)
            or (kempe_result is not None and kempe_result.budget_exhausted)
        ),
    }
    if class_attempts is None:
        audit["class_elimination_attempts_note"] = (
            "class elimination did not run (disabled, or no verified baseline)"
        )

    return {
        "input_hash": input_hash,
        "schedule_hash": schedule_hash,
        "publication": {"status": publication_status, "detail": publication_detail},
        "metrics": {
            "total_nodes": len(ordered_nodes),
            "communication_links": communication.number_of_edges(),
            "distance_2_conflict_edges": conflict.number_of_edges(),
            "initial_dsatur_frame_length": initial_slots,
            "optimized_frame_length": published_slots,
            "lower_bound": best_lower_bound,
            "lower_bound_gap": (
                published_slots - best_lower_bound if published_slots is not None else None
            ),
        },
        "verification": _verification_to_dict(
            final_verification, _status_of(final_verification), len(ordered_nodes)
        ),
        "initial_verification": _verification_to_dict(
            initial_verification,
            _status_of(initial_verification),
            len(ordered_nodes),
        ),
        "violations": [
            _violation_to_dict(violation)
            for violation in (
                final_verification.violations if final_verification is not None else ()
            )
        ],
        "rejected_optimization": rejected_optimization,
        "node_to_slot": (
            {node_id: published_coloring[node_id] for node_id in ordered_ids}
            if published_coloring is not None
            else {}
        ),
        "slot_matrix": {
            "legend": {"1": "assigned to slot", "0": "not assigned"},
            "rows": slot_matrix,
            "node_order": ordered_ids,
        },
        "optimization_audit": audit,
        "exact_solver": exact_block,
        "golden_self_tests": None,
    }


def _exact_block(
    conflict: Any,
    published_slots: int | None,
    *,
    run_exact: bool,
    exact_max_nodes: int,
    timeout_seconds: float,
    lower_bound: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "SKIPPED",
        "requested": run_exact,
        "max_nodes": exact_max_nodes,
        "time_limit_seconds": timeout_seconds,
        "chromatic_number": None,
        "lower_bound": lower_bound,
        "upper_bound": None,
        "nodes_explored": None,
        "runtime_seconds": None,
        "optimality_gap": None,
        "relative_optimality_gap": None,
    }
    if not run_exact:
        return base
    if conflict.number_of_nodes() > exact_max_nodes:
        base["status"] = "BOUNDED/SKIPPED"
        return base

    exact_result = solve_exact(conflict, time_limit_seconds=timeout_seconds)
    base.update(
        nodes_explored=exact_result.nodes_explored,
        runtime_seconds=exact_result.runtime_seconds,
        lower_bound=max(lower_bound, exact_result.lower_bound),
        upper_bound=exact_result.upper_bound,
    )
    base["_search_lower_bound"] = exact_result.lower_bound
    if not exact_result.is_optimal:
        # Time/node budget exhausted: report the interval, never an optimum.
        base["status"] = "BOUNDED"
        return base

    base["status"] = "EXACT"
    base["chromatic_number"] = exact_result.chromatic_number
    if published_slots is not None:
        base["optimality_gap"] = published_slots - exact_result.chromatic_number
        base["relative_optimality_gap"] = (
            (published_slots - exact_result.chromatic_number) / exact_result.chromatic_number
            if exact_result.chromatic_number
            else 0.0
        )
    return base


def _filter_output(result: dict[str, Any], output: OutputConfig) -> dict[str, Any]:
    """Drop sections the ``output`` configuration section turns off."""
    dropped = {
        "node_to_slot": output.include_node_slot_mapping,
        "slot_matrix": output.include_slot_node_matrix,
        "violations": output.include_conflict_details,
        "optimization_audit": output.include_optimization_metrics,
        "exact_solver": output.include_exact_solver_metrics,
    }
    return {key: value for key, value in result.items() if dropped.get(key, True)}


def format_report(result: dict[str, Any]) -> str:
    """Format the machine result as a readable terminal report."""
    metrics = result["metrics"]
    verification = result["verification"]
    exact = result.get("exact_solver")
    audit = result.get("optimization_audit")
    publication = result["publication"]
    withheld = publication["status"] == WITHHELD

    def shown(value: Any, missing: str = "N/A") -> Any:
        return missing if value is None else value

    lines = [
        "VERA-TDMA Schedule Report",
        "=========================",
        "",
        "Network",
        f"  Total Nodes: {metrics['total_nodes']}",
        f"  Communication Links: {metrics['communication_links']}",
        f"  Distance-2 Conflict Edges: {metrics['distance_2_conflict_edges']}",
        "",
        "Scheduling",
        f"  Initial DSATUR Frame Length: {metrics['initial_dsatur_frame_length']}",
        f"  Optimized Frame Length: {shown(metrics['optimized_frame_length'])}",
        f"  Proven Lower Bound: {metrics['lower_bound']}",
        f"  Lower-Bound Gap (not an exact gap): {shown(metrics['lower_bound_gap'])}",
    ]
    if exact is not None:
        lines.extend([
            f"  Exact Optimum / Bounded: {exact['chromatic_number'] if exact['chromatic_number'] is not None else exact['status']}",
            f"  Optimality Gap: {shown(exact['optimality_gap'])}",
        ])
    lines.extend([
        f"  Verification Status: {verification['status']}",
        f"  Conflict Violations: {verification['number_of_violations']}",
        f"  Publication: {publication['status']} ({publication['detail']})",
    ])
    if result.get("golden_self_tests"):
        golden = result["golden_self_tests"]
        lines.append(f"  Golden Self-Tests: {golden['status']} ({golden['passed']}/{golden['total']})")

    if not withheld:
        if "node_to_slot" in result:
            lines.extend(["", "Schedule", "  Node -> Slot:"])
            for node_id, slot in result["node_to_slot"].items():
                lines.append(f"    {node_id}: {slot}")
        if "slot_matrix" in result:
            lines.extend(
                [
                    "  Slot x Node Boolean Matrix (1 = assigned, 0 = not assigned):",
                    f"    Nodes: {' | '.join(result['slot_matrix']['node_order'])}",
                ]
            )
            for slot, values in result["slot_matrix"]["rows"].items():
                lines.append(f"    Slot {slot}: {' | '.join(str(values[node]) for node in result['slot_matrix']['node_order'])}")

    if audit is not None:
        attempts = audit["class_elimination_attempts"]
        lines.extend(
            [
                "",
                "Optimization Audit",
                f"  Class Elimination Attempts: {attempts if attempts is not None else 'N/A (did not run)'}",
                f"  Accepted Eliminations: {audit['accepted_eliminations']}",
                f"  Kempe Attempts: {audit['kempe_attempts']}",
                f"  Accepted Kempe Moves: {audit['accepted_kempe_moves']}",
                f"  Time Budget Exhausted: {audit['optimization_budget_exhausted']}",
            ]
        )
    if exact is not None:
        runtime = exact["runtime_seconds"]
        lines.extend(
            [
                "",
                "Exact Solver",
                f"  Status: {exact['status']}",
                f"  Runtime: {f'{runtime:.4f} s' if runtime is not None else 'N/A'}",
                f"  Nodes Explored: {shown(exact['nodes_explored'])}",
                f"  Exact Optimum: {shown(exact['chromatic_number'])}",
                f"  Proven Bounds on Frame Length: {exact['lower_bound']} <= K*"
                + (f" <= {exact['upper_bound']}" if exact["upper_bound"] is not None else ""),
            ]
        )
    lines.extend(
        [
            "",
            f"Input SHA-256: {result['input_hash']}",
            f"Schedule SHA-256: {shown(result['schedule_hash'])}",
        ]
    )
    rejected = result.get("rejected_optimization")
    if rejected:
        lines.append(f"Optimized schedule rejected: {rejected['reason']}")

    if verification["status"] == "PASS" and not withheld:
        direct = sum(
            1 for v in result.get("violations", ())
            if v["conflict_type"] in ("direct", "direct_and_two_hop")
        )
        two_hop = sum(
            1 for v in result.get("violations", ())
            if v["conflict_type"] in ("two_hop", "direct_and_two_hop")
        )
        lines.append("Schedule verified conflict-free.")
        lines.append(
            f"Pairs examined: {verification['direct_conflicts_checked']} direct, "
            f"{verification['two_hop_conflicts_checked']} two-hop"
        )
        lines.append(f"Direct-link conflicts checked: {direct} violations.")
        lines.append(f"Two-hop hidden-terminal conflicts checked: {two_hop} violations.")
        lines.append("Execution finalized cleanly.")
    elif verification["status"] == "FAIL":
        lines.append("VERIFICATION: FAIL")
        lines.append("  Violations:")
        for violation in result.get("violations", ()):
            lines.append(
                f"    {violation['node_1']} ({violation['color_1']}) - "
                f"{violation['node_2']} ({violation['color_2']}): "
                f"{violation['conflict_type']} - {violation['message']}"
            )
    else:
        lines.append("VERIFICATION: NOT RUN")
    if withheld:
        lines.append("SCHEDULE WITHHELD: no schedule is published unless it passes verification.")
    return "\n".join(lines)


def _verification_to_dict(
    result: VerificationResult | None,
    status: str,
    number_of_nodes: int,
) -> dict[str, Any]:
    if result is None:
        return {
            "status": status,
            "valid": None,
            "number_of_nodes": number_of_nodes,
            "direct_conflicts_checked": None,
            "two_hop_conflicts_checked": None,
            "number_of_violations": None,
            "coordinates_unchanged": None,
        }
    return {
        "status": status,
        "valid": result.valid,
        "number_of_nodes": result.number_of_nodes,
        "direct_conflicts_checked": result.direct_conflicts_checked,
        "two_hop_conflicts_checked": result.two_hop_conflicts_checked,
        "number_of_violations": result.number_of_violations,
        "coordinates_unchanged": result.coordinates_unchanged,
    }


def _violation_to_dict(violation: Any) -> dict[str, Any]:
    return {
        "node_1": violation.node_1,
        "node_2": violation.node_2,
        "color_1": violation.color_1,
        "color_2": violation.color_2,
        "conflict_type": violation.conflict_type,
        "distance_m": violation.distance_m,
        "path": list(violation.path) if violation.path is not None else None,
        "message": violation.message,
    }


def _sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="vera-tdma",
        description="Generate and independently verify a VERA-TDMA schedule.",
    )
    parser.add_argument(
        "--input",
        help="JSON node-coordinate input (default: paths.input_nodes from the configuration)",
    )
    parser.add_argument("--config", help="YAML configuration file (default: built-in defaults)")
    parser.add_argument("--output", help="write complete deterministic JSON result")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="run the exact solver if within the node limit (also enabled by exact_solver.enabled)",
    )
    parser.add_argument(
        "--no-exact", action="store_true", help="never run the exact solver"
    )
    parser.add_argument(
        "--exact-max-nodes",
        type=int,
        default=None,
        help=(
            "maximum nodes for exact solving "
            f"(default: exact_solver.max_nodes, normally {DEFAULT_EXACT_MAX_NODES})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.exact_max_nodes is not None and args.exact_max_nodes < 1:
        print("error: --exact-max-nodes must be a positive integer", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config) if args.config else AppConfig()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    exact_max_nodes = (
        args.exact_max_nodes if args.exact_max_nodes is not None else config.exact_solver.max_nodes
    )
    run_exact = (args.exact or config.exact_solver.enabled) and not args.no_exact
    input_path = args.input if args.input else str(config.paths.input_nodes)

    golden_summary = None
    if config.validation.run_golden_tests:
        golden_results = run_golden_tests()
        failed = [outcome for outcome in golden_results if not outcome.passed]
        if failed:
            for outcome in failed:
                print(f"error: golden self-test failed: {outcome.name}: {outcome.detail}", file=sys.stderr)
            return 3
        golden_summary = {
            "status": "PASS",
            "passed": len(golden_results),
            "total": len(golden_results),
        }

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", CoLocatedNodesWarning)
            warnings.simplefilter("always", ImplausibleCoordinateWarning)
            nodes = load_cli_coordinates(input_path)
        for warning in captured:
            print(f"warning: {warning.message}", file=sys.stderr)
        result = build_result(
            nodes,
            run_exact=run_exact,
            exact_max_nodes=exact_max_nodes,
            config=config,
        )
    except (CoordinateLoadError, ValueError, ArithmeticError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result["golden_self_tests"] = golden_summary

    published = result["publication"]["status"] != WITHHELD
    visible = _filter_output(result, config.output)
    report = format_report(visible)
    print(report)
    if args.output:
        if not published:
            print(
                "error: --output not written because no verified schedule exists",
                file=sys.stderr,
            )
        else:
            output_path = Path(args.output)
            try:
                output_path.write_text(
                    json.dumps(visible, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                print(f"error: could not write output file '{output_path}': {exc}", file=sys.stderr)
                return 2
    if published:
        return 0
    # validation.fail_on_verification_error only selects the exit status; the
    # schedule itself is never published when verification fails.
    return 1 if config.validation.fail_on_verification_error else 0
