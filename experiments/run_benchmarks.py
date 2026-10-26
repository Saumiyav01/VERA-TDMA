"""Benchmark VERA-TDMA against naive baselines and the exact optimum.

Usage (from the project root)::

    python experiments/run_benchmarks.py [--seeds 5] [--sizes 16 30 60] [--out experiments/results]

For every topology family, size and seed it records the first-fit frame length
(ID order, Distance-2 aware), the DSATUR frame length, the optimized frame
length, the proven lower bound, and (for small instances) the exact optimum
with the optimality gap. It also counts how often a *1-hop-only* coloring (which
ignores hidden terminals) is rejected by the independent verifier.
"""

import argparse
import csv
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # noqa: E402

from vera_tdma.cli import build_result  # noqa: E402
from vera_tdma.communication_graph import build_communication_graph  # noqa: E402
from vera_tdma.conflict_graph import build_conflict_graph  # noqa: E402
from vera_tdma.coordinates import Node  # noqa: E402
from vera_tdma.ordering import natural_sort_key  # noqa: E402
from vera_tdma.verification import verify_schedule  # noqa: E402


def _nodes(points):
    return [Node(f"N{i:03d}", float(x), float(y)) for i, (x, y) in enumerate(points)]


def sparse_random(n, rng):
    side = 300 * math.sqrt(n) * 1.8
    return _nodes([(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)])


def dense_random(n, rng):
    side = 300 * math.sqrt(n) * 0.8
    return _nodes([(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)])


def grid(n, rng):
    width = max(1, round(math.sqrt(n)))
    return _nodes([((i % width) * 300, (i // width) * 300) for i in range(n)])


def clustered(n, rng):
    centers = [(rng.uniform(0, 3000), rng.uniform(0, 3000)) for _ in range(max(2, n // 8))]
    points = []
    for _ in range(n):
        cx, cy = rng.choice(centers)
        points.append((cx + rng.gauss(0, 120), cy + rng.gauss(0, 120)))
    return _nodes(points)


def chain(n, rng):
    return _nodes([(i * 350, rng.uniform(-30, 30)) for i in range(n)])


def hub_spoke(n, rng):
    points = [(0.0, 0.0)]
    for i in range(1, n):
        angle = 2 * math.pi * i / (n - 1)
        radius = rng.uniform(300, 480)
        points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return _nodes(points)


FAMILIES = {
    "sparse_random": sparse_random,
    "dense_random": dense_random,
    "grid": grid,
    "clustered": clustered,
    "chain": chain,
    "hub_spoke": hub_spoke,
}


def first_fit(conflict):
    """Naive baseline: assign the smallest legal slot in node-ID order."""
    colors = {}
    for node in sorted(conflict.nodes, key=natural_sort_key):
        taken = {colors[m] for m in conflict.neighbors(node) if m in colors}
        colors[node] = next(c for c in range(len(conflict) + 1) if c not in taken)
    return colors


def one_hop_only(communication):
    """Greedy coloring of G (not G^2): ignores hidden-terminal conflicts."""
    return nx.greedy_color(communication, strategy="largest_first")


def run(sizes, seeds, exact_limit):
    rows = []
    for family, make in FAMILIES.items():
        for n in sizes:
            for seed in range(seeds):
                rng = random.Random(f"{family}-{n}-{seed}")
                nodes = make(n, rng)
                communication = build_communication_graph(nodes)
                conflict = build_conflict_graph(communication)
                result = build_result(
                    nodes, run_exact=n <= exact_limit, exact_max_nodes=exact_limit
                )
                hop_one = one_hop_only(communication)
                hop_one_valid = verify_schedule(nodes, hop_one).valid
                exact = result["exact_solver"]
                rows.append(
                    {
                        "family": family,
                        "n": n,
                        "seed": seed,
                        "links": result["metrics"]["communication_links"],
                        "first_fit": len(set(first_fit(conflict).values())),
                        "dsatur": result["metrics"]["initial_dsatur_frame_length"],
                        "optimized": result["metrics"]["optimized_frame_length"],
                        "lower_bound": result["metrics"]["lower_bound"],
                        "exact": exact["chromatic_number"],
                        "exact_status": exact["status"],
                        "gap_vs_exact": exact["optimality_gap"],
                        "one_hop_only_valid": hop_one_valid,
                        "verification": result["verification"]["status"],
                    }
                )
    return rows


def summarize(rows):
    lines = [
        f"{'family':<14}{'n':>4}{'runs':>5}{'FF':>7}{'DSATUR':>8}{'opt':>7}{'LB':>7}"
        f"{'exact':>7}{'max gap':>9}{'1-hop invalid':>15}"
    ]
    for family in FAMILIES:
        for n in sorted({r["n"] for r in rows if r["family"] == family}):
            group = [r for r in rows if r["family"] == family and r["n"] == n]
            exact_values = [r["exact"] for r in group if r["exact"] is not None]
            gaps = [r["gap_vs_exact"] for r in group if r["gap_vs_exact"] is not None]
            invalid = sum(1 for r in group if not r["one_hop_only_valid"])
            mean = lambda key: statistics.mean(r[key] for r in group)  # noqa: E731
            lines.append(
                f"{family:<14}{n:>4}{len(group):>5}{mean('first_fit'):>7.1f}{mean('dsatur'):>8.1f}"
                f"{mean('optimized'):>7.1f}{mean('lower_bound'):>7.1f}"
                f"{(statistics.mean(exact_values) if exact_values else float('nan')):>7.1f}"
                f"{(max(gaps) if gaps else 'n/a'):>9}{invalid:>10}/{len(group)}"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 30, 60])
    parser.add_argument("--exact-limit", type=int, default=16)
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    rows = run(args.sizes, args.seeds, args.exact_limit)
    if any(r["verification"] != "PASS" for r in rows):
        raise SystemExit("benchmark aborted: a published schedule failed verification")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\n{len(rows)} instances; all published schedules passed independent verification.")
    print(f"Wrote {out_dir / 'benchmark.csv'} and {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
