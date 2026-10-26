# Patch notes

Only the weaknesses from the review were patched. Untouched modules:
`dsatur.py`, `conflict_graph.py`, `ordering.py`, `scheduling/`, `__init__.py`,
`__main__.py`. On 300 random topologies the patched pipeline produces
schedules, hashes and metrics identical to the original.

## Defects
| # | Weakness | Fix | Where |
|---|---|---|---|
| 1 | CLI rejected the spec's `[x, y]` input | CLI accepts `[x, y]` and `{x, y}`; private `_make_node` import replaced by public `make_node` | `cli.py`, `coordinates.py` |
| 2 | No publication gate | Optimized schedule published only if verified; otherwise verified baseline; baseline failure/unverifiable -> `SCHEDULE WITHHELD`, no `--output`, non-zero exit; optimizer exceptions also fall back | `cli.py` |
| 3 | Verifier/scheduler disagree at 500 m for fractional coordinates | Both use a float fast path plus exact `Fraction` arithmetic near the boundary; inclusive/strict is a single policy (`network.inclusive_range`) | `communication_graph.py`, `verification.py` |
| 4 | Verifier not independent | Hop distance <= 2 by path expansion (not the scheduler's set intersection) plus cross-check against `networkx.power(G, 2)` (`crosscheck_mismatch`); geometry is cached so repeated verification is fast | `verification.py` |
| 5 | Exact solver: no timeout, always "optimal" | Time and node budgets, clique lower bound, DSATUR upper bound, statuses `EXACT` / `BOUNDED` (feasible coloring + proven lower bound, never an unproven optimum) | `exact_solver.py`, `bounds.py`, `cli.py` |
| 6 | Config was dead code | Wired: range, inclusive policy, stage switches, exact solver settings, matrix check, golden tests, exit-status flag, output flags, input path; optimizers verify with the configured range | `cli.py`, `config.py`, optimizers |

## Other weaknesses
- Lower bounds (closed-neighborhood and clique) with a lower-bound gap when the exact solver is skipped.
- Golden self-tests (`golden.py`, 10 cases) and mutation tests; slot-matrix consistency checker.
- Required report lines ("Direct-link conflicts checked: 0 violations." etc.); class-elimination attempts are measured, not "N/A"; exact runtime printed as `0.0001 s`.
- Robustness: 400-digit integers no longer crash; identifiers with surrounding whitespace rejected; implausibly large coordinates warn.
- Scalability: verification 7-19x faster (100/200/400 random nodes: 7.3/44.7/226 s -> 1.1/5.3/11.7 s), optional `scheduling.optimization_time_budget_seconds`.
- Added `experiments/run_benchmarks.py`, `data/example_optimizer_helps.json` (DSATUR 8 -> optimized 7 = proven optimum), `data/sample_16_nodes_grid.json` (a sample, NOT the assignment file).
- README / pyproject description updated; `__pycache__` removed.

## Existing tests that had to change (they asserted the old behavior)
1. `test_class_elimination_attempts_are_not_fabricated` -> attempts are now measured (2 for the two-node case).
2. `test_exact_solver_runs_for_small_graph`: status label `OPTIMAL` -> `EXACT` (spec section 6.2).
3. `test_verification_failure_never_prints_conflict_free_success` split into a baseline-failure test (withheld) and an optimized-failure test (baseline fallback), as the spec's publication gate requires. The NOT RUN test only gained one assertion.

## Not done
- EMANE adapter (bonus; needs official EMANE schema verification).
- `assignment_16_nodes.json`: the assignment's coordinates were not provided.
- The CLI report layout is unchanged and still differs from spec section 9.1 headings (only the missing lines were added).
- Kempe moves are still never accepted on the random cases tried; the optimizer's measured benefit comes from class elimination.
- Line endings kept as CRLF.
