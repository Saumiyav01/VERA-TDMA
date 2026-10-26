# VERA-TDMA

VERA-TDMA is a centralized TDMA schedule planner for static radio topologies.
It builds an undirected communication graph (inclusive Euclidean range, default
500 m), the Distance-2 conflict graph, a DSATUR schedule, class-elimination and
Kempe-chain refinement, an independent verifier, an exact reference solver and
proven lower bounds. EMANE integration is not implemented.

## Development

Install the package and run the tests:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the benchmark summary (writes `experiments/results/`):

```powershell
python experiments/run_benchmarks.py --seeds 5 --sizes 16 30 60
```

## Configuration

`config.yaml` documents the default typed configuration. Configuration files
may omit fields to use defaults. Unknown fields and invalid values are rejected
when loading the configuration. Independent verification and conflict checking
are mandatory and cannot be disabled. Pass a file with `--config`; without it
the built-in defaults (identical to `config.yaml`) apply.

The CLI honours: `network.communication_range_m` and `network.inclusive_range`
(used by the graph, the optimizers and the verifier alike), the
`scheduling.enable_*` switches and `scheduling.optimization_time_budget_seconds`,
`exact_solver.enabled` / `timeout_seconds` / `max_nodes`,
`validation.check_matrix_consistency`, `validation.run_golden_tests`,
`validation.fail_on_verification_error` (exit status only; a schedule that fails
verification is never published), the `output.include_*` switches, and
`paths.input_nodes` (used when `--input` is omitted). `paths.output_dir` and
`paths.report_file` are reserved and currently unused.

## Coordinate inputs

JSON files contain an object mapping each node identifier to an `[x, y]` pair,
for example:

```json
{
  "Node2": [10.0, 20.0],
  "Node10": [30.0, 40.0]
}
```

CSV files have the exact header `node_id,x_coordinate,y_coordinate`. In both
formats, coordinates are finite numbers measured in meters. Node identifiers
must be non-empty and unique. Distinct nodes may share a coordinate; loading
them emits a `CoLocatedNodesWarning`. Loaded nodes are returned in
case-sensitive natural order, so `Node2` precedes `Node10`.

## Production CLI

Run the production pipeline with:

```powershell
python -m vera_tdma --input data/nodes.json
python -m vera_tdma --input data/nodes.json --output outputs/result.json --exact
```

CLI JSON input maps each node identifier either to an `[x, y]` pair or to an
object containing exactly numeric `x` and `y` coordinates (both forms are
accepted):

```json
{
  "Node_01": [0.0, 0.0],
  "Node_02": {"x": 300.0, "y": 0.0}
}
```

Node identifiers must be non-empty, unique and free of leading or trailing
whitespace. Coordinates beyond 10 000 km produce a warning.

Range decisions near the 500 m boundary use exact rational arithmetic, so the
scheduler and the verifier always agree, including for fractional coordinates.

### Publication gate

The optimized schedule is published only if the independent verifier passes it.
If it fails (or the optimizer raises), the verified DSATUR baseline is published
instead and the rejection is reported. If the baseline itself fails or cannot be
verified, nothing is published: the report says `SCHEDULE WITHHELD`, `--output`
is not written, and the exit status is non-zero.

### Exact reference solver and bounds

The exact solver runs by default for up to `exact_solver.max_nodes` nodes (use
`--no-exact` to skip it, `--exact-max-nodes N` to change the limit). Status
`EXACT` means the optimum is proven. If `exact_solver.timeout_seconds` expires the
status is `BOUNDED` and the report gives the proven interval
`lower bound <= K* <= best feasible`; an optimum is never claimed. Above the node
limit the status is `BOUNDED/SKIPPED`. Every report also shows a proven lower
bound (closed-neighborhood and clique bounds) and the resulting lower-bound gap,
which is explicitly not an exact optimality gap.

### Self-tests

Unless `validation.run_golden_tests` is false, the CLI first runs ten built-in
golden topologies (`vera_tdma.golden`) and halts with exit code 3 if any fails.
The verifier cross-checks its own conflict set against `networkx.power(G, 2)`.

### Examples

`data/example_optimizer_helps.json` is a 14-node topology where DSATUR needs 8
slots and class elimination reaches the proven optimum of 7.
`data/sample_16_nodes_grid.json` is a 4x4 grid at 300 m spacing (a sample, not
the assignment's own coordinates). Kempe refinement rarely changes the result
on dense Distance-2 conflict graphs; the report states how many Kempe moves were
attempted and accepted.

Output includes canonical SHA-256 input and schedule hashes, network and schedule
metrics, verification details, and the slot-by-node matrix. The conflict-free
success lines are printed only after the independent verifier returns valid.
