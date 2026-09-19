# Output conventions

Experiments 1–4 are archived. New runs of these historical definitions use
`expN_archived_` experiment identities and `expN-archived-` Batch job prefixes.

Every FHP experiment writes a timestamped, single-seed run directory beneath
the selected `--output-root`:

```text
<output-root>/<expN_experiment_name>_<YYYYMMDD_HHMMSS>_seed_<seed>/
```

Experiment packages, manifest names, and checkpoint prefixes use `expN_`.
Google Cloud Batch job IDs use `expN-` because Batch job names do not accept
underscores.

## Common run artifacts

Successful Experiments 1-4 produce:

- `run_manifest.json` — immutable experiment identity, seed, canonical OpenSpiel
  FHP game definition, selected configuration and digest, checkpoint schedule,
  reference VM, backend, and implementation provenance where applicable;
- `checkpoint_rows.json` and `checkpoint_rows.csv` — training progress, losses,
  estimator diagnostics, nodes, timings, and backend-specific execution metrics
  at each checkpoint;
- `checkpoint_manifest.json` and `checkpoint_manifest.csv` — checkpoint index,
  target time, actual safe-boundary time, node count, path, size, and SHA-256;
- `checkpoints/*.pkl` — the reloadable policies saved at 6 and 12 effective
  training hours;
- `final_policy_checkpoint.pkl` — a stable copy of the 12-hour policy;
- `summary.json` — stop reason, throughput, final progress, memory use, final
  policy digest, execution metrics, and VM-capacity assessment.

The checkpoint schedule must contain exactly two increasing targets. A valid
production run stops with `stop_reason` equal to `training_time_budget` after
the 12-hour checkpoint. Training elapsed time excludes checkpoint policy
fitting and serialization, so compare training progress using
`training_elapsed_seconds`, not only wall-clock duration.

Checkpoint paths embedded in manifests may be absolute paths from the machine
that performed training. Treat the manifest entry as provenance after moving a
run; locate the file relative to the downloaded run directory.

## Reloadable policy contract

Checkpoint files contain the fitted average-policy state and metadata required
by `fhp_escher.checkpointing.LoadedFHPPolicy`. Keep the `.pkl` file together
with its manifest and verify its SHA-256 before evaluation or transfer.

```python
from fhp_escher.checkpointing import LoadedFHPPolicy
from fhp_escher.game import load_fhp_game

game = load_fhp_game()
policy = LoadedFHPPolicy(game, "outputs/RUN/final_policy_checkpoint.pkl")
```

## Evaluation outputs

Exact tabular exploitability, initial-policy evaluation, the 10,000-node
evaluation, and periodic whole-tree evaluation are intentionally absent. The
FHP tree is too large for the exact intermediate evaluation used in Leduc.

Policy-quality comparisons should use sampled, seat-swapped head-to-head play
and record at least:

- both policy checkpoint hashes and experiment manifests;
- evaluation seed or seed range;
- number of deals;
- seat-specific and seat-averaged payoff;
- uncertainty estimates such as standard errors or confidence intervals;
- the exact canonical FHP game parameters.

Do not interpret training loss or nodes per second as policy strength. They are
training and systems diagnostics.

## Archived Experiment 2 versus Experiment 3 comparison

After downloading matched Experiment 2 and Experiment 3 runs, create the
time-aligned systems comparison with:

```bash
python -m experiments.fhp.compare_archived_exp2_exp3 \
  --sequential-run outputs/RUN_FOR_EXP2 \
  --parallel-run outputs/RUN_FOR_EXP3 \
  --output-dir outputs/archived_exp2_exp3_comparison
```

The comparison directory contains:

- `archived_exp2_exp3_comparison.json`;
- `archived_exp2_exp3_checkpoint_comparison.csv`.

These files compare nodes, trajectories, iterations, fitting loss, and
parallel-over-sequential throughput at the matched checkpoints. They do not
measure exploitability.

## Failure artifacts

If solver initialization or training raises an exception, the timestamped run
directory contains `failure.json` with the exception type, message, traceback,
phase or progress information, and peak RSS when available. Partial checkpoint
artifacts should be retained for diagnosis but must not be presented as a
completed run.

Google Cloud Batch writes these job-level files in the job output root, next to
the timestamped experiment directory:

- `batch_run.log`;
- `resource_snapshots.jsonl`;
- `batch_diagnostics.json`;
- `batch_status.json`.

Preserve all four when diagnosing memory exhaustion, SIGKILL, timeout, disk
pressure, or an installation failure.

## Storage and git boundary

`outputs/` and `cloud_outputs/` are working data and are ignored by git. They
may contain large model checkpoints, detailed logs, and partial failed runs.
Do not force-add those directories. Curated lightweight thesis artifacts, when
needed, should follow the separate guidance in `docs/THESIS_ARTIFACTS.md`.
