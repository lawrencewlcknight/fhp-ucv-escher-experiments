# FHP ESCHER Architecture Experiments

This repository studies Unbiased Control-Variate ESCHER (UCV-ESCHER) in
two-player flop hold'em poker (FHP). It uses OpenSpiel for the complete game
implementation and preserves the selected UCV-ESCHER training configuration.

## Documentation

- [`docs/GCP_BATCH_EXPERIMENTS.md`](docs/GCP_BATCH_EXPERIMENTS.md) — complete
  Google Cloud setup, smoke/full submissions, monitoring, retrieval, and failure
  diagnosis;
- [`docs/OUTPUT_CONVENTIONS.md`](docs/OUTPUT_CONVENTIONS.md) — run directory,
  checkpoint, comparison, evaluation, and diagnostic artifact contracts;
- [`docs/THESIS_ARTIFACTS.md`](docs/THESIS_ARTIFACTS.md) — safe curation guidance
  for future thesis-facing FHP results.

## Canonical game

`fhp_escher.game.load_fhp_game()` loads OpenSpiel `universal_poker` with the
VR-Deep FHP contract:

| Setting | Value |
|---|---|
| Players | 2 |
| Betting | Fixed limit |
| Deck | 13 ranks × 4 suits |
| Private cards | 2 per player |
| Public cards | 3-card flop |
| Betting rounds | 2 |
| Blinds | 50 / 100 |
| Raise sizes | 100 / 100 |
| First player | Player 1 / Player 2 |
| Maximum raises | 3 / 3 |

The exact parameter mapping is tested so a silent game-definition change fails
CI. Provenance is recorded in [`fhp_escher/game.py`](fhp_escher/game.py).

## exp1_fhp_ucv_escher_baseline

Experiment 1 transfers the selected UCV-ESCHER architecture and optimisation
settings to FHP and trains by elapsed training time rather than a node budget.
It saves exactly two reloadable average-policy checkpoints: the first at the
first safe trajectory boundary after 6 hours, and the final checkpoint after
12 hours. Policy fitting and checkpoint-writing time is excluded from the
training timer. Training stops immediately after the 12-hour checkpoint.

Initial-policy, early node-threshold, and periodic outer-iteration evaluations
are disabled. Exact tabular exploitability is intentionally not attempted
because enumerating the FHP tree is impractical. The saved policies are designed
for sampled head-to-head evaluation.

Experiment packages, canonical experiment names, output directories, and
numbered checkpoint filenames use an `expN_` prefix (for example,
`exp1_fhp_ucv_escher_baseline`). Google Cloud job IDs use the corresponding
hyphenated `expN-` prefix because underscores are not valid in Batch job names.

Local orchestration smoke test:

```bash
python -m experiments.fhp.exp1_ucv_escher_baseline.run --smoke
```

Full run:

```bash
python -m experiments.fhp.exp1_ucv_escher_baseline.run
```

Outputs include:

- `run_manifest.json`: immutable game, VM, budget, and training configuration;
- `checkpoint_rows.csv`: per-checkpoint losses, estimator diagnostics, nodes,
  and elapsed time;
- `checkpoint_manifest.json`: checkpoint hashes and paths;
- `final_policy_checkpoint.pkl`: stable checkpoint name for later evaluation;
- `summary.json`: throughput, memory use, stop reason, and VM capacity finding.

Reload the final policy:

```python
from fhp_escher.checkpointing import LoadedFHPPolicy
from fhp_escher.game import load_fhp_game

game = load_fhp_game()
policy = LoadedFHPPolicy(game, "outputs/RUN/final_policy_checkpoint.pkl")
```

## exp2_fhp_ucv_escher_sequential and exp3_fhp_ucv_escher_ray_parallel

Experiments 2 and 3 are a matched, single-seed comparison of sequential and
synchronous Ray-parallel UCV-ESCHER. Both use seed 0, the exact Experiment 1
algorithm configuration, the canonical OpenSpiel FHP game, and reloadable
checkpoints at 6 and 12 effective training hours. Both stop after the 12-hour
checkpoint. The only intended difference is the execution backend.

Experiment 3 uses 12 persistent traversal actors, one authoritative learner,
deterministic driver-side replay merging, and concurrent training of the three
independent Q folds plus the calibration learner. The 10,000-trajectory phase is
split into synchronized 1,200-trajectory dispatches so time checkpoints are
observed at bounded safe merge boundaries without introducing asynchronous
gradients. The implementation is ported from the Leduc repository; see
[`unbiased_escher/PARALLEL_UPSTREAM.md`](unbiased_escher/PARALLEL_UPSTREAM.md).

Local smoke tests:

```bash
python -m experiments.fhp.exp2_ucv_escher_sequential.run --smoke
python -m experiments.fhp.exp3_ucv_escher_parallel.run --smoke
```

After both production outputs have been downloaded, validate that the arms are
matched and generate their time-aligned throughput comparison:

```bash
python -m experiments.fhp.compare_exp2_exp3 \
  --sequential-run outputs/RUN_FOR_EXP2 \
  --parallel-run outputs/RUN_FOR_EXP3 \
  --output-dir outputs/exp2_exp3_comparison
```

The comparison reports nodes, trajectories, outer iterations, policy-fit loss,
and parallel-over-sequential node throughput at 6 and 12 hours. These are
systems and training-progress measurements, not exact exploitability. Policy
quality comparison should use sampled evaluation of the saved checkpoints.

## exp4_fhp_ucv_escher_cpu_optimized

Experiment 4 is the high-throughput CPU arm. It retains Experiment 1's UCV
estimator, network architecture, optimiser and learner-step counts, 10,000
traversals per player, synchronous player-update ordering, frozen inference
targets, seed, and 6-hour/12-hour checkpoint schedule.

The execution backend uses 28 traversal actors on the 32-vCPU reference VM,
two 5,000-trajectory dispatches per player, actor-side snapshot caching, compact
typed driver replay, vectorised exact Algorithm-R reservoir ingestion,
vectorised without-replacement minibatch selection with independent per-buffer
RNGs, and four concurrent learners with an explicit four-thread intra-op budget.
These are systems optimisations; they do not alter the UCV estimator or multiply
the configured traversal or gradient-update budgets.

Local smoke test:

```bash
python -m experiments.fhp.exp4_ucv_escher_cpu_optimized.run --smoke
```

The run records compact replay allocation, worker snapshot reload time, actor
time, merge time, learner time, payload size, dispatch count, and thread/worker
settings in its checkpoint rows and final summary.

## Google Cloud Batch

Before submitting any job, configure the Google Cloud project, region, results
bucket, and Batch service account:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="YOUR_BATCH_REGION"
export BUCKET="gs://YOUR_RESULTS_BUCKET"
export SA_EMAIL="YOUR_BATCH_SERVICE_ACCOUNT"
```

Every smoke job below uses `n2-standard-4`, a two-hour Batch limit, 4 vCPUs,
16,000 MiB of requested memory, and a 100 GiB boot disk. The `--smoke` flag
retains the production orchestration and checkpoint/reload path while reducing
the replay sizes, traversal counts, learner steps, and checkpoint thresholds.

### Experiment 1: exp1_fhp_ucv_escher_baseline

#### GCP Batch smoke test

```bash
JOB_NAME="exp1-fhp-ucv-baseline-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_ucv_escher_baseline.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100
```

#### GCP Batch full run

The Experiment 1 reference machine is `n2-standard-8` with 8 vCPUs and 32 GB
RAM. Its 14-hour Batch limit covers 12 effective training hours plus setup,
checkpoint fitting, teardown, and upload.

```bash
JOB_NAME="exp1-fhp-ucv-baseline-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_ucv_escher_baseline.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 50400 8000 32000 100
```

### Experiment 2: exp2_fhp_ucv_escher_sequential

#### GCP Batch smoke test

```bash
JOB_NAME="exp2-fhp-ucv-sequential-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_ucv_escher_sequential.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100
```

#### GCP Batch full run

The Experiment 2 full run uses `c4-standard-32`, 32 vCPUs, 120,000 MiB of
requested memory, a 200 GiB boot disk, and a 14-hour Batch limit.

```bash
JOB_NAME="exp2-fhp-ucv-sequential-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_ucv_escher_sequential.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200
```

### Experiment 3: exp3_fhp_ucv_escher_ray_parallel

#### GCP Batch smoke test

```bash
JOB_NAME="exp3-fhp-ucv-parallel-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_ucv_escher_parallel.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100
```

#### GCP Batch full run

The Experiment 3 full run uses the same production allocation as Experiment 2:
`c4-standard-32`, 32 vCPUs, 120,000 MiB of requested memory, a 200 GiB boot
disk, and a 14-hour Batch limit.

```bash
JOB_NAME="exp3-fhp-ucv-parallel-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_ucv_escher_parallel.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200
```

### Experiment 4: exp4_fhp_ucv_escher_cpu_optimized

#### GCP Batch smoke test

```bash
JOB_NAME="exp4-fhp-ucv-cpu-optimized-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_ucv_escher_cpu_optimized.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100
```

#### GCP Batch full run

Experiment 4 uses `c4-standard-32`, 32 vCPUs, 120,000 MiB of requested memory,
an 8 GiB Ray object store within that allocation, a 200 GiB boot disk, and a
14-hour Batch limit.

```bash
JOB_NAME="exp4-fhp-ucv-cpu-optimized-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_ucv_escher_cpu_optimized.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200
```

The cleanup trap uploads outputs from smoke and full runs even after a failed
job. An independent monitor writes `resource_snapshots.jsonl` every 15 seconds,
including cgroup memory current/peak/limit values, OOM counters, system memory,
disk use, load, cgroup CPU counters, per-process CPU ticks, and the largest
processes. Compact resource heartbeats also reach Cloud Logging every minute.
On cleanup, `batch_diagnostics.json` preserves
the detailed evidence and `batch_status.json` classifies confirmed cgroup OOM,
reported allocator errors, probable OOM/SIGKILL, timeout/termination, Python
exceptions, and other nonzero exits. The run log also attempts to capture
kernel OOM messages when the VM permits it.

## Verification

```bash
python -m pytest
python -m ruff check .
```

See [`vr_deep_cfr/UPSTREAM.md`](vr_deep_cfr/UPSTREAM.md) and
[`unbiased_escher/PARALLEL_UPSTREAM.md`](unbiased_escher/PARALLEL_UPSTREAM.md)
for algorithm-code provenance and redistribution cautions.
