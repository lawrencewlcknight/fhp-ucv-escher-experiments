# FHP UCV-ESCHER Experiments

> **Archive status:** The former Experiments 1–4 are retained as runnable historical
> references. Their package names, run identities, checkpoint prefixes, and GCP
> job names contain `archived`. New research experiments should reuse useful
> components without modifying these archived definitions.

## Active Experiment 1: grouped-wide FHP baseline

The active baseline is `exp1_fhp_grouped_wide_ucv_baseline`. It ports the best
training configuration from Leduc ESCHER architecture Experiment 35 to the
canonical OpenSpiel FHP game. Seeds `0`, `1`, and `2` run concurrently on three
independent GCP Batch VMs for 24 effective training hours each.

Its continuous replay fields are stored as NumPy `float32`, matching the
precision used by network training. The raw OpenSpiel representation, generic
networks, full-width integer fields, legacy Algorithm-R replacement and
minibatch RNG behaviour remain unchanged. This storage-only correction keeps
the baseline on the same `n2-standard-8` VM class as Experiments 2 and 3.

Every seed saves a reloadable policy and exact continuation state at the first
completed outer iteration after 6, 12, 18, and 24 hours. The remote controller
requires a successful cloud smoke before production, permits one automatic
retry per worker, resumes from durable Cloud Storage state, aggregates only
after all seeds succeed, and records independent resource/OOM diagnostics.

Local smoke:

```bash
./gcp/run_exp1_grouped_wide.sh smoke-local
```

Full GCP submission after exporting the variables documented in
[`docs/GCP_BATCH_EXPERIMENTS.md`](docs/GCP_BATCH_EXPERIMENTS.md):

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp1-fhp-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp1_grouped_wide.sh run
```

See the [complete experiment protocol](experiments/fhp/exp1_fhp_grouped_wide_ucv_baseline/README.md).

## Active Experiment 2: lossless structured FHP representation

`exp2_fhp_lossless_structured_ucv` keeps Experiment 1's UCV-ESCHER estimator
and 24-hour, three-seed protocol but replaces the generic OpenSpiel input with
a versioned FHP encoder. It uses canonical suits, exact compact betting history,
poker-derived features, a non-duplicated critic state, structured card/context
networks, and float32 replay. It does not bucket hands or discard strategically
relevant information.

```bash
./gcp/run_exp2_lossless_structured.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp2-fhp-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp2_lossless_structured.sh run
```

See the [Experiment 2 protocol](experiments/fhp/exp2_fhp_lossless_structured_ucv/README.md).

## Active Experiment 3: wider structured FHP networks

`exp3_fhp_wider_lossless_structured_ucv` is a controlled network-capacity test
based on Experiment 2. It increases card/context branches from 64 to 96 units,
regret/critic/calibration trunks from two 128-unit layers to two 192-unit
layers, and the average-policy trunk from two 192-unit layers to two 256-unit
layers. All other algorithm, representation, runtime, seed, hardware, and
checkpoint settings are unchanged. Online trainable capacity increases from
310,993 to 611,729 parameters (1.97x).

```bash
./gcp/run_exp3_wider_structured.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp3-fhp-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp3_wider_structured.sh run
```

See the [Experiment 3 protocol](experiments/fhp/exp3_fhp_wider_lossless_structured_ucv/README.md).

## Shared policy evaluation

FHP checkpoints are evaluated with the validated `fhp-evaluation-suite`
implementation, with checkpoint reconstruction supplied by
`fhp_escher.evaluation_adapter`. A provenance-recorded snapshot is vendored in
this repository so pinned cloud jobs are self-contained. The sibling package
can still be installed during evaluator development with
`python -m pip install -e ../../fhp-evaluation-suite`. Experiment 2 and
Experiment 3 checkpoints must use the encoder-aware evaluators shown below.

The benchmark uses both-seat duplicate deals and corrected LooseAggressive
bands `(-300,-100)`. LBR is reported as a lower bound, not exact exploitability.

An archived experiment run's verified 6-hour and 12-hour checkpoints can be
evaluated together, with common random numbers and paired temporal intervals,
using:

```bash
python -m experiments.fhp.evaluate_checkpoints \
  --source-run cloud_outputs/JOB/outputs/cloud/JOB/RUN_DIRECTORY
```

The production defaults are 10,000 duplicate pairs per rule agent and
checkpoint, 1,000 LBR pairs with 4,096 pre-flop rollouts, and 50,000 direct
checkpoint cross-play pairs. Results are written beneath
`outputs/evaluation/EXPERIMENT_NAME/RUN_DIRECTORY/`.

The active Experiment 1 evaluator verifies and evaluates all four six-hourly
checkpoints for one seed worker:

```bash
python -m experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.evaluate_checkpoints \
  --source-run cloud_outputs/RUN_ID/workers/TASK_DIRECTORY
```

Experiments 2 and 3 use the same evaluation schedule with the encoder-aware
loader:

```bash
python -m experiments.fhp.exp2_fhp_lossless_structured_ucv.evaluate_checkpoints \
  --source-run cloud_outputs/RUN_ID/workers/TASK_DIRECTORY

python -m experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.evaluate_checkpoints \
  --source-run cloud_outputs/RUN_ID/workers/TASK_DIRECTORY
```

### Single-VM retrospective evaluation of Experiments 2 and 3

The joint evaluation is an extension of Experiments 2 and 3, not a new
training experiment. One `n2-standard-8` VM evaluates all three seeds and all
four checkpoints against the five rule agents and LBR, performs temporal and
direct cross-play, and creates seed-level tables and charts. Training-state
files are explicitly excluded from the cloud download. A small real-checkpoint
smoke test must succeed on the VM before the production evaluation begins.
The job has an 18-hour safety ceiling, while its observed production runtime
is approximately 12 hours; the VM terminates immediately when the analysis
finishes.

Set the two immutable source runs and a new append-only evaluation run ID:

```bash
export EXP2_RUN_ID="exp2-fhp-20260921-093839"
export EXP3_RUN_ID="exp3-fhp-20260921-093839"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="fhp-eval23-$(date -u '+%Y%m%d-%H%M%S')"

./gcp/run_retrospective_exp2_exp3_evaluation.sh run
```

The submitting computer is not needed after Batch accepts the job. Monitor it
with the same variables:

```bash
./gcp/run_retrospective_exp2_exp3_evaluation.sh status
```

An optional local smoke uses the already-downloaded checkpoints:

```bash
./gcp/run_retrospective_exp2_exp3_evaluation.sh smoke-local
```

Download the compact analysis after the job succeeds:

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp --recursive \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/"
```

See the [retrospective evaluation protocol](experiments/fhp/retrospective_exp2_exp3_evaluation/README.md).

### Directly comparable evaluation of Experiments 1, 2 and 3

After Experiment 1 has completed, extend the frozen Experiment 2/3 evaluation
without repeating its expensive LBR analysis. The runner verifies the source
checkpoint hashes and reference protocol, evaluates Experiment 1 identically,
adds Experiment 1 versus 2 and Experiment 1 versus 3 cross-play, and produces
unified tables and charts for all three experiments.

```bash
export EXP1_RUN_ID="exp1-fhp-20260923-233627"
export EXP2_RUN_ID="exp2-fhp-20260921-093839"
export EXP3_RUN_ID="exp3-fhp-20260921-093839"
export EXP23_EVAL_RUN_ID="fhp-eval23-20260924-120059"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="fhp-eval123-$(date -u '+%Y%m%d-%H%M%S')"

./gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh run
```

The job uses one `n2-standard-8` VM and runs independently after Batch accepts
it. Monitor it with the same variables:

```bash
./gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh status
```

Download the compact completed analysis with:

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp --recursive \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/"
```

See the [three-experiment evaluation protocol](experiments/fhp/retrospective_exp1_exp2_exp3_evaluation/README.md).

After archived Experiments 1--4 have been evaluated with the same configuration,
build the common-deal 6-hour and 12-hour cross-play matrices with:

```bash
python -m experiments.fhp.compare_evaluated_experiments
```

The production default is 50,000 duplicate deal pairs for each experiment
pair and checkpoint. Consolidated results are written beneath
`outputs/evaluation/cross_experiment_comparison/seed_0/`.

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

## Archived: exp1_archived_fhp_ucv_escher_baseline

Archived Experiment 1 transfers the selected UCV-ESCHER architecture and optimisation
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
`exp1_archived_fhp_ucv_escher_baseline`). Google Cloud job IDs use the corresponding
hyphenated `expN-` prefix because underscores are not valid in Batch job names.

Local orchestration smoke test:

```bash
python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run --smoke
```

Full run:

```bash
python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run
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

## Archived: exp2_archived_fhp_ucv_escher_sequential and exp3_archived_fhp_ucv_escher_ray_parallel

Archived Experiments 2 and 3 are a matched, single-seed comparison of sequential and
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
python -m experiments.fhp.exp2_archived_ucv_escher_sequential.run --smoke
python -m experiments.fhp.exp3_archived_ucv_escher_parallel.run --smoke
```

After both production outputs have been downloaded, validate that the arms are
matched and generate their time-aligned throughput comparison:

```bash
python -m experiments.fhp.compare_archived_exp2_exp3 \
  --sequential-run outputs/RUN_FOR_EXP2 \
  --parallel-run outputs/RUN_FOR_EXP3 \
  --output-dir outputs/archived_exp2_exp3_comparison
```

The comparison reports nodes, trajectories, outer iterations, policy-fit loss,
and parallel-over-sequential node throughput at 6 and 12 hours. These are
systems and training-progress measurements, not exact exploitability. Policy
quality comparison should use sampled evaluation of the saved checkpoints.

## Archived: exp4_archived_fhp_ucv_escher_cpu_optimized

Archived Experiment 4 is the high-throughput CPU arm. It retains Experiment 1's UCV
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
python -m experiments.fhp.exp4_archived_ucv_escher_cpu_optimized.run --smoke
```

The run records compact replay allocation, worker snapshot reload time, actor
time, merge time, learner time, payload size, dispatch count, and thread/worker
settings in its checkpoint rows and final summary.

## Google Cloud Batch

The commands below remain available to reproduce the archived experiments. Use
the archived job-name prefixes exactly as shown so historical reruns cannot be
mistaken for new experiments.

Before submitting any job, configure the Google Cloud project, region, results
bucket, and Batch service account:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="YOUR_BATCH_REGION"
export BUCKET="gs://YOUR_RESULTS_BUCKET"
export SA_EMAIL="YOUR_BATCH_SERVICE_ACCOUNT"
```

Every smoke job below uses `n2-standard-4`, a two-hour Batch limit, 4 vCPUs,
16,000 MiB of requested memory, and a 100 GiB `pd-balanced` boot disk. The
`--smoke` flag retains the production orchestration and checkpoint/reload path
while reducing the replay sizes, traversal counts, learner steps, and checkpoint
thresholds. C4 full runs explicitly use `hyperdisk-balanced`; the submission
helper rejects C4/Persistent Disk combinations before contacting Batch.

### Archived Experiment 1: exp1_archived_fhp_ucv_escher_baseline

#### GCP Batch smoke test

```bash
JOB_NAME="exp1-archived-fhp-ucv-baseline-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

#### GCP Batch full run

The Experiment 1 reference machine is `n2-standard-8` with 8 vCPUs and 32 GB
RAM. Its 14-hour Batch limit covers 12 effective training hours plus setup,
checkpoint fitting, teardown, and upload.

```bash
JOB_NAME="exp1-archived-fhp-ucv-baseline-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 50400 8000 32000 100 pd-balanced
```

### Archived Experiment 2: exp2_archived_fhp_ucv_escher_sequential

#### GCP Batch smoke test

```bash
JOB_NAME="exp2-archived-fhp-ucv-sequential-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_archived_ucv_escher_sequential.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

#### GCP Batch full run

The Experiment 2 full run uses `c4-standard-32`, 32 vCPUs, 120,000 MiB of
requested memory, a 200 GiB `hyperdisk-balanced` boot disk, and a 14-hour Batch
limit.

```bash
JOB_NAME="exp2-archived-fhp-ucv-sequential-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_archived_ucv_escher_sequential.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

### Archived Experiment 3: exp3_archived_fhp_ucv_escher_ray_parallel

#### GCP Batch smoke test

```bash
JOB_NAME="exp3-archived-fhp-ucv-parallel-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_archived_ucv_escher_parallel.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

#### GCP Batch full run

The Experiment 3 full run uses the same production allocation as Experiment 2:
`c4-standard-32`, 32 vCPUs, 120,000 MiB of requested memory, a 200 GiB
`hyperdisk-balanced` boot disk, and a 14-hour Batch limit.

```bash
JOB_NAME="exp3-archived-fhp-ucv-parallel-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_archived_ucv_escher_parallel.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

### Archived Experiment 4: exp4_archived_fhp_ucv_escher_cpu_optimized

#### GCP Batch smoke test

```bash
JOB_NAME="exp4-archived-fhp-ucv-cpu-optimized-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_archived_ucv_escher_cpu_optimized.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

#### GCP Batch full run

Experiment 4 uses `c4-standard-32`, 32 vCPUs, 120,000 MiB of requested memory,
an 8 GiB Ray object store within that allocation, a 200 GiB
`hyperdisk-balanced` boot disk, and a 14-hour Batch limit.

```bash
JOB_NAME="exp4-archived-fhp-ucv-cpu-optimized-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_archived_ucv_escher_cpu_optimized.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
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
