# Experiment 8: 48-hour parallel UCV-ESCHER with full continuation

## Frozen training contract

This is Experiment 7 with a longer training horizon, not a new architecture.
The complete learning configuration, structured lossless encoder, compact
replay, losses, optimiser settings, actor configuration and hardware are
unchanged.

- **Three seeds: 0, 1, 2**, each on a separate **n2-standard-16**.
- **Eight single-threaded traversal actors per VM**; central fitting retains
  eight Torch intra-op threads and sequential learner updates.
- **10,000 total traversals per player per iteration**, not per actor.
- **48 active training hours** per seed, stopping at the first completed outer
  iteration crossing the boundary. There is no training-node cap.
- Playable policies **every six active hours**:
  6, 12, 18, 24, 30, 36, 42 and 48.
- **One full continuation state per seed, only at 48 hours**. Intermediate
  checkpoints contain playable policies and diagnostics, not replay/optimizers.
- The same timing/memory diagnostics and throughput plots as Experiment 7.
- Frozen critic-target caching remains disabled.

Three seeds run concurrently by default: **144 active training VM-hours** in
total. Elapsed completion time is approximately the longest seed's 48 active
hours plus setup, checkpoint fitting, serialization, uploads and aggregation.
Those overheads are billable and excluded from the active clock, just as in
Experiment 7. The final completed iteration can slightly exceed 48 hours.

The training task timeout is increased from 36 to **72 wall-clock hours per
attempt**; automatic training retries are disabled. This is a safety cap, not a request
to train for 72 hours. The controller retains its seven-day cap and retry
policy. Interrupted training before the final state requires a new RUN_ID;
there is no six-hourly recovery state. Training retains the 200 GiB disk allocation.

## Start the initial 48-hour run

Run from the repository root, using its Python environment for local smoke.
PROJECT_ID, REGION, BUCKET and SA_EMAIL must be set as for Experiment 7.
The new code must be committed and pushed before cloud submission.

```bash
./gcp/run_exp8_parallel_48h.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp8-par48-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
export EXP8_TOTAL_HOURS=48
unset EXP8_SOURCE_RUN_ID
./gcp/run_exp8_parallel_48h.sh run
```

Set PYTHON=/path/to/venv/bin/python before local smoke if necessary.
Three concurrent seeds require 48 available regional N2 vCPUs. PARALLELISM
controls concurrent seeds, not the eight actors within each seed.

The controller runs cloud smoke, training and aggregation remotely. Your laptop
can disconnect after submission. Cloud smoke includes production-capacity
checkpoint allocation/saving/loading plus small eight-actor training, restart
equivalence and completed-endpoint extension. The production-capacity check
does not train or measure worst-case grouped-policy-fitting memory.

```bash
./gcp/run_exp8_parallel_48h.sh status
# Recover orchestration/aggregation or a saved final state, NOT intermediate training;
# keep the original RUN_ID, REPO_REF and
# EXP8_TOTAL_HOURS (and EXP8_SOURCE_RUN_ID for an extension):
./gcp/run_exp8_parallel_48h.sh resume
```

## What is saved at 48 hours?

For each seed, the worker's checkpoint manifest identifies both:

- A **playable policy** in checkpoints/, and final_policy_checkpoint.pkl.
- A **full training state** in training_states/ ending in _time_48h.pt.

The full state contains learned models, target models, Adam optimiser state,
average-policy/critic/calibration replay, critic target history, iteration and
node counters, cumulative active time, private replay sampler states, and all
eight actors' Python/NumPy/Torch RNG states. Transient regret rows are rebuilt
next iteration; their persistent sampler RNGs are saved. Saving only the
playable neural policy would NOT be sufficient to resume training.

Checksums, run/runtime manifests and seed identity accompany these artifacts.
The cloud worker uploads policies/metadata at every checkpoint; the full state
is uploaded only at the final endpoint and checked again on completion.
Keep the original GCS run prefix, full continuation state and metadata.
An analysis-only download intentionally omits the large training state.

## Extend a completed run later

This differs from failure recovery. Ordinary resume reuses a durable final
state or completed outputs within the existing contract; it does not add
training after completion. Before that state exists, an interrupted run is
not resumable and must use a new RUN_ID.

To continue the same three trajectories to **72 cumulative active hours**
(approximately 24 more), use the **same original REPO_REF**, and a **new RUN_ID**:

```bash
export EXP8_SOURCE_RUN_ID="YOUR_COMPLETED_EXP8_RUN_ID"
export EXP8_TOTAL_HOURS=72
export RUN_ID="exp8-to72-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
# REPO_REF must still be the original run's pinned commit, not a newer HEAD.
./gcp/run_exp8_parallel_48h.sh extend
```

The new controller downloads each source seed's small checkpoint/metadata
files and its final full training state, verifies checksums, configuration,
seed, commit and runtime compatibility, then continues its cumulative clock,
iteration numbers and RNG streams. The source folder is never modified.
New checkpoints are appended at 54, 60, 66 and 72 active hours; analysis includes
the imported 6--48-hour policies as well as the new observations.
continuation_source.json records source provenance per seed, and
analysis/continuation_sources.json collects it.

Each extension can add 6--48 hours in six-hour increments. A later extension
can use the completed extension's RUN_ID as its source. Its total-hours setting
is always **cumulative**, not an increment. Its per-attempt timeout remains 72
wall-clock hours. The importer reads the source final state as a temporary
input, excludes it from uploads and removes the private copy after use. The
source archive is untouched. The extension saves a new full state only at its
own final endpoint (for example 72 hours), alongside every playable policy.
If a required state is incompatible or corrupt, the run
fails rather than silently starting from scratch.

For local continuation, after making the source worker artifacts available:

```bash
python -m experiments.fhp.exp8_fhp_parallel_48h.run worker \
  --task-index 0 --total-hours 72 \
  --source-worker /absolute/path/to/completed/workers/task_000_parallel_structured_ucv_escher_48h_seed_0 \
  --output-root /absolute/path/to/new-extension
```

The same command can be applied to tasks 1 and 2. Local continuation requires
the same software/thread contract and enough memory; GCP is recommended.

## Download and interpret results

```bash
EXP8_BUCKET="${BUCKET%/}"
[[ "$EXP8_BUCKET" == gs://* ]] || EXP8_BUCKET="gs://$EXP8_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$EXP8_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

The analysis contains checkpoint and seed tables, provenance, per-seed timing
and memory diagnostics, nodes_by_training_time.png and
training_time_breakdown.png. Every six-hour policy is retained for subsequent
encoder-aware evaluation using the existing FHP evaluation suite.
No approximate-exploiter or head-to-head job is automatically added, and exact
FHP exploitability is not computed. Throughput and a longer run alone do not
establish strategic convergence.

The tiny local/cloud smoke tests eight checkpoint hooks, verifies that a
completed state can resume without extra work, imports it into a separate
extension and verifies additional iterations and four new checkpoints. It
also requires bitwise equality of learner/RNG state between interrupted and
uninterrupted fixed-iteration training. This is not a completed 48-hour trial.
