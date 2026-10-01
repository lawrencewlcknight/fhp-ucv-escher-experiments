# Experiment 7: eight-worker structured UCV-ESCHER

## Question and controls

Does synchronous parallel traversal collection improve training throughput on
the same machine as Experiment 6? The learning configuration is an exact deep
copy of Experiment 2 (also used in Experiment 6), with strict validation.
Neither cached critic targets nor a new network/loss/replay configuration is
introduced. Use the same source commit for Experiments 6 and 7.

- Three independent seeds: **0, 1, 2**.
- One **n2-standard-16 (16 vCPU, 64 GiB)** per seed; three VMs concurrently by default.
- **24 active training hours**, stopping after the completed outer iteration
  crossing the boundary; no node cap.
- Playable grouped-policy checkpoints at **6, 12, 18, 24 hours**; no full training states.
- **Eight single-threaded Ray traversal actors per VM**, not eight VMs per seed.
- **10,000 total traversals per player per iteration**, partitioned across actors.
- Bounded collection chunks: 1,200 total trajectories, at most 150 per actor
  per request; final chunks may be smaller. A 4 GiB Ray object store is included
  within the VM's memory allocation.
- Eight Torch intra-op threads on the central learner, matching Experiment 6.
  Regret, calibration and critic fits retain their sequential update order.
- The same two critics, fixed beta=1, disabled predictor, four-fit critic-target
  averaging, structured canonical features, compact replay and grouped reset
  cross-entropy policy fit as Experiment 2.

The driver holds authoritative replay, models and optimisers. Actors hold
inference models and bounded scratch replay, not full copies of the persistent
reservoirs. Each traverser phase uses one frozen inference snapshot; results are
merged in a deterministic actor order before fitting. There are no asynchronous
or stale-gradient updates. Every request checks its returned traversal count.

Each actor has a reproducible independent random stream. Consequently the
parallel run preserves the learning rules and total budgets, but is not expected
to produce the same sampled trajectories or bitwise networks as a serial run.
Do not interpret matching seed labels as identical sampled training data.

## Run and recover

Use `gcp/run_exp7_parallel_structured_n2_standard16.sh` from the repo root.
The usual PROJECT_ID, REGION, BUCKET and SA_EMAIL must already be configured;
REPO_REF must identify a **pushed commit containing this experiment**.

```bash
./gcp/run_exp7_parallel_structured_n2_standard16.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp7-par8-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp7_parallel_structured_n2_standard16.sh run
```

Set `PYTHON=/path/to/venv/bin/python` for local smoke if necessary. Ray and the
repository dependencies must be installed in that environment. The local smoke
really starts eight actors but uses small buffers, a 128 MiB object store, and
one learner thread. It is not a throughput benchmark.

The remote controller first runs a cloud smoke on n2-standard-16, then the seed
array, then aggregation. The laptop can disconnect after submission. The
controller and aggregator retain Experiment 6's smaller auxiliary VMs.
`PARALLELISM=3` means three concurrent seeds, requiring 48 regional N2 vCPUs;
changing it to 1 or 2 does not change the eight actors within each seed.

```bash
./gcp/run_exp7_parallel_structured_n2_standard16.sh status
# Orchestration/aggregation recovery; interrupted training needs a new RUN_ID:
./gcp/run_exp7_parallel_structured_n2_standard16.sh resume
```

Do not run two controllers against the same output prefix. No full training
states are retained. Completed workers are hash-checked and reused; an
interrupted worker fails with instructions to start a new RUN_ID, rather than
silently mixing a restarted trajectory with earlier checkpoints. Runtime,
configuration and source identities must match for completed-output reuse.
Actors are shut down on success and failure.

## Smoke and capacity checks

Both local and cloud smoke:

1. Train one tiny iteration, save a real disk-backed continuation, then compare
   a restarted second iteration with uninterrupted training. Models, optimiser
   states, critic history, replay, random states and node counts must match
   bitwise. Wall-clock measurements are intentionally excluded from equality.
2. Exercise all eight actors, multiple chunks and cached snapshot reuse.
3. Complete all four time-checkpoint hooks, reload a playable policy, reuse
   completed results without extra training, and aggregate the outputs.

Synthetic restart states are private temporary test inputs; only their report
is retained, not the state itself.

Cloud smoke also runs `capacity-preflight` **before** the tiny training test.
It allocates the production replay and actors, touches allocated replay pages,
and saves/loads/restores a full synthetic continuation checkpoint using the
production format. Temporary synthetic checkpoint data are removed locally;
only its JSON report is uploaded. Failure prevents production submission.
This tests allocation/serialization memory, not the worst-case memory of a full
grouped policy fit or 24 hours of training. Local smoke deliberately omits this
large allocation. Runtime resource monitoring still records memory/OOM evidence.

## Time, cost and diagnostics

The clock definition matches Experiment 6: checkpoint policy fitting,
evaluation, persistence and uploads are excluded from active training time.
Setup and Ray startup are also outside it. These operations remain billable.
Training tasks retain the 36-hour wall-time cap per attempt, with **no automatic
training retries** because no intermediate training state exists.
The controller retains the existing seven-day safety cap and retry policy.

Parallel collection does not imply eightfold end-to-end speedup. Central
fitting, serialization, transfers and synchronisation can dominate. Node rate
also depends on sampled game paths; report completed-iteration rate alongside
nodes per active second and actual VM wall time.

Each checkpoint and endpoint records collection/wait time, snapshot construction,
merge time, regret fitting, critic/calibration fitting, worker time/imbalance,
dispatch counts and payload size. Merge time is already included in collection;
summed actor times are not extra driver elapsed time. The stacked timing plot
uses only non-overlapping driver phases and omits uninstrumented overhead.
Memory reports distinguish driver peak RSS, the sum of actor peak RSS values
(not a simultaneous VM peak), and cgroup current/peak memory where available.
The cloud resource monitor supplies whole-job resource time series.

## Analysis and comparison

The small `analysis/` directory contains the inherited seed/checkpoint tables,
provenance manifest and completion marker, plus:

- `throughput_summary.json`: nodes, iterations, throughput, per-seed execution
  diagnostics and mean/SE summaries.
- `nodes_by_training_time.png`: individual seeds and mean with one SE.
- `training_time_breakdown.png`: non-overlapping instrumented driver phases.

```bash
EXP7_BUCKET="${BUCKET%/}"
[[ "$EXP7_BUCKET" == gs://* ]] || EXP7_BUCKET="gs://$EXP7_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$EXP7_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

After downloading both small analysis folders, create a comparison locally
(EXP6_RUN_ID and EXP7_RUN_ID should identify your completed runs):

```bash
python -m experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.run compare \
  --baseline-root "cloud_outputs/$EXP6_RUN_ID" \
  --parallel-root "cloud_outputs/$EXP7_RUN_ID" \
  --output-root "cloud_outputs/$EXP7_RUN_ID/comparison"
```

This creates a combined node/time chart and descriptive per-seed node- and
iteration-throughput ratios. It rejects mismatched learning/VM contracts,
incomplete seed sets and smoke outputs. Confirm source/runtime manifests before
making a same-commit causal efficiency claim; the small analysis tables alone
do not prove identical installed dependencies.

No policy-quality evaluator is launched automatically. The saved structured
policies use the existing encoder-aware evaluation adapter. Use the same
rule-agent, approximate-exploiter and head-to-head protocol as Experiment 6 to
check quality separately; throughput alone does not establish stronger play and
none of these evaluators is exact FHP exploitability.
