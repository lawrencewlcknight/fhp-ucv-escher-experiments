# FHP UCV-ESCHER Experiments

> **Archive status:** The former Experiments 1–4 are retained as runnable historical
> references. Their package names, run identities, checkpoint prefixes, and GCP
> job names contain `archived`. New research experiments should reuse useful
> components without modifying these archived definitions.

## Parallel implementation review

The [parallel efficiency review](docs/PARALLEL_EFFICIENCY_REVIEW.md) documents
tested replay, preprocessing and snapshot-cache optimisations, plus parallel
adapters for the selected grouped-policy configurations used by active
Experiments 1–3. It includes equivalence tests and reproducible local
microbenchmarks. Existing experiment launchers remain unchanged; the adapters
require actor-aware resume integration before a new long-running GCP study.

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

## Active Experiment 4: frozen average-policy fitting audit

This is a **new** Experiment 4, unrelated to the archived CPU-optimisation study.
It reuses the completed Experiment 2 **24-hour reservoirs from seeds 0, 1, 2**;
there is no new regret learning. Four reset policy fits cross uniform versus
replay-mass-proportional group sampling with **20,000 versus 60,000 updates**.
Network, loss objective, loss scale, learning rate and paired initialisations
remain fixed. Separate held-out-group diagnostic fits test generalisation;
the four playable policies use the complete reservoir.

The remote controller runs a real-source cloud smoke, three parallel
`n2-standard-8` workers (one per source seed), then aggregation. Evaluation
includes restricted LBR, published rule agents and seat-swapped head-to-head
play against the archived policy and the new 20,000-update control. It does
**not** calculate exact exploitability. No full training states are uploaded
again or needed on your laptop. Only the selected 24-hour states are fetched
on the VMs, approximately 4.5 GB per seed.

Run from this repository root, with the usual `PROJECT_ID`, `REGION`, `BUCKET`
and `SA_EMAIL` already set. **The new code must first be committed and pushed.**
Use a Python environment with this repository's dependencies for local smoke:

```bash
./gcp/run_exp4_average_policy_audit.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export EXP2_RUN_ID="exp2-fhp-20260921-093839"
export RUN_ID="exp4-audit-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp4_average_policy_audit.sh run
```

The laptop may disconnect after submission. `run` always gates production on
a successful **real-reservoir** cloud smoke; the local smoke uses synthetic
targets on real FHP encodings. Check progress or resume a failed audit with
the **same** source/run ID and pinned code revision:

```bash
./gcp/run_exp4_average_policy_audit.sh status
./gcp/run_exp4_average_policy_audit.sh resume
```

Resume reuses completed fitting endpoints and cached evaluation tasks, and
does not retrain Experiment 2. There are no automatic worker retries. Each
worker has a 24-hour safety cap; this is a timeout, **not** a training budget.
See the [full audit protocol](experiments/fhp/exp4_fhp_average_policy_audit/README.md)
for resource limits, interpretation and service-account requirements.

Download **all analytical outputs only** after aggregation succeeds:

```bash
AUDIT_BUCKET="${BUCKET%/}"
[[ "$AUDIT_BUCKET" == gs://* ]] || AUDIT_BUCKET="gs://$AUDIT_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$AUDIT_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Active Experiment 5: frozen critic-target cache efficiency

This paired component benchmark reuses FHP Experiment 2's final 24-hour states, seeds **0, 1, 2**.
It compares the existing critic fit, which recomputes TD targets in each
minibatch, with an opt-in fit that computes each replay row's frozen target
once per fitting block. It does **not** retrain the whole algorithm or change
the average-policy configuration.

Each of three separate `n2-standard-8` VMs handles one source seed. The two
arms run sequentially on the same VM, from identical model/Adam states and
minibatch RNG states: two critic folds, 10,000 updates per fold, batch 2,048,
three timing repeats. Timings include cache construction. Targets, final
parameters, optimizer states, target averaging and RNG equality are checked;
a speedup alone is not a correctness result. Caching remains **off by default**.

Run from this repository root with `PROJECT_ID`, `REGION`, `BUCKET` and
`SA_EMAIL` set. Commit and push this implementation before cloud submission.
Use this repo's Python environment for the optional local smoke
(or set `PYTHON=/path/to/venv/bin/python`):

```bash
./gcp/run_frozen_critic_target_efficiency.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export SOURCE_RUN_ID="exp2-fhp-20260921-093839"
export RUN_ID="exp5-cache-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_frozen_critic_target_efficiency.sh run
```

Always reset the source/run variables when switching between repos.
Use the bucket containing this repository's source run; it is also the output
destination. Three simultaneous workers require 24 available N2 vCPUs.
The remote controller runs a mandatory real-source smoke, the three paired
workers, and aggregation. Your laptop may disconnect after submission.

```bash
./gcp/run_frozen_critic_target_efficiency.sh status
# Only after a failure, with the same RUN_ID, SOURCE_RUN_ID and REPO_REF:
./gcp/run_frozen_critic_target_efficiency.sh resume
```

After aggregation, download the complete small analysis folder (not the
multi-GB source training states):

```bash
CACHE_BUCKET="${BUCKET%/}"
[[ "$CACHE_BUCKET" == gs://* ]] || CACHE_BUCKET="gs://$CACHE_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$CACHE_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Read `analysis/summary.json`'s `all_equivalence_checks_passed` as well as its
speedup: a successfully completed benchmark can report a failed numerical
comparison. See the [full protocol](experiments/fhp/exp5_fhp_critic_target_cache/README.md)
for correctness gates, interpretation, memory and timeout limits.

## Output retention for active Experiments 4–8

Keep all analysis/evaluation results, diagnostic logs, reproducibility metadata
and playable policy checkpoints. Experiments 4–7 produce **no retained training
states or replay archives**. Experiment 5 already produces only diagnostics.
Experiment 8 retains **one final full training state per seed**, at 48 active
hours (or the final requested endpoint of a later extension), never intermediate
full states. Temporary smoke-test states and imported source copies are not
published. Existing Experiment 1–3 archives remain untouched; Experiments 4/5
still read Experiment 2's final reservoirs as inputs.

Playable policies cannot restart training. An interrupted Experiment 6/7 run,
or Experiment 8 before its final state is durable, requires a **new RUN_ID**;
the worker refuses to silently restart in the old output directory. Completed
workers can be reused for aggregation. Automatic training-task retries are
disabled. Experiment 4 can reuse completed fits/evaluation shards, but repeats
an interrupted fitting path from its matched reset initialization.

## Active Experiment 6: Experiment 2 on n2-standard-16

This is a fresh **24-active-hour, three-seed** replication of Experiment 2 on
a larger VM. Seeds **0, 1, 2** run independently and, by default, concurrently
on three **n2-standard-16 (16 vCPU, 64 GiB)** VMs. The learner configuration is
identical to Experiment 2: sequential collection, eight fitting threads,
the same structured networks, replay capacities and update budgets.
There are **no Ray traversal workers** and frozen-target caching is disabled.
`PARALLELISM` controls concurrent seeds, not collection workers.

The separate experiment name is `exp6_fhp_structured_n2_standard16`; policies,
manifests and GCS prefixes are separate from Experiment 2.
Reloadable policies (without full training states) are saved at the first completed
iterations crossing **6, 12, 18 and 24 active hours**. As before, policy fitting,
checkpoint persistence and upload time are excluded from the training clock,
so billed/runtime hours exceed 24. There is no training-node cap.

A larger VM does not guarantee higher throughput: collection remains serial
and fitting still uses eight threads. This run supplies a same-machine
sequential baseline for a future parallel-collection experiment.
Historical Experiment 2 results may also reflect intervening runtime
optimisations; do not attribute every difference solely to VM size.

Run from this repo root with the usual `PROJECT_ID`, `REGION`, `BUCKET` and
`SA_EMAIL` already set. The new code must be committed and pushed before GCP
submission. Use the repository Python environment for local smoke; set
`PYTHON=/path/to/venv/bin/python` if needed.

```bash
./gcp/run_exp6_structured_n2_standard16.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp6-vm16-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp6_structured_n2_standard16.sh run
```

The cloud controller first runs the smoke on n2-standard-16, then training,
then aggregation. The laptop can disconnect once submission succeeds.
Three simultaneous seeds require **48 available regional N2 vCPUs**;
`PARALLELISM=1` or `2` lowers concurrency without changing individual runs.

```bash
./gcp/run_exp6_structured_n2_standard16.sh status
# Orchestration/aggregation recovery only; interrupted training needs a new RUN_ID:
./gcp/run_exp6_structured_n2_standard16.sh resume
```

Download the small analytical outputs after aggregation succeeds:

```bash
EXP6_BUCKET="${BUCKET%/}"
[[ "$EXP6_BUCKET" == gs://* ]] || EXP6_BUCKET="gs://$EXP6_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$EXP6_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

The analysis includes checkpoint/seed tables, a throughput summary and
`nodes_by_training_time.png`. It does not automatically run LBR or
head-to-head evaluation; the saved policies support the same evaluator as
Experiment 2. See the [full protocol](experiments/fhp/exp6_fhp_structured_n2_standard16/README.md).

## Active Experiment 7: eight-worker counterpart to Experiment 6

This retains the exact Experiment 2/6 learning configuration, **three seeds
(0, 1, 2)**, **24 active hours**, and one **n2-standard-16** per seed. Only
trajectory collection becomes synchronous Ray-parallel: **eight single-threaded
actors per VM**, sharing **10,000 total traversals per player per iteration**.
The central learner still fits sequentially with eight Torch threads. Critic
target caching is off. Run Experiments 6 and 7 from the same source commit.

Playable policies, without full training states, are saved at completed
iterations crossing **6, 12, 18 and 24 active hours**. Active time excludes policy
checkpoint fitting, saving and uploads, so VM runtime exceeds 24 hours.
Seed labels match, but parallel actor RNG streams mean sampled trajectories
will not be identical to serial training.

From the repo root, with the usual cloud variables set and the new code pushed:

```bash
./gcp/run_exp7_parallel_structured_n2_standard16.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp7-par8-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp7_parallel_structured_n2_standard16.sh run
```

Use `PYTHON=/path/to/venv/bin/python` for local smoke if needed. The controller
runs cloud smoke, three training VMs, and aggregation without the laptop staying
on. `PARALLELISM` controls concurrent **seeds**, not actors. Three seeds require
48 available regional N2 vCPUs. Cloud smoke includes production-capacity replay
checkpoint saving/loading and a real eight-actor restart-equivalence check;
production is submitted only after these pass. Local smoke uses tiny buffers.

```bash
./gcp/run_exp7_parallel_structured_n2_standard16.sh status
# Orchestration/aggregation recovery only; interrupted training needs a new RUN_ID:
./gcp/run_exp7_parallel_structured_n2_standard16.sh resume
```

The analysis provides node/time curves, a training-time breakdown, memory and
per-seed throughput diagnostics. It does not assume an eightfold speedup or
automatically launch strategic evaluation. See the [full protocol](experiments/fhp/exp7_fhp_parallel_structured_n2_standard16/README.md)
for downloads and the analysis-only comparison command for Experiments 6/7.

## Active Experiment 8: 48-hour parallel run with resumable final states

Experiment 8 retains Experiment 7's exact learning configuration and hardware:
**three seeds (0, 1, 2), one n2-standard-16 each, eight traversal actors per VM**.
Each seed trains for **48 active hours**, with playable policies at
**6, 12, 18, 24, 30, 36, 42 and 48 hours**, but a full training state **only at 48 hours**.
The complete final state includes optimisers, replay, critic history and driver/
actor random-number states, so training can genuinely continue later.

From the repo root, after committing/pushing the code and setting the usual
PROJECT_ID, REGION, BUCKET and SA_EMAIL:

```bash
./gcp/run_exp8_parallel_48h.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp8-par48-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
export EXP8_TOTAL_HOURS=48
unset EXP8_SOURCE_RUN_ID
./gcp/run_exp8_parallel_48h.sh run
```

Set PYTHON to the repository virtual environment's interpreter if needed for
local smoke. The remote controller runs smoke, training and aggregation without
the laptop remaining on. Cloud smoke checks full-capacity checkpoint storage;
the eight-actor training smoke also tests extension past a completed endpoint.

The training timeout is **72 wall-clock hours per attempt**, with no automatic training retry.
The requested budget is still 48 active hours; checkpoint policy fits, saving
and uploads add billable overhead. Three concurrent seeds use 144 active VM-hours
and require 48 available regional N2 vCPUs.

```bash
./gcp/run_exp8_parallel_48h.sh status
# Recover orchestration/aggregation or a saved final state; no mid-training recovery:
./gcp/run_exp8_parallel_48h.sh resume
```

To extend a completed 48-hour run to 72 cumulative hours later, retain its
original REPO_REF and choose a **new** RUN_ID:

```bash
export EXP8_SOURCE_RUN_ID="YOUR_COMPLETED_EXP8_RUN_ID"
export EXP8_TOTAL_HOURS=72
export RUN_ID="exp8-to72-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp8_parallel_48h.sh extend
```

The original results are preserved. Each seed resumes its final full state,
not just the playable policy weights. The new analysis includes the earlier
checkpoints and the added training. See the [full protocol](experiments/fhp/exp8_fhp_parallel_48h/README.md)
for download commands, compatibility checks and continuation details.

## Active Experiment 9: cached critics in the 24-hour parallel run

Experiment 7's **24-active-hour**, **three-seed** configuration with cached
frozen critic targets. Three **n2-standard-16** VMs run concurrently, each with
eight traversal actors. Networks, replay, sampling and fitting budgets stay
unchanged. Cache targets are rebuilt once per critic fit; this is an efficiency
change, not a new learning rule.

Keep playable policies at **6, 12, 18 and 24 hours**, analysis and diagnostics.
As in Experiment 8, save **one full resumable state per seed at the final
endpoint only**, including optimisers, replay and driver/actor RNG state.
No intermediate full training states are retained. Temporary target caches
are rebuilt after continuation.

After committing/pushing, from the FHP repository root with existing
PROJECT_ID, REGION, BUCKET and SA_EMAIL:

```bash
./gcp/run_exp9_cached_parallel_24h.sh smoke-local  # Optional local check

export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP9_RUN_ID="exp9-cache24-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP9_RUN_ID"
export PARALLELISM=3
export EXP9_TOTAL_HOURS=24
unset EXP9_SOURCE_RUN_ID
./gcp/run_exp9_cached_parallel_24h.sh run
./gcp/run_exp9_cached_parallel_24h.sh status
```

The remote controller runs mandatory smoke, training and aggregation. It needs
48 available N2 vCPUs for three simultaneous training VMs; the laptop can
disconnect after submission. Initial workers have a 36-hour wall ceiling and
zero automatic retries. Allow 72 active VM-hours plus overhead. Interrupted
training before the final state cannot be resumed from a playable policy.

To continue a completed run to 48 cumulative active hours, keep its original
REPO_REF, and use a new destination:

```bash
export EXP9_SOURCE_RUN_ID="$FHP_EXP9_RUN_ID"
export EXP9_TOTAL_HOURS=48
export RUN_ID="exp9-to48-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp9_cached_parallel_24h.sh extend
```

The original run remains untouched. Outputs include checkpoint/seed tables,
throughput charts, existing parallel diagnostics and new critic-cache timings.
No exact exploitability or additional exploiter/head-to-head job is claimed.
See the [full configuration, continuation, validation and download protocol](experiments/fhp/exp9_fhp_cached_parallel_24h/README.md).

## Active Experiment 10: explicit hand-strength and board-interaction features

Experiment 9 plus **30 additive player-information descriptors**: exact made-hand
ranks and kickers, pocket-pair position, hole/board rank matches, board texture,
suit concentration and current straight structure. All exact suit-canonical
cards and betting information remain; there are no new buckets or state mergers.
Policy inputs grow from 183 to **213**; critics gain per-player descriptors and
grow from 263 to **323**. Hidden widths and all learning budgets are unchanged.

The experiment retains **three seeds, 24 active hours, eight traversal actors
per n2-standard-16 VM, cached critic targets**, playable policies at 6/12/18/24
hours and one final resumable state per seed. Existing encoders/checkpoints are
unaffected. It starts fresh rather than resuming Experiment 9's incompatible
input layout. After committing and pushing, with the usual FHP cloud variables:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP10_RUN_ID="exp10-features-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP10_RUN_ID"
export PARALLELISM=3
export EXP10_TOTAL_HOURS=24
unset EXP10_SOURCE_RUN_ID
./gcp/run_exp10_hand_board_features.sh run
```

Cloud smoke precedes training and aggregation. The hypothesis is improved
generalisation, not a guaranteed policy-quality gain. Saved policies support
the existing encoder-aware evaluator; rule-agent, LBR and cross-play evaluation
remain separate from training, as in Experiment 9. See the
[feature specification, comparison protocol and continuation instructions](experiments/fhp/exp10_fhp_hand_board_features/README.md).

## Active Experiment 11: explicit pot and betting-economics features

An independent follow-up to **Experiment 9**, without Experiment 10's extra card
features. Sixteen additive public descriptors expose pot/call cost in BB, pot
odds, current-round contributions, remaining raises, position and recent
aggression. Exact cards and betting history remain intact. Features are cached
by public betting history; player/critic inputs become **199/295** while hidden
widths and all optimisation budgets stay fixed.

Retains three seeds, **24 active hours**, eight traversal actors per
**n2-standard-16**, cached critics, playable 6/12/18/24-hour checkpoints and one
final resumable state per seed. After committing/pushing, with existing cloud
variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP11_RUN_ID="exp11-econ-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP11_RUN_ID"
export PARALLELISM=3
export EXP11_TOTAL_HOURS=24
unset EXP11_SOURCE_RUN_ID
./gcp/run_exp11_betting_economics.sh run
```

Mandatory cloud smoke precedes training/aggregation. Follow-up rule-agent, LBR
and cross-play evaluation remains separate, as in Experiment 9. See the
[feature definitions, validation, continuation and download instructions](experiments/fhp/exp11_fhp_betting_economics/README.md).

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
