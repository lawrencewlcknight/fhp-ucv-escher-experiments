# Experiment 9: 24-hour parallel UCV with cached critic targets

This is **FHP**, not Leduc. The baseline is Experiment 7: Experiment 2's
structured, lossless suit-canonical configuration using eight synchronous
traversal actors. The only training-path change is enabling the per-fit frozen
critic-target cache tested by Experiment 5. Experiment 8 supplies the final
continuation-state mechanism. Existing experiments and default cache settings
are unchanged.

## Frozen design

- Three seeds **0, 1, 2**, each on a separate **n2-standard-16**, concurrent by
  default. Eight single-threaded traversal actors and eight central learner
  Torch threads per VM; independent learner fits remain sequential.
- **24 active training hours** per seed, stopping at the first completed outer
  iteration crossing the boundary, with no node cap.
- **10,000 total traversals per player per iteration**, not per actor. Actor
  chunking and RNG stream assignment are unchanged from Experiment 7.
- Same structured networks, compact replay capacities, fixed beta=1, two
  cross-fitted critics, four-fit critic-target averaging, disabled predictor,
  residual calibration and grouped reset soft-target cross-entropy policy fit.
- Only `cache_frozen_critic_targets=True` differs in the learning config.
  Each eligible critic fit computes all occupied replay-row targets once,
  using its frozen critic/regret inputs and current iteration. Minibatch
  sampling, losses, optimiser steps and target averaging are unchanged. The
  temporary target array is discarded after the fit and rebuilt next time.
- Same active clock as Experiment 7: checkpoint policy fitting, persistence
  and uploads are excluded; these costs remain billable and separately logged.

The hypothesis is increased throughput at unchanged learning rules, not a
guaranteed speedup or strategic improvement. Equal active time allows more
iterations if caching helps, so endpoints need not match the uncached run.
Historical comparisons also include machine/runtime variation. Exact
exploitability is not computed for FHP.

## Outputs and continuation

Playable policies are saved at **6, 12, 18 and 24 active hours**. Retain all
analysis, diagnostics and metadata. There is **one full training state per
seed at the final 24-hour endpoint**, and no intermediate full states.

The final state includes driver networks, critic target history, optimisers,
persistent replay, learner/sampler RNGs, all eight traversal actors' RNGs,
iteration/node/time counters, and cache diagnostic counters. Transient regret
rows are rebuilt next iteration; their sampler RNGs are retained. Computed
critic targets are not checkpointed: they are rebuilt at the next fit.

These final states are larger than playable policies and must stay in GCS to
allow continuation; an analysis-only local download need not include them.
This policy reduces archival volume but **does not offer mid-run recovery**.
If training fails before its final state is durable, use a new RUN_ID. Zero
automatic training retries prevent silently paying for a restart.

Analysis includes the existing checkpoint index, seed summaries, manifests,
nodes-by-time chart and parallel timing breakdown. The new
`critic_cache_summary.csv` records cached fit counts, rows, peak target-cache
bytes, critic fitting time and cache construction time. Cache-build time is
part of critic-fit time, which is part of learner time: do not add overlapping
timers. All counters continue across a resumed run.

No additional head-to-head or approximate-exploiter job is automatically
launched. Retained policies contain the same encoder metadata and can be
evaluated retrospectively with the existing encoder-aware FHP suite.

## Run on GCP

Run from this repository root after the new code has been committed and
pushed. Reuse PROJECT_ID, REGION, BUCKET and SA_EMAIL from your working FHP
runs. Do not reuse the Leduc repository's REPO_REF.

```bash
# Optional local smoke, using a Python environment containing Ray and this repo:
./gcp/run_exp9_cached_parallel_24h.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP9_RUN_ID="exp9-cache24-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP9_RUN_ID"
export PARALLELISM=3
export EXP9_TOTAL_HOURS=24
unset EXP9_SOURCE_RUN_ID
./gcp/run_exp9_cached_parallel_24h.sh run
```

The controller runs cloud smoke, three training workers, then aggregation.
After successful submission the laptop may disconnect. Three concurrent VMs
require **48 available regional N2 vCPUs**. Total active training is 72
VM-hours; elapsed completion is more than 24 hours because setup, checkpoint
fits, final serialization, uploads and aggregation add overhead.

Initial training tasks have a **36-hour wall-clock ceiling**, matching
Experiment 7. The controller retains the existing seven-day ceiling and
controller retry policy; controller failure does not cancel running children.
The smoke's production-capacity checkpoint test runs only on GCP. It does not
prove worst-case fitting memory safety or long-run convergence.

```bash
./gcp/run_exp9_cached_parallel_24h.sh status
# Same RUN_ID, REPO_REF, EXP9_TOTAL_HOURS and source setting:
./gcp/run_exp9_cached_parallel_24h.sh resume
```

`resume` recovers orchestration/aggregation or a durable final state. It
does not manufacture a recovery point from an intermediate playable policy.

## Extend a completed run

For example, continue the same trajectories from 24 to **48 cumulative active
hours** (24 additional hours). Keep the **original pinned REPO_REF** and
software/runtime contract; use a new destination:

```bash
export EXP9_SOURCE_RUN_ID="$FHP_EXP9_RUN_ID"
export EXP9_TOTAL_HOURS=48
export RUN_ID="exp9-to48-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp9_cached_parallel_24h.sh extend
```

The source must be a completed **Experiment 9** run, not Experiment 7 or 8.
Configuration, seed, source commit, runtime, cache mode and checkpoint hashes
are validated. Source objects are never modified. Earlier playable checkpoints
are imported; new policies are appended at 30, 36, 42 and 48 hours. Only the
extension's final full state is retained in its output prefix. The copied
input state is temporary, excluded from publication and removed after use.

Extensions may add 6--48 hours in six-hour increments and can themselves be
extended later. Total hours are cumulative, not an increment. Extension tasks
retain Experiment 8's 72-hour wall-clock ceiling.

## Validation and downloads

Cloud smoke must pass before production starts. It checks:

1. Three complete cached/uncached iterations with all eight actors, including
   average-policy fitting; compares learner, actor and sampler RNG state,
   replay, model weights, target history and optimiser state.
2. An additional critic fit using 2,048-row minibatches, eight fitting threads
   and a partial final cache chunk.
3. Disk-backed restart versus uninterrupted training; verifies bitwise state
   equality and preservation of cumulative cache counters.
4. Four playable checkpoints, final-state reuse without additional training,
   completed-endpoint extension, policy reload and both aggregations.
5. Production-capacity replay/checkpoint round-trip (cloud only).

These are correctness gates, not performance evidence. Local tests:
`python -m pytest tests/test_exp9_cached_parallel_24h.py`; omit real Ray work
with `-m 'not ray'`.

Download the small analysis folder after completion:

```bash
export RUN_ID="$FHP_EXP9_RUN_ID"
FHP_EXP9_BUCKET="${BUCKET%/}"
[[ "$FHP_EXP9_BUCKET" == gs://* ]] || FHP_EXP9_BUCKET="gs://$FHP_EXP9_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_EXP9_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```
