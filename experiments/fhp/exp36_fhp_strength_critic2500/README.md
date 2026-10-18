# Experiment 36 — richer strategic inputs with 2,500 critic updates

## Question and frozen specification

Does combining the smaller critic budget with the strongest rare-decision feature
package produce a better 24-hour policy? This is a fresh-start experiment, not a
warm start from a differently shaped network.

- Baseline: Experiment 20, with **2,500 gradient updates per critic per fit**
  instead of 5,000. Both critics use the reduced budget. Ordinary cached TD
  fitting, target averaging, traversal sampling and all other learning settings
  remain unchanged; no adaptive controller or alternative critic objective.
- Representation: **Experiment 33's AE encoder**, unchanged. It retains the
  original information-preserving hand/board encoding and adds relative strength,
  an unbeatable-hand indicator, betting context and strength–betting interactions.
  Relative strength is against uniformly distributed compatible holdings, **not**
  an action-conditioned range and never the opponent's actual hidden cards.
  It is not the union of all packages in Experiments 29–33. Input sizes are
  231 for player networks and 359 for the full-state critic. Hidden widths,
  grouped policy loss and fitting budgets are unchanged.
- Three independent seeds, **0, 1 and 2**, run concurrently on separate
  **n2-standard-16** CPU VMs (16 vCPU, 64 GiB each). Each has eight Ray collectors
  and eight central fitting threads; no GPU.
- Train for **24 active hours per seed**. Fit/save playable average policies at
  **6, 12, 18 and 24 hours**. Stop after finalisation; do not launch evaluation.
- Save a **complete training state at 24h**, including networks, optimisers,
  replay, target histories, calibration/gating state, counters, timing and driver
  and all eight actor RNG states. Earlier six-hour files are playable policies,
  not full restart points. Preserve the final `.pt` and manifests in GCS.

The active clock matches Experiment 20: policy-checkpoint fitting, saving and
uploads are outside it. Checkpoints occur at safe iteration boundaries, so
24 hours is not an exact VM shutdown deadline. Allow setup and finalisation on
top of 24h (36h initial task ceiling). Three seeds use at least 72 VM-hours in
aggregate. Higher throughput is not evidence of stronger play.

## Validation and observability

The smoke gate checks the AE encoder against the original feature prefix, real
eight-actor cached/uncached equivalence, exact next-iteration restart from disk,
and extension of a **completed** endpoint. The cloud smoke also fills production
replay allocations and round-trips a full state. Actual critic update budgets
are logged and checked when aggregating production runs.

Retain run/runtime/checkpoint manifests, source commit and checksums, worker logs,
resource snapshots, failure diagnostics, critic-cache counters, peak RSS and
phase timings. Periodic and final cloud uploads are inherited from Experiment
20. Paid training starts only after smoke succeeds; existing successful stages
are reused. Failure is not permission to automatically start training again.

## Run from the repository root

Commit and push these files first. Assume `PROJECT_ID`, `REGION`, `BUCKET` and
`SA_EMAIL` are set. Keep the run ID and original commit for later continuation.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP36_TRAIN_REF="$REPO_REF"
export RUN_ID="exp36-rich2500-$(date -u +%Y%m%d-%H%M%S)"
export EXP36_RUN24_ID="$RUN_ID"
export EXP36_TOTAL_HOURS=24
unset EXP36_SOURCE_RUN_ID

# Optional local specification inspection: no cloud actions.
bash gcp/run_exp36_strength_critic2500.sh dry-run

# Cloud smoke only.
bash gcp/run_exp36_strength_critic2500.sh smoke-cloud
```

After that smoke succeeds, **keep the same RUN_ID**:

```bash
bash gcp/run_exp36_strength_critic2500.sh run
```

The controller reuses that smoke, trains the three seeds and aggregates results.
You can close your laptop. Alternatively, `run` alone performs smoke → training
→ aggregation without requiring another command. It never starts evaluation.
Use `status` for job status. `resume` restarts only the controller and reuses
successful stages; it stops on failed child jobs, rather than silently retraining.

## Separate evaluation

```bash
export RUN_ID="$EXP36_RUN24_ID"
export REPO_REF="$EXP36_TRAIN_REF"
export EXP36_TOTAL_HOURS=24
unset EXP36_SOURCE_RUN_ID
bash gcp/run_exp36_strength_critic2500.sh evaluate
```

One separate CPU job evaluates all four cohorts' **24h policies**, with:

- Primary head-to-head: Experiment 36 versus Experiment 20.
- Secondary head-to-head: Experiment 36 versus Experiments 33 and 10.
- 50,000 duplicated deal pairs per comparison per training seed; both seats.
- The established five rule agents (10,000 pairs each) and LBR (1,000 pairs,
  4,096 rollouts, shards of 10) for every policy; no repeated candidate-panel
  evaluation for each comparator.
- Batched terminal call/fold diagnostics: folding probability and conditional
  BB loss for guaranteed winners, including tails and the known straight-flush
  and four-of-a-kind examples. Assessment board classes exclude the recorded
  earlier audits and Experiment 35's assessment classes; they are not guaranteed
  absent from training. Models never train on this panel.

Historical sources are `exp20-critic24-20261006-164357`,
`exp33-features-20261012-143026` and `exp10-features-20261001-161740`.
Only policy/metadata files are downloaded for evaluation, not replay states.
Evaluation first runs a small protocol smoke. It uploads atomic shard results
periodically and on exit; `evaluate-resume` reuses validated completed work at
the same revision. Its 48h wall ceiling is a limit, not a runtime prediction.

Training seeds, not individual hands or selected positions, are the inferential
units. Positive direct play favours Experiment 36; lower LBR exploiter payoff is
better. Rare-state loss is conditional on reaching the selected decision, not
whole-game exploitability. Pointwise intervals are exploratory. The comparison
with Experiment 20 changes two factors; that with Experiment 33 holds features
fixed but changes the critic budget from 10,000 to 2,500. These comparisons do
not identify a factorial interaction or establish convergence.

## Optional continuation to 48 hours — only when requested

Retain the **original training commit** and use a **new run ID**. This continues
the full Experiment 36 states, not Experiment 20/33 policies. Source artefacts
are read-only and not overwritten.

```bash
export REPO_REF="$EXP36_TRAIN_REF"
export EXP36_SOURCE_RUN_ID="$EXP36_RUN24_ID"
export EXP36_TOTAL_HOURS=48
export RUN_ID="exp36-rich2500-48h-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp36_strength_critic2500.sh extend
```

This adds 24 active hours, keeps the source policies and saves new policies at
30/36/42/48h and a full state at 48h. It does not retrain the first 24h or evaluate
automatically. The evaluator above deliberately accepts only the original 24h
cohort; a later 24-versus-48h study should have its own comparison specification.

## Download analysis

```bash
export RUN_ID="$EXP36_RUN24_ID"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"

# After the separate evaluation completes:
mkdir -p "cloud_outputs/$RUN_ID/evaluation/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/evaluation/analysis" \
  "cloud_outputs/$RUN_ID/evaluation/analysis"
```

Analysis downloads do not include the full restart states. Keep those in GCS;
the continuation launcher downloads them with checksum/provenance validation.
