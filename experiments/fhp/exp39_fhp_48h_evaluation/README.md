# Experiment 39 — 48-hour configurations and continuation benefit

Evaluation only: no new training, policy fitting or inference safeguard. The
scoring, rule-agent panel, LBR implementation, duplicate-deal protocol and
resumable process executor are reused from the previous evaluation experiments.

## Questions and frozen sources

| Cohort | Training | Critic updates | Input features | Source run |
|---|---:|---:|---|---|
| Experiment 16 | 48 active hours | 10,000 | Original hand/board | `exp16-feat48-20261004-182051` |
| Experiment 36 | 24 active hours | 2,500 | Richer AE strength/betting | `exp36-rich2500-20261018-205125` |
| Experiment 38 | 48 active hours | 2,500 | Richer AE strength/betting | `exp38-rich48-20261019-230016` |

There are three trained seeds per cohort. Experiment 38 continues the exact
Experiment 36 trajectories, rather than training three additional independent
models. Experiment 16 continues Experiment 10. Native checkpoint identities
remain Experiment 36 and Experiment 10 respectively.

The two prespecified primary comparisons are:

1. **38 versus 16:** which complete configuration performs better at 48 hours?
   Features and critic budget both differ, so this is not a critic-budget ablation.
2. **38 versus 36:** does a further 24 hours improve the richer configuration?
   Also report the three same-seed predecessor matches separately. Beating an
   earlier policy does not establish convergence or a smaller Nash gap.

The runner checks native manifests, actual policy payloads, encoder metadata,
runtime/configuration hashes, complete checkpoint schedules, continuation source
receipts and unchanged imported 6/12/18/24h policies before scoring. Experiment
10 (`exp10-features-20261001-161740`) is downloaded only to validate Experiment
16's lineage. It is **not** an additional evaluated cohort. Large `.pt` full
training states and replay buffers are not downloaded.

## Fixed evaluation budgets

| Component | Production allocation |
|---|---|
| Head-to-head | Two contrasts × nine seed pairings × 50,000 duplicate pairs; 5,000-pair shards |
| Rule panel | Five established agents × 10,000 duplicate pairs for each of nine policies |
| LBR | 1,000 duplicate pairs per policy, 4,096 preflop rollouts, 10-pair shards |
| Rare states | Exact frozen panel from `exp37-critic-compare-20261019-222256/analysis/rare_panel.json` |

A duplicate pair plays both seats. Total: **1,125 match tasks, 1,359,000 duplicate
pairs, 2,718,000 hands**, plus batched rare-state inference. Only final 24/48h
endpoints are scored; 30/36/42h policies are validated but not evaluated.

The frozen rare panel is pinned by SHA-256
`092dcd1c88d40037df03afe0441cbc03d297cca9d4ed06e18bec6d7af642f2f4`.
Its historical development/assessment labels are retained for comparability,
but **all of it is now previously examined diagnostic data**, not a new holdout.
Reports include folding probabilities, conditional loss, state-level tails and
per-position action probabilities. These are not reach-weighted whole-game losses.

## Statistical interpretation

- All-nine-pair H2H means weight each seed pairing equally. The nine cells are
  not nine independent training runs.
- Bootstrap training-model weights are shared between Experiment 36 and 38,
  preserving their paired trajectory structure; Experiment 16 is independently
  resampled. The empirical all-pair endpoint includes both diagonal and
  off-diagonal matches. Same-trajectory predecessor results are a separate
  secondary endpoint, not a substitute for that average.
- Report pointwise 95% and Bonferroni 97.5% percentile intervals for the two
  primary contrasts. Independent cell-level Gaussian deal-SE perturbations are
  an additional conservative sensitivity component; empirical model effects
  already contain deal noise. Separate conditional deal-only SEs/intervals are
  provided. Three seeds make all training-population inference exploratory.
- Rule-agent deals/action streams are matched between each 36/38 trajectory.
  Supporting common-panel changes use paired three-seed t intervals; they do
  not claim per-deal paired confidence intervals. LBR streams are independent
  between models and interpreted as restricted-exploiter performance, not exact
  exploitability. Higher target payoff and lower LBR payoff are favourable.
- No automatic algorithm promotion or full league ranking: there is no 16
  versus 36 primary match, and poker comparisons need not be transitive. A
  stronger configuration should have corroborating common-opponent results,
  no important diagnostic regression and acceptable cost. An interval spanning
  zero is inconclusive, not evidence of equivalence or convergence.

## Run on GCP Batch

Commit and **push this implementation before launching**: the VM uses the pinned
Git revision, never uncommitted local files. From the repository root:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp39-48h-compare-$(date -u +%Y%m%d-%H%M%S)"

bash gcp/run_exp39_48h_evaluation.sh run
```

One `n2-standard-16` CPU Batch job runs validation, an integrated small smoke
test, then the full evaluation and report. Full scoring uses 16 processes and
one computation thread per process. The laptop may disconnect after submission.
Estimated wall time is 3–5 hours based on the previous comparison, **not a
guarantee**. The default 48-hour Batch ceiling leaves substantial headroom;
`EXP39_MAX_HOURS` can change it (1–72). There is no performance-timing gate and
no automatic paid retry. A standalone optional smoke job is available:

```bash
export RUN_ID="exp39-48h-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp39_48h_evaluation.sh smoke-cloud
```

For production after a standalone smoke, use a **new RUN_ID** and `run`. Smoke
scores are not reused as production measurements. `dry-run` builds job JSON
without cloud submission; `status` lists jobs for the output namespace.

## Recovery and outputs

Every completed match shard and rare-policy assessment is atomically cached.
Uploads occur every five minutes and at normal/handled termination; resource
snapshots occur every 30 seconds. An abrupt VM loss can lose work since the last
successful upload, but not the entire already-uploaded analysis. A task failure
stops new dispatch while collecting successful in-flight tasks. Reports include
attempt timing and per-task cost. Batch logs and memory diagnostics remain available.

After a failure, with the **same RUN_ID, REPO_REF and scoring budgets**:

```bash
bash gcp/run_exp39_48h_evaluation.sh resume
```

This submits a new job name but restores the same output prefix. Checksums and
fingerprints prevent mixing changed sources, code, runtime, panel or budgets;
such changes require a new run. No active job may share the output prefix.
Do not delete remote `task_results` if further recovery may be needed.

Download analytical summaries after success (the large task cache is optional):

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Key outputs are `analysis_summary.md`, `summary.json`, H2H seed-pair tables and
interval plots, `same_trajectory_head_to_head.csv`, paired common-opponent
changes, LBR/rule scores, rare-state diagnostics and full checkpoint provenance.
Training context includes nodes, active time and recorded critic/collection
timings. The pinned native restart format restores cumulative critic and parallel
counters, so these include the original 24h segment, not only the continuation.
Nested timers must not be added together. Evaluation task-time sums are not VM
elapsed time; setup, extraction and infrastructure overhead are separate.
