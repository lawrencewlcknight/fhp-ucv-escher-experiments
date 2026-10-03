# FHP Experiments 9–12: four-way frozen-policy evaluation

Does adding explicit hand/board, betting-economics or critic-only showdown
features improve the cached UCV-ESCHER baseline at equal active training time?
This job evaluates existing policies; it does **not** train or refit any model.
The numerical scorer, rule agents, restricted LBR, duplicate-deal seed schedule,
task caching and seed-level aggregation are reused from the earlier evaluations.
Older evaluators and their resume fingerprints are unchanged.

## Frozen sources

| Experiment | Source run | Change relative to Exp9 |
|---|---|---|
| 9 | `exp9-cache24-20261001-132550` | Cached parallel baseline |
| 10 | `exp10-features-20261001-161740` | Explicit visible hand/board features |
| 11 | `exp11-econ-20261001-153659` | Explicit betting-economics features |
| 12 | `exp12-showdown-20261001-165650` | Training-critic-only showdown features |

All **48 policies** are required: seeds **0, 1, 2**, saved at **6, 12, 18 and
24 active hours**, for each experiment. These are completed-iteration threshold
crossings, not exact wall-clock instants. The source configurations must equal
the canonical per-experiment configuration, differing only in the prescribed
feature encoder. Require completed production runs, eight traversal actors,
eight learner threads and `n2-standard-16` source VMs.

Before scoring, verify checksums, checkpoint metadata, game definition,
reloadability, feature-encoder identity, runtime metadata and one coherent
source commit within each experiment. Different experiments may have different
commits: this is a historical comparison, not a synchronised new training trial.
Smoke also validates all 48 policies. Missing, mixed, corrupt, smoke-training or
extended-horizon inputs are rejected. Inputs are read-only and full continuation
states/replay are neither required nor downloaded.

Exp12's privileged showdown comparison is **not** supplied to the deployed
policy. Its checkpoint uses the versioned critic-showdown encoder, whose
policy-side features remain the original visible-information representation.
Each checkpoint is loaded with its own encoder; unlike-sized inputs are never
forced into a common network or silently converted.

## Evaluation protocol

| Evaluation | Coverage | Duplicate deal pairs per matchup |
|---|---|---:|
| Five existing rule agents | Each policy against each opponent | 10,000 |
| Restricted Local Best Response | Each policy; 4,096 preflop rollouts | 1,000 |
| Matched-time head-to-head | All six experiment pairs, matched seed labels, all four checkpoints | 50,000 |
| Temporal head-to-head | All six later-versus-earlier contrasts within every experiment/seed | 50,000 |

An unordered experiment pair is played once per matched seed/checkpoint using
two seat-swapped hands per deal. No nine-way cross-seed tournament is introduced.
The full plan has **5,184 scoring tasks**: 240 rule matches, 4,800 ten-pair LBR
shards, 72 matched-time matches and 72 temporal matches. This totals **9,648,000
duplicate pairs / 19,296,000 hands**, excluding LBR's internal rollouts and smoke.
Common chance/action seeds across methods and checkpoints match the previous
protocol. Scoring randomness is separate from training randomness.

The primary endpoint family is **Exp10, Exp11 and Exp12 versus Exp9 at 24
hours**. The other three pairings and earlier/temporal results are supporting
comparisons. Policy A is always the higher-numbered experiment, so positive
direct payoff favours A; positive temporal payoff favours the later policy.
Higher rule payoff is better, whereas lower LBR payoff is better. Paired
rule/LBR differences are A minus B (negative LBR differences favour A).

Three training-seed means are the uncertainty unit. Keep within-match
duplicate-pair intervals, and report across-seed standard errors and Student-t
95% intervals. Compute diagnostic differences within matched seed before
aggregation; average temporal contrasts within each run before aggregating.
Hands, LBR shards, opponents and checkpoint pairs are not independent training
replicates. With only three seeds and multiple comparisons, intervals are
exploratory and not multiplicity-adjusted; do not infer a definitive winner
from isolated positive cells or treat the primary family as one hypothesis.

Report training nodes alongside active time. Different feature encoders change
input sizes, sometimes parameter counts, RNG consumption and throughput; seed
labels do not guarantee paired trajectories. There is **no arbitrary
node-matched head-to-head comparison** and no interpolation of policies.
Node-based curves are descriptive, not a controlled equal-node experiment.
Active time excludes policy fitting/checkpoint overhead and is not billable
VM time. Restricted LBR is **not exact exploitability**, and strong direct or
rule-agent play alone does not establish equilibrium convergence.

## GCP launch

Commit and push the new code first. From the repository, with `PROJECT_ID`,
`REGION`, `BUCKET` and `SA_EMAIL` already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP9_RUN_ID=exp9-cache24-20261001-132550
export EXP10_RUN_ID=exp10-features-20261001-161740
export EXP11_RUN_ID=exp11-econ-20261001-153659
export EXP12_RUN_ID=exp12-showdown-20261001-165650
export RUN_ID="fhp-eval9to12-$(date -u '+%Y%m%d-%H%M%S')"

# Cloud smoke followed automatically by full evaluation, only if smoke passes.
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh run
```

The four source IDs above are also the launcher's defaults. `BUCKET` accepts
either a bare bucket name or a `gs://` bucket. One on-demand **n2-standard-16**
VM uses **16 scoring processes**, one Torch thread each. Its **48-hour elapsed
safety limit** includes setup, smoke and uploads; this is a cap, not a runtime
estimate or a required training duration. `EVAL_MAX_HOURS` can override it
(1–72). Python is pinned to **3.11.16** for reproducible restarts. No automatic
paid retries are enabled. The job exits as soon as scoring/uploads finish;
you may disconnect after submission.

Optional standalone smoke, using a separate output namespace:

```bash
export RUN_ID="fhp-eval9to12-smoke-$(date -u '+%Y%m%d-%H%M%S')"
EVAL_MAX_HOURS=2 bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh smoke-cloud
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh status
```

Smoke validates every checkpoint but scores seed 0 at 6/12h using tiny budgets
for all four methods and all six pairings. Smoke is an integration test, never
policy-strength evidence. A full `run` always performs its own smoke gate.
For a full run after this standalone test, set a new `RUN_ID` and use `run`.

```bash
# Generate a job specification without submission (requires a committed ref).
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh dry-run
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh status

# Following failure, keep the original full-run ID, source IDs and REPO_REF.
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh resume
```

Resume creates a new Batch job name and reuses the same output namespace.
Completed scoring tasks are checksum-verified and reused. Changed sources,
budgets, code or dependencies are rejected; the launcher also rejects active
jobs sharing the output namespace. Small results are uploaded every five
minutes and at exit. An abrupt termination can lose work since the last sync.
Resource snapshots, failure diagnostics and Cloud Logging output are retained.
Only outputs are uploaded, not the downloaded source policies.

## Outputs

`$BUCKET/$RUN_ID/analysis/` contains per-seed and aggregate CSV tables, a
plain-language `analysis_summary.md`, source/evaluation provenance,
`evaluation_manifest.json`, `SUCCESS.json` and resumable `task_results/`.
The seven charts are:

- Rule-agent and LBR trajectories against active hours and nodes (four charts).
- `matched_time_crossplay.png`: all six pairwise trajectories with seed intervals.
- `final_crossplay_matrix.png`: the 24-hour round-robin matrix; row policy payoff
  against column policy. Reversed cells are sign inversions, not new observations.
- `temporal_crossplay.png`: later-versus-earlier heatmaps for all four methods.

`direct_crossplay_aggregate.csv` includes every pair/checkpoint;
`final_crossplay_aggregate.csv` isolates the six final comparisons and marks the
three primary ones. Seed-level tables preserve node counts and checkpoint hashes.

```bash
export BUCKET_ROOT="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${BUCKET_ROOT%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Local tests

```bash
python -m pytest tests/test_retrospective_exp9_exp10_exp11_exp12_evaluation.py
python -m experiments.fhp.retrospective_exp9_exp10_exp11_exp12_evaluation.run \
  --exp9-run "cloud_outputs/$EXP9_RUN_ID" --exp10-run "cloud_outputs/$EXP10_RUN_ID" \
  --exp11-run "cloud_outputs/$EXP11_RUN_ID" --exp12-run "cloud_outputs/$EXP12_RUN_ID" \
  --output-dir /tmp/fhp-eval9to12-smoke --workers 2 --smoke \
  --rule-deals 4 --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 --crossplay-deals 10
```

Local scoring requires policy files and metadata under all four `workers/`
trees; the downloaded analysis-only folders are not sufficient. Cloud execution
downloads the required source workers automatically. No training resumes.
