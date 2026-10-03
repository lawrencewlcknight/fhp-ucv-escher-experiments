# FHP Experiments 7 and 8: longer training and late-stage progress

Evaluation only: does extending the unchanged Experiment 7 learning configuration
from 24 to 48 active hours improve policy quality, and is improvement still
detectable near the end of training? No solver or output-policy network is refitted.

## Frozen sources and comparisons

- Exp7: `exp7-par8-20261001-005151`, evaluated at **24h**.
- Exp8: `exp8-par48-20261001-005208`, evaluated at **24, 30, 36, 42 and 48h**.
- All **three training seeds (0, 1, 2)**: **18 scored policies**.
- Primary endpoint: **Exp8 48h versus its own 24h checkpoint**, within seed.
- Prespecified late-stage diagnostics: **Exp8 48h versus 42h and versus 36h**.
- All ten unordered pairs of the five Exp8 checkpoints are played, including
  all four consecutive six-hour comparisons. Later policy is always A.
- Each of the five Exp8 checkpoints plays the fixed Exp7 24h baseline, with
  matching seed labels: five further pairings. Exp8 is always A. In particular,
  24h versus 24h checks repeatability and 48h versus 24h tests the external baseline.

This is not a nine-pair cross-seed tournament. Each contrast has three matched
seed-level observations, consistent with prior evaluations. Exp8 started its
own trajectories, not from the Exp7 saved policies. Its within-run comparison
is the cleaner test of additional training. The original two runs share source
commit `c331185819a108df63bc1eb73fcdba3da0d870bf`; provenance is recorded rather
than assuming seed labels guarantee identical trajectories or hardware performance.

Before scoring, even in smoke, validate **all 36 policies** in both original
source schedules (Exp7 6–24h; Exp8 6–48h). Require completed production workers,
all three seeds, canonical learning configuration/game/encoder, n2-standard-16,
eight actors, eight Torch intra-/inter-op threads, uncached critic targets,
the original checkpoint schedules and horizons, matching hashes and reloadable
payload metadata. Missing, corrupt, smoke, mixed-commit-within-experiment or
imported-continuation sources are rejected. Do not download large training
states or replay reservoirs. Sources are read-only.

## Established evaluation protocol

The rule agents, LBR scorer, duplicate-deal/seat-swap implementation, common
deal/action seed schedules and durable task caching are reused from the existing
retrospective evaluators. Earlier evaluators are unchanged.

| Evaluation | Coverage | Duplicate deal pairs per matchup |
|---|---|---:|
| Five published rule agents | All 18 policies against every agent | 10,000 |
| Restricted Local Best Response | All 18 policies; 4,096 preflop rollouts | 1,000 |
| Exp8 temporal head-to-head | All ten checkpoint pairs, each seed | 50,000 |
| Exp8 versus Exp7 24h | All five Exp8 checkpoints, each seed | 50,000 |

Each duplicate pair uses the same chance deal in both seat allocations. There
are **1,935 tasks** (LBR split into ten-pair shards), **3,168,000 duplicate pairs /
6,336,000 hands**, excluding internal LBR rollouts. No exact exploitability is run.

All scores are mbb/hand. Positive direct scores favour Exp8/the later checkpoint.
Higher rule payoff is better; lower LBR payoff is better. Paired diagnostic
differences are A minus B, so a negative LBR difference favours A.

Compute differences within seed before aggregation. Report individual seed
results, within-match deal uncertainty, and Student-t 95% intervals across the
three seed means. Hands, shards and checkpoints are not independent training
replicates. Common evaluation deals across seeds make the seed intervals
conditional on that deal schedule; they are not a full decomposition of training
and Monte Carlo uncertainty. Secondary intervals are exploratory and not
multiplicity-adjusted; overlapping temporal contrasts are correlated.

## Assessing whether to train longer

The report separates overall 24→48h improvement from the recent 36→48h and
42→48h comparisons. Inspect all four six-hour gains alongside rule-agent and LBR
trajectories, their confidence bounds, and consistency across individual seeds.
Sustained late gains across these diagnostics would support a longer-training
follow-up. Flat point estimates with wide intervals are **inconclusive**, not
evidence that additional training cannot help.

There is no automatic convergence verdict, equivalence test or extrapolation
beyond 48 hours. A practical-equivalence margin has not been prespecified.
Failure to detect improvement is not evidence of equivalence, especially with
only three seeds. Cross-play may be non-transitive; beating previous selves or
rule agents does not prove general strength. Restricted LBR is approximate,
not exact exploitability. Even concordant flat results support at most a
plateau on this evaluation suite, not convergence to equilibrium.

Time is **active training time** at completed-iteration crossings. Policy
checkpoint fitting, saving and uploads are excluded. Equal time is not equal
node exposure; time/node plots are descriptive, not matched-node interventions.

## GCP full run and smoke test

Run from the repository root **after committing and pushing this code**.
Assume `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` are set. `BUCKET` may be
a bare bucket name or begin with `gs://`.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP7_RUN_ID=exp7-par8-20261001-005151
export EXP8_RUN_ID=exp8-par48-20261001-005208
export RUN_ID="fhp-eval78-$(date -u '+%Y%m%d-%H%M%S')"

# Real-checkpoint cloud smoke, then full scoring only if smoke passes.
bash gcp/run_retrospective_exp7_exp8_evaluation.sh run
```

One **n2-standard-16** VM (16 vCPUs, 64 GiB RAM; 62,000 MiB requested) runs 16
single-threaded scoring processes, with a 100 GiB pd-balanced boot disk. Python
is pinned to 3.11.16 and dependencies use the repository requirements. The
**48-hour elapsed safety cap** includes setup and smoke; scoring exits as soon
as complete. Override with `EVAL_MAX_HOURS` (1–72) if necessary. This is not a
prediction that evaluation will need 48 hours. Automatic paid retries are off.
The laptop can disconnect after submission; Batch owns the temporary VM lifecycle.

Optional **standalone GCP smoke only**, using its own new output prefix:

```bash
export RUN_ID="fhp-eval78-smoke-$(date -u '+%Y%m%d-%H%M%S')"
EVAL_MAX_HOURS=2 bash gcp/run_retrospective_exp7_exp8_evaluation.sh smoke-cloud
```

Choose a **new full-run RUN_ID** before the full command. Smoke validates all
36 policies but scores only seed 0, at all six selected method/hour combinations,
using tiny budgets. Its 51 tasks cover every comparison type and checkpoint;
its tables and all seven charts are explicitly labelled as integration-only.

```bash
# No cloud submission: writes a job specification using the pinned commit.
bash gcp/run_retrospective_exp7_exp8_evaluation.sh dry-run
bash gcp/run_retrospective_exp7_exp8_evaluation.sh status
```

Results upload every five minutes and on exit. Resource snapshots, failure
diagnostics and Cloud Logging support diagnosis of memory pressure or crashes.
After failure, retain the original full-run `RUN_ID`, both source IDs and
`REPO_REF`:

```bash
bash gcp/run_retrospective_exp7_exp8_evaluation.sh resume
```

Resume submits a new Batch job name to the same output prefix and reuses completed
tasks. Active concurrent jobs, changed policies, evaluator code, dependencies
or budgets are rejected. Abrupt termination may lose tasks since the last upload.
Use a new run ID for a changed evaluation contract.

## Outputs and download

`$BUCKET/$RUN_ID/analysis/` contains per-seed and aggregate rule/LBR results,
paired metric changes for every contrast, baseline and temporal cross-play,
separate adjacent-checkpoint gains and late-training endpoints. It includes
`analysis_summary.md`, all checkpoint provenance, `evaluation_manifest.json`,
`SUCCESS.json` and resumable `task_results/*.json`.

Seven figures:

1. Rule-agent payoff against active hours.
2. Rule-agent payoff against training nodes.
3. LBR payoff against active hours.
4. LBR payoff against training nodes.
5. Exp8 versus the fixed Exp7 24h baseline, with seed-level intervals.
6. Exp8's five-checkpoint cross-play matrix (reverse cells are sign-flipped,
   not independently scored).
7. Consecutive six-hour gains: direct play, rule payoff change and LBR payoff
   change, each with seed-level intervals.

Download analysis without the recovery cache:

```bash
FHP_EVAL_BUCKET="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${FHP_EVAL_BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Local verification

```bash
python -m pytest -q tests/test_retrospective_exp7_exp8_evaluation.py
python -m experiments.fhp.retrospective_exp7_exp8_evaluation.run \
  --exp7-run "cloud_outputs/$EXP7_RUN_ID" --exp8-run "cloud_outputs/$EXP8_RUN_ID" \
  --output-dir /tmp/fhp-eval78-smoke --workers 2 --smoke \
  --rule-deals 4 --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 --crossplay-deals 10
```

Local scoring requires the policy files and metadata in each source `workers/`
tree; analysis-only downloads are insufficient. The GCP job downloads these
automatically, so local analysis-only downloads do not prevent cloud launch.
Use a fresh output directory or `--resume` with exactly the same protocol.
