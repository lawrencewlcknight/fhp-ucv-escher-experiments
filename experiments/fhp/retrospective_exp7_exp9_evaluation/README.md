# FHP Experiments 7 and 9: retrospective policy-quality evaluation

Evaluation only: does Experiment 9's critic-target caching translate its higher
training throughput into stronger policies than Experiment 7 at equal active
training time? Neither the solver nor the output-policy network is refitted.
Both runs used three seeds and eight traversal actors on `n2-standard-16`.

## Frozen sources and validation

- Experiment 7: `exp7-par8-20261001-005151` (uncached critic targets).
- Experiment 9: `exp9-cache24-20261001-132550` (cached critic targets).
- Seeds **0, 1, 2**, at **6, 12, 18 and 24 active training hours**: 24 policies.
  Saved checkpoints are completed-iteration crossings, not exact clock instants.
- Require successful production workers, the canonical per-experiment learning
  configuration, eight actors, eight Torch threads and matching parallel settings.
  After normalising the cache flag, learning configurations must be identical.
- Verify policy checksums, canonical FHP game, reloadability and checkpoint
  metadata before scoring, including during smoke. Missing, smoke, corrupt or
  mixed source runs are rejected.
- Require one training commit per experiment, **not the same commit across both
  experiments**: the downloaded runs were trained at `c3311858...` and
  `3342ae09...`. Record both commits and runtime manifests. This historical
  comparison cannot exclude all implementation or machine/runtime differences.
- Sources are read-only. Do not download training states or replay reservoirs.
  Full states are unnecessary for evaluation. Exact exploitability is not run.

## Protocol

The scorer, rule agents, LBR implementation, deal/action seed schedules,
duplicate-deal protocol and budgets are reused from the existing retrospective
evaluations. Durable task caching is shared with the Experiment 6/7 runner.

| Evaluation | Coverage | Duplicate deal pairs per matchup |
|---|---|---:|
| Five published rule agents | Each of the 24 policies versus every agent | 10,000 |
| Local Best Response | Every policy; 4,096 preflop rollouts | 1,000 |
| Matched-time direct play | Exp9 versus Exp7, same seed, at every checkpoint | 50,000 |
| Temporal direct play | Every later versus earlier checkpoint, within method and seed | 50,000 |
| Approximate node comparison | Exp9 at 12h versus Exp7 at 18h, same seed | 50,000 |

Each duplicate pair plays the same chance deal in both seat allocations.
Production has **2,571 scoring tasks**, including ten-pair LBR shards, and
**3,774,000 duplicate pairs / 7,548,000 hands**, excluding LBR's internal rollouts.
Direct comparisons use matched training-seed labels, not all nine seed pairings,
consistent with the previous evaluation experiments.

The **primary endpoint is 24-hour Exp9-versus-Exp7 direct play**. Earlier
checkpoints, rule/LBR scores, temporal progression and node-based comparisons
provide supporting diagnostics. Positive direct scores favour Experiment 9;
positive temporal scores favour the later policy. Higher rule-agent payoff is
better; lower LBR payoff is better. Paired rule/LBR differences are Exp9 minus Exp7,
so negative LBR differences favour Experiment 9. Values are in mbb/hand.

The secondary node comparison is fixed from throughput metadata before scoring:
Exp9 at 12h averages 85.385 million nodes, versus 78.151 million for Exp7 at 18h.
This is about **9.3% more nodes**, not exact budget matching. Tables retain
per-seed counts and mismatches; no policy interpolation or post-result
checkpoint selection is used. Equal active time is not equal billable VM time:
checkpoint policy fitting, saving and uploads are excluded from the active clock.

Training-seed means are the inferential unit, not hands or LBR shards. Retain
within-match duplicate-deal uncertainty and across-seed standard errors and
Student-t 95% intervals. Compute paired metric differences within seed before
aggregation; temporal run summaries first average the six within-run contrasts.
Three seeds give limited power, and intervals are exploratory, not
multiplicity-adjusted. Matching seed labels does not guarantee identical
trajectories. LBR is approximate, **not exact exploitability**; head-to-head
success alone does not demonstrate convergence to equilibrium.

## GCP smoke and full evaluation

Run from this repository after committing and pushing the evaluation code.
Use the existing FHP `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` variables.
`BUCKET` accepts a bare bucket name or a `gs://...` bucket.

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export EXP7_RUN_ID="exp7-par8-20261001-005151"
export EXP9_RUN_ID="exp9-cache24-20261001-132550"
export RUN_ID="fhp-eval79-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_retrospective_exp7_exp9_evaluation.sh run
```

This submits **one `n2-standard-8` VM**. It downloads the frozen policies, runs
a real-checkpoint **cloud smoke test first**, and starts full evaluation only if
smoke succeeds. Full evaluation uses eight scoring processes, one Torch thread
each. There is no training phase and the laptop can disconnect after submission.
The task exits when scoring and uploads finish. The default **36-hour elapsed
safety limit** includes setup and smoke; it is not a required runtime. Override
with `EVAL_MAX_HOURS` (1--72) when necessary.

```bash
# Optional: generate the job specification without cloud submission.
bash gcp/run_retrospective_exp7_exp9_evaluation.sh dry-run

bash gcp/run_retrospective_exp7_exp9_evaluation.sh status
```

Results are uploaded every five minutes and on exit; resource snapshots and
failure diagnostics are saved alongside them. After a FAILED job, retain the
same `RUN_ID`, `REPO_REF` and source IDs:

```bash
bash gcp/run_retrospective_exp7_exp9_evaluation.sh resume
```

Resume uses a new Batch job name but the same output prefix. Completed tasks
are not rescored. Active concurrent jobs, changed policies, evaluator code,
dependencies or budgets are rejected. An abrupt VM termination may lose tasks
since the last upload. Use a new RUN_ID for a changed experiment.

## Outputs and download

`$BUCKET/$RUN_ID/analysis/` contains:

- Rule-agent/LBR trajectories by time and nodes, with per-seed and aggregate
  tables and paired Exp9-minus-Exp7 metric differences.
- Matched-time, all later-versus-earlier and approximate node-matched
  head-to-head tables. Six charts include a temporal comparison heatmap.
- `analysis_summary.md`, checkpoint provenance, source/evaluator hashes,
  `evaluation_manifest.json` and `SUCCESS.json`.
- Small `task_results/*.json` recovery files; source policies are not re-uploaded.

Download analysis without the recovery cache:

```bash
export BUCKET_ROOT="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${BUCKET_ROOT%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Local verification

```bash
python -m pytest tests/test_retrospective_exp7_exp9_evaluation.py
python -m experiments.fhp.retrospective_exp7_exp9_evaluation.run \
  --exp7-run "cloud_outputs/$EXP7_RUN_ID" --exp9-run "cloud_outputs/$EXP9_RUN_ID" \
  --output-dir /tmp/fhp-eval79-smoke --workers 2 --smoke \
  --rule-deals 4 --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 --crossplay-deals 10
```

Local scoring requires the full `workers/` policy/metadata trees for both runs;
the small downloaded `analysis/` folders alone are insufficient. Cloud execution
downloads these automatically. Even smoke validates all 24 policies, but scores
only seed 0 at 6h and 12h. Smoke is an integration check, never strength evidence.
