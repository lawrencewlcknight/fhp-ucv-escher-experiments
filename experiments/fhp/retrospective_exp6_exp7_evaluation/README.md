# FHP Experiments 6 and 7: retrospective policy-quality evaluation

This is an evaluation-only experiment, not another solver-training run. It tests
whether Experiment 7's eight-worker collection improves policy quality per unit
of active training time over Experiment 6's sequential collection. Both source
experiments used `n2-standard-16`, three seeds (0, 1, 2), identical learning
configuration and 24 active training hours. Different collector RNG streams mean
that matching seed labels does not imply identical sampled trajectories.

## Frozen inputs

- Experiment 6: `exp6-vm16-20261001-005140`.
- Experiment 7: `exp7-par8-20261001-005151`.
- All playable grouped average-policy networks at 6, 12, 18 and 24 active hours:
  24 policies in total. Actual checkpoints are completed-iteration crossings.
- Require successful production workers, matching learning configurations and
  training commits, the expected VM/thread/collector setup, checksum-verified
  policy files, and matching checkpoint payload metadata. Wrong games, smoke
  training runs, missing checkpoints and mixed configurations fail before scoring.
- Read source data only. No policy updates, source output changes, full training
  states, replay reservoirs or exact-exploitability calculations are needed.

## Evaluation protocol

Uses the **same vendored evaluation suite, budgets, seed schedule and seat-swapped
duplicate-deal protocol** as the completed Experiments 1–3 retrospective evaluation.

| Test | Coverage | Budget |
|---|---|---:|
| Published rule agents | All 24 policies versus each of five agents | 10,000 duplicate deal pairs per matchup |
| Local Best Response (LBR) | All 24 policies | 1,000 duplicate pairs, 4,096 preflop rollouts |
| Matched-time direct play | Exp7 versus Exp6, same seed, at each of four hours | 50,000 duplicate pairs per matchup |
| Temporal direct play | Each later policy versus every earlier policy within the same method and seed | 50,000 duplicate pairs per matchup |
| Approximately node-matched direct play | Exp7 at 12h versus Exp6 at 18h, same seed | 50,000 duplicate pairs per matchup |

Each duplicate pair plays the same chance deal in both seat allocations. Common
chance/action seed schedules are retained across compared policies. Production
uses 2,571 tasks (including memory-bounded ten-pair LBR shards), 3,774,000 duplicate
pairs / 7,548,000 played hands, excluding internal LBR rollouts.

**Positive direct values favour Experiment 7.** Positive temporal values favour
the later checkpoint. Higher rule-agent payoff is better; lower LBR payoff is
better. All payoff tables use milli-big-blinds per hand (mbb/hand).

The primary comparison is at equal active training time. The secondary node
comparison is prespecified from throughput metadata, before strength evaluation:
Exp7 at 12h averages about 52.0 million nodes; Exp6 at 18h about 49.6 million.
This is only an **approximate match**, not an exact matched-interaction-budget
test. Tables retain each policy's true node count and the relative mismatch.
No policy weights are interpolated, and no claim of identical outputs is implied.

Training-seed means, not hands, LBR shards or temporal cells, are the inferential
unit. Tables retain both within-policy duplicate-deal uncertainty and across-seed
standard errors / Student-t 95% intervals. Paired rule/LBR differences are formed
within seed before aggregation. Temporal summaries first average the six contrasts
within each run. Three seeds give limited power: intervals are exploratory, not
multiplicity-adjusted confirmation. LBR is an approximate best-response diagnostic,
**not exact exploitability**; neither head-to-head wins nor weak LBR exploitation
proves convergence to equilibrium.

## GCP launch (after committing and pushing)

Use the **FHP UCV-ESCHER repository and bucket**, not the SD-CFR bucket. Existing
`PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` variables must identify this project
and its runner account. `BUCKET` accepts either a bare bucket name or `gs://...`.

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export EXP6_RUN_ID="exp6-vm16-20261001-005140"
export EXP7_RUN_ID="exp7-par8-20261001-005151"
export RUN_ID="fhp-eval67-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_retrospective_exp6_exp7_evaluation.sh run
```

The launcher checks the pinned commit, service account and source manifests.
One `n2-standard-8` VM runs a real-checkpoint smoke test followed by the full
evaluation with eight scoring processes (one Torch thread each). Evaluation
does not require the same VM size as training. There is no seed-training phase,
controller or need to keep the laptop connected after successful submission.
The default safety cap is **36 elapsed hours**, not a requested training budget:
the job exits as soon as evaluation completes. Override with `EVAL_MAX_HOURS`
(1–72) if needed. LBR and model-inference costs determine actual runtime.

```bash
bash gcp/run_retrospective_exp6_exp7_evaluation.sh status
```

Each completed task is saved atomically and uploaded every five minutes and on
exit. After a FAILED job, retain **the same RUN_ID, REPO_REF and source IDs**:

```bash
bash gcp/run_retrospective_exp6_exp7_evaluation.sh resume
```

Resume creates a new Batch job name but reuses the evaluation output prefix.
It rejects active concurrent jobs and mismatched policies, evaluator code,
dependencies or evaluation budgets. Completed tasks are not rerun. An abruptly
terminated VM may lose work completed since the last upload. Do not use resume
to append results from a changed implementation; use a new RUN_ID instead.

## Outputs and downloading

`$BUCKET/$RUN_ID/analysis/` contains:

- Rule-agent, LBR, paired metric-difference, temporal, direct and approximately
  node-matched tables, with individual seed results and aggregate uncertainty.
- Six charts: rule-agent and LBR results against time and nodes; matched-time
  Exp7-minus-Exp6 head-to-head; two-panel temporal head-to-head heatmap.
- `analysis_summary.md`, complete checkpoint provenance, protocol/source hashes,
  `evaluation_manifest.json`, and `SUCCESS.json` after all scoring completes.
- Small `task_results/*.json` files for resuming. These are evaluation results,
  not model or training checkpoints. Source policies are not uploaded again.

The cloud output root also contains smoke results and resource diagnostics.
For analysis, omit the per-task recovery cache when downloading:

```bash
export BUCKET_ROOT="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${BUCKET_ROOT%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

For local testing with all playable source checkpoints present:

```bash
python -m experiments.fhp.retrospective_exp6_exp7_evaluation.run \
  --exp6-run "cloud_outputs/$EXP6_RUN_ID" --exp7-run "cloud_outputs/$EXP7_RUN_ID" \
  --output-dir /tmp/fhp-eval67-smoke --workers 2 --smoke \
  --rule-deals 4 --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 --crossplay-deals 10
```

Production validation still checks all 24 checkpoint files during a smoke run.
The smoke budget is solely an integration check; never report it as evidence of
policy quality. Unit tests also exercise corrupted-input rejection, source
provenance, task coverage, resume safety, aggregation and generated Bash syntax.
