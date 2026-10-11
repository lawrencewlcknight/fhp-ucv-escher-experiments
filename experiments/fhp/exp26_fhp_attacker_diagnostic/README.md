# Experiment 26 — attacker correctness and learning curves

Follow-up to the failed scientific qualification in Experiment 21. This is
**not policy repair** and does not relax the original gate.

Use the frozen Experiment 10 **seed-0 24-hour policy**. First check OpenSpiel
reward signs, terminal/illegal-action handling, actor-only features and
native/collection inference agreement. Then train two Double-DQN attackers per
seat, unchanged from Exp21, with snapshots at **0, 5, 15, 30 and 60 active
minutes**. Keep the original 15-minute exploration horizon during extension;
do not retrain or modify the UCV policy.

Two thousand duplicate pairs per repetition/snapshot supply development curves.
Lock the best 15- and 60-minute repetitions before evaluating four fixed
candidates on 10,000 **new** paired deals each. Report gains, uncertainty,
per-seat/street actions and Q calibration. Main evaluation totals 120,000 hands
plus small correctness/calibration samples. One defender seed means development
evidence, not a general exploitability claim.

One `n2-standard-8` executes checks, four concurrent two-thread attacker runs,
assessment and reporting. Allow approximately 1.5–3 hours as a planning estimate;
the full job's hard safety cap is six hours. No automatic retries, follow-up
jobs or resume. Snapshots preserve playable attackers and optimizer metadata,
**not transition replay**, so do not support exact interrupted-training resume.

The full protocol and implementation live in the sibling evaluation suite:
`docs/exp26_attacker_diagnostic.md` and `fhp_evaluation/adversarial/diagnostic.py`.
Both local checkouts are source-bundled; `REPO_REF` and `SMOKE_RUN_ID` are unused.
Source preparation downloads only the small playable policy, not the historical
5.4-GB training state. No Exp21 outputs are overwritten or reused as final tests.

## Cloud execution smoke

From this repository root with `PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL` set:

```bash
export RUN_ID="exp26-attack-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp26_attacker_diagnostic.sh run-smoke
bash gcp/run_exp26_attacker_diagnostic.sh status
```

The smoke uses tiny budgets on the real pinned source and is execution-only.

## Full diagnostic

After reviewing smoke success, use a new ID:

```bash
export RUN_ID="exp26-attack-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp26_attacker_diagnostic.sh run
bash gcp/run_exp26_attacker_diagnostic.sh status
```

All correctness checks run again before full training. `prepare` is an offline
preview; `submit` submits that exact bundle. The `-smoke` suffix also applies
to those actions. Use `EXP26_RUN_DIR` only for a deliberate custom prepared
directory; `FHP_EVAL_REPO` can identify a nonstandard evaluation-suite location.

## Results

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

`analysis/report.md` summarises all curves and independent final comparisons;
`summary.json` includes raw paired payoffs, action diagnostics, all training
manifests and timing/resource measurements. Detailed logs and attacker snapshots
stay under `preflight/`, `train/`, `assessment/` in GCS.

A completed negative diagnostic is `SUCCEEDED`, with
`policy_repair_authorised: false`; it must not unlock Experiment 21. Correctness,
resource or provenance failures remain actual job failures. Preserve both
positive and negative outcomes and review before choosing the next experiment.
