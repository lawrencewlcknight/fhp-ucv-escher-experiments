# Experiment 27 — attacker learning controls and loss comparison

Follow-up to Experiment 26. This tests whether the learned attackers are capable
of exploiting weak targets and whether changing their fitting loss improves
their response to frozen Experiment 10 policies. **No UCV policy is retrained.**

One job runs all permitted stages:

1. Five-minute controls for both objectives: uniform Leduc (with OpenSpiel exact
   best-response benchmarks), fold-prone FHP (80% fold / 20% call), and calling-only
   FHP. Report initial/final strength and exact Leduc gaps. Stop with a completed
   negative report if neither objective meets the predeclared capability criteria.
2. One-hour Exp10 seed-0 comparison: original Smooth L1 versus half-squared-error
   loss, preserving architecture, optimizer, replay, gradient clipping and other
   settings. Four attackers per arm; snapshots at 0, 15 and 60 minutes. Match
   initialisation seeds, not the subsequent policy-dependent replay trajectories.
3. Lock the best eligible objective using development deals. Evaluate each arm's
   final policy and its own initial policy on 10,000 new duplicate pairs. Only a
   locked candidate with positive lower confidence bounds for payoff and learning
   gain proceeds to one-hour replications against Exp10 defender seeds 1 and 2.

Development panels contain 2,000 duplicate pairs per candidate. Replications use
only the selected objective: they test attacker usefulness across defenders, not
a replicated objective treatment effect. A success still requires human review
before any revision/rerun of Experiment 21; its existing gate is untouched.

One `n2-standard-8`, four concurrent two-thread learners, 60-GB disk, **eight-hour
Batch cap**. Approximate fitting time is 30 minutes for controls, 2.5 hours through
seed 0, or 4.5 hours including both gated replications. Allow about 5–6 hours overall
if all stages run; these are estimates. No automatic retries or additional jobs.

Full protocol: `fhp-evaluation-suite/docs/exp27_attacker_development.md` in the
sibling suite. Both local checkouts are bundled; no `REPO_REF` is required.

## Cloud smoke test

From this repository root, assuming `PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL`
are set:

```bash
export RUN_ID="exp27-attack-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp27_attacker_development.sh run-smoke
bash gcp/run_exp27_attacker_development.sh status
```

Smoke deliberately exercises every stage with tiny non-scientific budgets. It
does not qualify attackers or authorise policy repair.

## Full run

After checking smoke success, use a fresh namespace:

```bash
export RUN_ID="exp27-attack-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp27_attacker_development.sh run
bash gcp/run_exp27_attacker_development.sh status
```

`prepare` / `prepare-smoke` are offline previews. `submit` / `submit-smoke` submit
an already prepared bundle. Override the suite location with `FHP_EVAL_REPO` or
the prepared directory with `EXP27_RUN_DIR` only when needed. No separate stage
submissions are necessary; the laptop can disconnect after submission.

## Download results

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

This includes the report, detailed measurements, raw paired results, selection
records, training manifests and costs. Logs and attacker snapshots stay in GCS.
Snapshots omit replay and are not exact training-resume checkpoints. Negative
scientific results receive `SUCCESS.json` with `policy_repair_authorised: false`;
actual execution/correctness failures remain failures. No exact FHP exploitability
or proof of near-Nash play is produced.
