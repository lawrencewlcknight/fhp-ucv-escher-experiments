# Experiment 37 — 24-hour critic-configuration comparison

Evaluate the original deployed **average-policy** checkpoints, not current
regret policies or retrained controls. This is evaluation only; it never resumes
training, modifies a policy or enables the terminal-call safeguard.

| Source | Critic steps per fit | Inputs | Frozen source run |
|---|---:|---|---|
| Experiment 10 | 10,000 | Hand/board baseline | `exp10-features-20261001-161740` |
| Experiment 20 | 5,000 | Same baseline | `exp20-critic24-20261006-164357` |
| Experiment 36 | 2,500 | Richer AE strength/betting features | `exp36-rich2500-20261018-205125` |

Use all three seeds at the original 24-active-hour checkpoint in each run.
Source validation checks configuration, feature encoder, runtime, source commit,
policy hashes, completed-worker receipts and actual critic budgets. A 48h
continuation cannot silently replace the original 24h run. Experiment 33 is not
included: this is the approved three-configuration comparison.

## Endpoints and computational budget

1. **Primary: full cross-seed head-to-head play.** Compare 20–10, 36–10 and
   36–20. Each contrast contains all nine seed pairings, each with 50,000
   duplicated deal pairs (same deal/action randomness with seats exchanged).
   Total: 27 matchups, **1.35 million pairs / 2.7 million hands**. Save each
   5,000-pair shard independently; pooling includes between-shard variation.
2. **Common opponent panel:** five established rule agents, 10,000 duplicated
   pairs per agent per model. Total: 450,000 pairs. Report each opponent and an
   equally weighted panel mean, including individual training-seed results.
3. **Established LBR:** 1,000 duplicated pairs per model, 4,096 preflop rollouts,
   ten-pair saved shards. Total: 9,000 pairs. LBR reports restricted exploiter
   payoff, **not exact exploitability**. This does not launch specialised
   best-response training. The local CLI permits explicitly disabling LBR with
   `--lbr-deals 0`; the production Batch protocol includes it.
4. **Rare-state diagnostics:** known straight-flush and quads examples, plus
   the existing frozen Exp36 assessment-panel generator. Assessment board
   classes exclude recorded earlier strategic audits and Experiment 35, but
   are not claimed unseen in training or any subsequent audit. Report every
   model's probabilities, mean/p95/max conditional BB loss, and severe-fold
   frequency. The explicit paired diagnostic contrasts compare 36 with 20 and
   10; the by-seed table contains all three configurations. These are selected
   terminal call/fold positions, not reach-weighted whole-game losses.
5. **Training context:** nodes, iterations, active time, critic/cache fitting,
   learner and collection times, driver memory, training start/end timestamps
   and VM specification from validated source summaries. Critic fitting is
   inside learner time; cache construction is inside critic fitting. Do not
   add nested timers or equate driver RSS with total VM memory consumption.

Overall match budget: **1,809,000 duplicated pairs / 3,618,000 hands**. The
existing native OpenSpiel match runner, rule agents and LBR implementation are
reused. Independent deterministic streams are assigned by task identity,
including separate LBR sampling streams. There are no shared deal streams
between matchups; common randomness is within duplicated seat pairs.

## Inference and interpretation

Nine pairings do **not** create nine independent training seeds. The primary
estimator equally weights the nine payoff cells for each contrast. Report:

- Cell-level results and conditional deal-noise SEs/intervals for the fixed
  nine-model population.
- A crossed-model bootstrap: independently resample the three models within
  each configuration, reusing each configuration's weights in every contrast.
  This accounts for models appearing in multiple matchups. Do not assume equal
  seed labels imply paired training trajectories.
- 30,000 deterministic bootstrap replicates, with additional independent
  Gaussian cell perturbations using deal-level SEs. Empirical seed effects
  already contain some deal noise, so this addition is a conservative
  sensitivity calculation, not an exact hierarchical sampling distribution.
- Pointwise 95% and Bonferroni-adjusted 98.333% percentile intervals for the
  three primary contrasts. **Only three training seeds are available:** neither
  the bootstrap nor its nominal multiplicity adjustment guarantees calibrated
  coverage. These are exploratory findings, not confirmatory significance
  claims. More hands reduce deal noise, not the uncertainty from too few
  independent training runs.

A provisional head-to-head leader is flagged only if its adjusted intervals
favour it over both alternatives. Cycles and inconclusive comparisons remain
inconclusive. No algorithm is automatically promoted. Review LBR, common-panel
results, rare-state failures and training cost before making a choice.

**Confounding:** 20 versus 10 is the historical fixed-feature critic-budget
comparison. Experiment 36 changes both critic budget and input features.
Its result establishes full-configuration performance, not the isolated effect
of reducing critic steps to 2,500. All comparisons are time-budget matched, not
node matched; they cannot establish convergence or distance to Nash equilibrium.

## GCP Batch: one invocation

After these files have been committed and pushed, from the repository root:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp37-critic-compare-$(date -u +%Y%m%d-%H%M%S)"

bash gcp/run_exp37_critic_budget_evaluation.sh run
```

One `n2-standard-16`, CPU only, 16 evaluation processes, one computation thread
per process. The job installs pinned dependencies, validates sources, runs a
seed-0 smoke with tiny match budgets, then performs the full analysis. The
laptop may disconnect. A separate smoke is available with `smoke-cloud`; use a
new `RUN_ID` for full `run`, or `resume` to advance that same smoke namespace
after the smoke job has finished. `dry-run` generates JSON without submission;
`status` lists matching Batch jobs.

Plan roughly **3–5 hours**, allowing **6–8 hours** for slower execution, plus
queueing. This is an estimate from prior evaluations, not a performance gate.
The default Batch ceiling is 48 hours (override `EXP37_MAX_HOURS`, 1–72).
There is no new training budget and no automatic retraining.

## Recovery and outputs

Each completed match shard and per-model rare-state result is saved atomically.
The dispatcher keeps at most 16 tasks in flight. If one fails, it stops issuing
new work and saves other in-flight successes before reporting failure; it does
not leave hours of queued computation running without collecting its results.
Files are uploaded every five minutes and on clean exit, with resource snapshots
every 30 seconds and failure diagnostics. An abrupt VM loss can lose the newest
not-yet-uploaded work; this is not zero-loss checkpointing. Keep the cloud
`task_results/` even if you omit them from a laptop download.

After a failure, keep the **same RUN_ID, REPO_REF, sources and budgets**, and run:

```bash
bash gcp/run_exp37_critic_budget_evaluation.sh resume
```

This creates a new Batch job name but reuses the original output prefix.
Only identical-policy/code/runtime/protocol results are reused. Checks prevent
overwriting existing outputs or resuming while a matching job remains active.
Do not submit simultaneously from two terminals: the preflight is a check, not
a distributed cloud lock. Protocol mismatches fail before match computation;
completed shards remain available for inspection.

Results under `$BUCKET/$RUN_ID/analysis/`:

- `analysis_summary.md`, `summary.json`, source `checkpoint_index.csv` and
  `training_context.csv`.
- `head_to_head_by_seed_pair.csv`, `head_to_head_summary.csv`, individual-model
  and aggregate rule-agent/LBR tables.
- `seed_pair_payoffs.png`, `head_to_head_intervals.png`,
  `common_opponent_scores.png`.
- `rare_panel.json`, its provenance receipt, `rare_by_seed.csv`,
  `rare_summary.json` and `rare_results/` with individual-position probabilities.
- `evaluation_manifest.json` with immutable fingerprints and attempt timings,
  `task_results/` recovery shards, and `SUCCESS.json` only after completion.

Download the lightweight analysis after success:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Training states/replay are excluded from VM downloads. Original checkpoint
schedules are retained for source integrity validation, but only 24h policies
are played. No changes to the training experiments or thesis are required.
