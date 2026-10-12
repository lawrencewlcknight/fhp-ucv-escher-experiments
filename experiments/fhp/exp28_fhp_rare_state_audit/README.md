# Experiment 28 — rare-state coverage and learning-path audit

Investigate the strategically implausible folds identified in the Exp16 audit:
unbeatable straight flushes and quads facing a terminal call/fold decision. The
question is **where the weakness enters the pipeline**, before funding a new
full training run or introducing adversarial repair.

## Design

1. **Historical coverage:** read the complete native Exp10 24-hour and Exp16
   48-hour states for source seeds 0, 1 and 2. Verify manifests, hashes, learning
   configuration, saved policy identity and native runtime. Count exact
   information-state coverage and broader hand-category coverage in policy
   replay; report retained counts, iteration-weighted mass, effective sample
   size and iteration quartiles. Count critic replay separately: it is a recent
   circular buffer, not a lifetime reservoir.
2. **Compare representations:** at the frozen development positions, record
   the deployed average-policy output, empirical weighted replay target where
   present, and current regret-network scores/strategy. Do **not** treat the
   current strategy as the teacher of the historical average. The saved states
   do not contain a complete historical regret-data or visitation log.
3. **Probe the estimator without learning:** at guaranteed-winner terminal
   decisions, enumerate both action outcomes using OpenSpiel and apply the
   *actual native* corrected-regret estimator. Check its exact conditional
   action expectation and variance under the native full-support proposal,
   across 32 compatible hidden hands and both critic folds at up to 24 roots.
   These diagnostic samples never enter training replay. The guaranteed-winner
   classification enumerates all 1,081 compatible opponent hands; it does not
   require a guessed opponent range or access to an opponent's hidden cards
   when choosing the hero action.
4. **Passively instrument a short continuation:** restore each 48-hour state
   into the native eight-Ray-worker trainer and perform ten complete outer
   iterations with the unchanged Experiment 10 learning configuration. Record
   natural visits, generated targets, reservoir arrival acceptance, batch
   survival, retained category counts and actual regret-fitting minibatch
   targets/predictions. Save small regret-model/Adam snapshots. There is no
   average-policy fit or full training-state save in this diagnostic continuation.
   Observation must not consume training RNG draws or supply training examples.
5. **Policy-only intervention on frozen 48-hour replay:** compare the original
   policy against ordinary extra grouped cross-entropy fitting, targeted rare
   group weighting, and an equal-count/random-group weighting control. All
   fitted arms start from identical saved policy weights **and Adam state**, use
   identical replay, architecture and minibatch streams, and save candidates
   after 1,000 and 5,000 extra updates. Two fitting replicas are nested within
   each of the three independent source seeds. No regret or critic training
   happens in these arms.
6. **Assessment:** lock all fixed candidates before evaluating new board suit
   classes, plus 2,000 duplicate-deal pairs per final candidate/opponent against
   the original same-seed policy, uniform play and a fixed Exp9 seed-0 policy.
   Include the unchanged-policy controls on the same deal streams. This uses
   the repository's shared `fhp_evaluation` duplicate-play implementation. It is
   a modest whole-game guardrail, not an exact exploitability measurement or a
   powered non-inferiority claim. No LBR, best-response training or full new
   solver-training campaign is launched.

The job performs the seed-0 pilot first and validates its artifacts before
repeating the frozen protocol for seeds 1 and 2. This gate tests correctness,
not whether the scientific result is positive. No manual intermediate
submissions are needed.

## Target weighting and assessment safeguards

- Target only existing quads/straight-flush replay groups where folding is legal,
  at least two rows remain, effective sample size is at least 1.5, and the
  empirical teacher folds at most 5%. This is a prespecified broad category
  predicate, not selection of the biggest observed model errors.
- Increase their loss mass towards 20%, capped at 10x per target group; never
  downweight them to reach that ceiling. Apply a common global normalization
  to keep the expected loss scale unchanged. The disjoint random control
  matches the number of upweighted groups and **total additional weighted
  mass**. Sampling streams remain the same in every arm.
- If there are no sufficiently supported sensible rare labels, run ordinary
  fitting but explicitly mark targeted/random arms **not identifiable**. Do not
  manufacture labels, inject oracle calls or silently weaken the thresholds.
- New assessment board suit classes exclude every development, reference and
  reserved board from the previous strategic audit. Exclude all replay rows
  from those new assessment classes from every refit arm. Historical training
  could nevertheless have seen them: this is held-out **refitting** assessment,
  not a claim of unseen-from-birth data.
- Both update prefixes and all fixed arms are assessed; there is no
  assessment-selected winner. Report fitting replicas averaged within training
  seed, then paired seed-level differences. Three-seed t intervals are
  descriptive and low-power. Hundreds of correlated board/history positions
  must not be counted as hundreds of independent experimental units.
- Report mean/tail conditional payoff gaps and fold probabilities, exact
  coverage, replay masses and all measured subprocess costs. Conditional
  errors at rare/off-path decisions are not natural whole-game error rates.

## Interpretation

| Observation | Supported next step |
|---|---|
| Supported low-fold replay target, high-fold deployment; targeted refit beats ordinary and random controls | Investigate policy fitting/coverage allocation as the mechanism, subject to whole-game guardrails. |
| Replay itself folds strongly; current regrets also prefer folding | Inspect natural corrected targets and their fit before blaming final-policy compression. |
| Exact estimator expectation is correct but sampled targets are noisy or badly approximated | Investigate sampling variance or regret-network learning; estimator correctness alone does not imply learned regrets are good. |
| Few retained rows, no exact natural trace in the short continuation | Insufficient coverage evidence; consider an explicitly separate sampling/retention intervention. |
| No held-out benefit | No demonstrated benefit from this frozen late-stage repair; do not generalize to all training stages. |

No retained row is **not** proof the state was never visited. An empirical
reservoir target is **not** the exact CFR average. Distinguishing those limits
is part of the result, not a reason to fill in missing evidence.

## Runtime and resource bounds

One `n2-standard-16` (16 vCPUs, 64 GB RAM), 250 GB disk, eight native learner
threads and eight one-thread traversal actors during continuation. Read-only
diagnostics/gameplay use one inference thread. Fresh subprocesses release each
large source state before the next stage; downloaded full states stay in private
VM scratch and are not re-uploaded. The production **safety cap is 24 hours**,
not a promised duration or training allocation. Each continuation has a two-hour
between-iteration safety check and a 150-minute hard subprocess ceiling. A cap
hit produces incomplete/failure artifacts, never a successful partial result.

Source downloads, grouping, diagnostic sampling, fitting, assessment, gameplay
and continuation costs are retained. Resource monitoring records RSS, available
memory, disk and cgroup/OOM evidence, with an independent final-upload runnable.
Automatic retries are disabled to avoid paying twice for an unchanged failure.
Cloud smoke loads both real source states for seed 0 and restores all eight
actors, so it includes substantial download/verification work even though its
learning and gameplay budgets are tiny.

## Run from the repository root

First commit and push this implementation; these jobs clone the **pushed
commit** selected by `REPO_REF`. The local synthetic smoke cannot authorize a
production job. Assume `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` are set.

```bash
# Optional local plumbing check (synthetic, not scientific evidence).
bash gcp/run_exp28_rare_state_audit.sh smoke-local

# After committing and pushing:
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp28-rare-smoke-$(date -u +%Y%m%d-%H%M%S)"
export SMOKE_RUN_ID="$RUN_ID"
bash gcp/run_exp28_rare_state_audit.sh smoke-cloud
```

When `$SMOKE_RUN_ID-smoke` has succeeded:

```bash
# Keep the same REPO_REF and successful SMOKE_RUN_ID.
export RUN_ID="exp28-rare-audit-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp28_rare_state_audit.sh run
bash gcp/run_exp28_rare_state_audit.sh status
```

`dry-run` prepares job JSON without cloud submission or downloads. The launcher
ignores unrelated `PARALLELISM` values. It validates source metadata/availability
before submission, and production requires a checksummed real cloud smoke from
the same code commit.

## Outputs and download

`$BUCKET/$RUN_ID/analysis/` contains `report.md`, `summary.json`, `per_seed.json`,
the frozen protocol/panel, coverage and per-case diagnostics, all refit curves,
estimator probes and continuation traces. Analysis-only packaging omits model
binaries but has its own checksums. Candidate policies, policy Adam states and
regret snapshots are under `$BUCKET/$RUN_ID/audit/workers/`; original multi-GB
source states remain at their unchanged Exp10/Exp16 URIs.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
```

On failure, inspect `$BUCKET/$RUN_ID/diagnostics/` and partial `audit/workers/`.
There is no silent resume/overwrite of partially completed scientific stages:
use a new run ID after correcting the cause. This experiment does not modify
the archived models, Experiment 21 qualification, or the thesis documents.
