# Experiment 23 — paired gameplay with test-time flop re-solving

## Current experiment: expanded two-second gameplay evaluation

The 500 ms cloud feasibility run `exp23-search-feas-20261011-200559` completed
preparation on all 72 probes but accepted only 2/24, 0/24 and 1/24 continuations
for seeds 0, 1 and 2. None overran the overall deadline. It stopped at the
minimum-10%-acceptance gate, not because of a VM or memory failure.

The subsequent run `exp23-search-2s-20261011-210319` passed with 13/24, 12/24 and
9/24 accepted continuations (34/72 overall), about 272 solver iterations per
position, and no two-second deadline overruns. These are feasibility results,
not evidence of stronger gameplay.

The approved follow-up evaluates **two seconds per public-flop-entry solve**
on fresh duplicate deals against the original policy, Exp8 and Exp9. The same
three Exp16 models, worker allocation, solver, 10,000-iteration ceiling,
per-iteration verification and safety tolerances are retained. This is
**evaluation only**: no model is fitted, retrained, replaced or resumed.

`EXP23_MODE=feasibility` is now the launcher's default and never starts gameplay.
Its search deadline is two seconds, its maximum allowed overrun rate is still
5%, and its separate pathological-latency guard scales from one to **four
seconds**. The latter is not extra usable search time: any continuation that
arrives after the two-second search deadline is discarded. Every seed must
still achieve at least 10% acceptance (at least 3 of 24 probes).

Explicit `EXP23_MODE=gameplay` now selects the expanded two-second study below,
under protocol `fhp_exp23_flop_resolving_v2_expanded_2s`. It replaces the former
500 ms/eight-opponent gameplay plan, whose source remains in Git history. It
cannot reuse old gameplay or feasibility output identities. A low acceptance
result is useful development evidence, not proof that search cannot improve
the policy at any compute budget.

## Question and frozen design

Does a **two-second additional public-flop solve** improve the deployed Experiment
16 48-hour policy without retraining? Use its three original training seeds.
No critic, learned verifier, opponent model or opponent private cards enters
the solve. Pre-flop play is unchanged. FHP has no future board cards, allowing
exact terminal utilities and full legal flop betting trees.

The shared implementation lives in `fhp-evaluation-suite/fhp_evaluation/resolving/`.
The local training configuration, saved models and Experiments 1–22 are unchanged.

| Item | Prespecified setting |
| --- | --- |
| Candidate | Exp16, 48 hours, seeds 0/1/2, checkpoint SHA-256 pinned |
| Search | Range-vector regret-matching+, root opponent opt-out gadget |
| Allowance | 2 seconds including state/feature preparation, inference, optimisation and verification |
| Trigger | Once at public flop entry, **before either player's first flop action** |
| Continuation | Whole verified range strategy retained for the rest of that hand |
| Primary matchup | Search against its own unchanged blueprint |
| External panel | Same-label Exp8 48h and Exp9 24h policies; no rule-agent panel in this run |
| Paired controls | Unchanged Exp16 against each external opponent on the identical deals/action seeds |
| Sampling | 10,000 duplicate pairs per seed against own blueprint; 5,000 per seed against each external opponent; both seats |
| Total | 120,000 search hands + 60,000 external-panel control hands = **180,000 hands** |
| Hardware | One `n2-standard-16`, four evaluator processes, four Torch threads each |
| Limits | 22-hour gameplay deadline per attempt; 24-hour Batch cap including setup and uploads |

The primary comparison contains 60,000 hands. Each external comparison contains
30,000 search hands and 30,000 unchanged-policy control hands. At most one search
is requested per searched hand, and none if the hand ends pre-flop. Nominal search
work is 16.7 hours on four workers if every hand reaches the flop, excluding other
costs; this is a projection, not a guarantee. The job exits as soon as the fixed
sample and reports are complete; it does not deliberately run for 24 hours.

An independent exact floating-point best-response recursion checks **every
supported opponent root hand** against the batched blueprint continuation.
Counterfactual reach uses the hero's original pre-flop strategy, not an assumed
opponent policy. Only nontrivially changed candidates with no checked per-hand
increase beyond `1e-8` BB and reduced mean opponent response value are retained.
Otherwise play falls back to the blueprint. The solver does not choose a
continuation using the hero's actual private hand; that hand selects a row only
when an action is requested. No cross-hand solution/policy-table cache is used.

This is a **numerically verified local continuation constraint**, not a claim of
measured whole-game exploitability or an exact-arithmetic safety theorem.
Native scalar/batch inference parity is checked at `3e-6` probability tolerance;
rounding can therefore differ from the mathematical blueprint. No privileged
critic is used to score hypothetical actions. Exact game rules/terminal values
are available at evaluation, unlike the model-free training process.

## Automatic launch gates

Explicit `gameplay` mode runs the following automatically in **one Batch job**.
No separate controller, aggregation job or new training run is needed:

1. Reduced-deck independent BR differential and zero-budget simulator tests.
2. Real checkpoint hashes, native scalar/batch/feature-reuse parity, and
   hidden-opponent-card invariance checks.
3. Twenty-four cold-board integration probes per training seed under the same
   four-process allocation as gameplay. At least 10% must produce a changed,
   verified continuation for **each seed**; at most 5% may exceed two seconds and
   no probe may exceed four seconds. Late results are never deployed. These are
   engineering gates, not quality tests.
4. A short real-policy gameplay smoke across all nine model/opponent matchups:
   two duplicate pairs each, 60 diagnostic hands including paired controls.
   These use a separate RNG stream and are excluded from the final analysis.
   There is no payoff-based pass/fail condition.
5. Full paired gameplay and aggregation; report the same timing/acceptance gates
   on the played flops. No EV-based stopping, model selection or retuning.

Python deadlines are cooperative: a native kernel or OS scheduling can overrun.
**Late results are never deployed**, but fallback cannot undo time already spent.
Actual overruns and their incidence are reported. This does not implement a hard
real-time service. Failed feasibility checks stop the expensive gameplay phase:
near-universal fallback is not a negative result about the value of search.

Exact card features are now constructed for the entire private-hand range with
batched NumPy operations, once per street within a solve, and reused across
betting contexts. Integer suit signatures reproduce the native canonicalisation;
made-hand, kicker and board features retain their exact definitions. One native
representative state supplies the betting context. No feature/model changes,
hand bucketing, opponent-card inputs or cross-hand solution cache are introduced.
The previous per-state implementation is retained as the regression reference.
At fixed iteration counts the implementations must match; at fixed wall time the
faster version can perform more iterations and therefore change the played policy.

Stage timings distinguish hand-space construction, public-tree construction,
card features, neural inference, blueprint verification, solver iterations and
candidate verification. Uninstrumented overhead is included in total latency.
`solver_deadline_overrun` denotes the slightly shorter internal allowance;
`deadline_overrun` denotes the configured overall two-second allowance. Only the latter defines
the reported total-latency overrun rate; either can cause conservative fallback.

## Running on GCP

**Engineering status (11 October 2026):** the original GCP run failed its gate:
0/72 accepted continuations and only 24/72 completed preparations. After batched
encoding, three local genuine seed-0 fixed-work comparisons reduced preparation
from 312--327 ms to 20--31 ms (10--16x), with bitwise-identical reach weights,
blueprint probabilities and retained continuations. A sequential 24-position
seed-0 check completed every preparation, accepted 5/24 continuations, and had
no overall 500 ms overruns. These are local engineering observations, not cloud
throughput or playing-strength results. The unchanged gate must still pass for
all three seeds under production concurrent load. The subsequent optimised
cloud run is the 2/24, 0/24, 1/24 result above: preparation improved by 16--18x,
but the solver performed only about 64 iterations within 500 ms. The two-second
cloud run then passed as described above; the expanded gameplay study tests
whether those verified changes improve realised payoffs.

The regression tests additionally check 19,116 full feature rows bitwise against
the native encoder, all 24 suit permutations on representative hands, native
scalar/batched inference, fixed-work solving and independent response recursion.

Both checkouts must contain the new implementation. Preparation is stdlib-only
and uploads a small **source-only, SHA-pinned bundle of both checkouts**, so a stale
`REPO_REF` cannot select the wrong code. Uncommitted source is recorded in the
manifest; review it before submission. No training states are downloaded.

From the UCV repository, submit the expanded gameplay study with a fresh ID:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export EXP23_MODE="gameplay"
export RUN_ID="exp23-search-full2s-$(date -u '+%Y%m%d-%H%M%S')"

# Review source and job configuration without any cloud calls:
bash gcp/run_exp23_flop_resolving.sh prepare

# Submit that exact prepared bundle:
bash gcp/run_exp23_flop_resolving.sh submit
```

Use the actual existing runner account if yours has another name. It needs
read/write access to this bucket and standard Batch runtime permissions, but
does not create child jobs or alter IAM. `run` combines prepare and submit for
a **new** run ID. `FHP_EVALUATION_SUITE` optionally overrides the sibling path.
The 24-hour Batch cap is a safety limit, not a training duration or runtime
prediction. A 22-hour cooperative gameplay deadline reserves time for setup,
diagnostics, reporting and uploads. The feasibility job has a one-hour safety cap
(including installation/uploads) and a ten-minute cap on the diagnostic Python
phase. Neither mode includes additional training.

Optional `EXP23_MODE=feasibility` with its own new run ID reruns the earlier
two-second diagnostic only; it is not required as a separate job before the
expanded run, which repeats its gates internally. **Feasibility mode never
starts gameplay**, even if all checks pass. On the same
`n2-standard-16` / four-process / four-thread-per-process allocation, it loads
only the three Exp16 48h checkpoints, checks native correctness, compares legacy
and vectorised preparation on three positions per seed at 16 fixed iterations,
then runs the same 24 timed positions per seed at **two seconds**. Legacy/native
feature caches are cleared for the fixed-work comparison, and execution order
is alternated. `preparation_profile.json` contains timings and parity results;
`preflight.json` contains all timed probes and numerical constraint
diagnostics: failed-probe distributions of the best maximum per-hand violation,
iteration/check counts and stage timings. These violations are not exploitability
estimates. `feasibility_gates.json` and `feasibility_report.md` report **every
seed's** gate outcome, even if multiple seeds fail. Gate failure retains these outputs
but returns failure, rather than silently declaring a successful experiment.
`FEASIBILITY_SUCCESS.json` means the engineering checks passed, not that gameplay
improvement was established. No `SUCCESS.json` or gameplay shards are generated.

For a one-command preparation and submission, use
`bash gcp/run_exp23_flop_resolving.sh run` instead of the separate
`prepare`/`submit` commands above. Do not use `run` after already preparing that
run ID.

Mode, timing configuration, per-opponent sample sizes and source are pinned into
the prepared bundle: changes require a fresh preparation/run ID. The launcher
rejects obsolete configurations before cloud submission. Do not recover either
feasibility run or the original 500 ms bundle to obtain this expanded study.
No automatic budget escalation is implemented.

Check status with `bash gcp/run_exp23_flop_resolving.sh status`.

To download the new diagnostic report from the UCV repository root:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Failed infrastructure or exhausted runtime: once no attempt is active, reuse
the **expanded study's** prepared directory/run ID and the same source and mode:

```bash
export EXP23_MODE="gameplay"
export RECOVERY_TAG="retry1"
bash gcp/run_exp23_flop_resolving.sh recover
```

Recovery rechecks the gates and resumes complete signed 250-pair shards; an
interrupted shard is rerun. Each shard contains both seats and all required
controls. There are 240 shards, interleaved across seeds and opponents. Source,
checkpoint hashes, sample sizes, RNG streams and runtime-package versions must
match; changed configuration requires a new run ID. Recovery starts another
explicitly authorised, capped attempt; it never happens automatically. Keep the
remote signed shards if you may need recovery.

Reaching the time budget before the fixed sample finishes is **incomplete**, not
a smaller valid final experiment. `analysis/progress.json` records completed
counts by seed and matchup, and status becomes `budget_exhausted`. No final
summary or success marker is produced. There is no stopping, extension or model
selection based on interim payoffs or significance. Do not use recovery to bypass
failed scientific/latency gates.

## Outputs and interpretation

`analysis/manifest.json`, `algebra_checks.json`, `native_checks.json`,
`preflight.json`, `feasibility_gates.json`, `feasibility_report.md`,
`preparation_profile.json` (feasibility mode), `gameplay_smoke.json`,
`workload.json`, `progress.json`, `gameplay_gates.json`, `summary.json`,
`report.md` and `matchups.csv` (gameplay mode) are compact
analysis outputs. `analysis/task_results/` retains deal-level paired utilities,
latencies, acceptance, fallback reasons and local verification diagnostics.
The selected solver iteration is retained for replay diagnostics. The time cap
makes iteration counts sensitive to CPU scheduling; source/deal pinning does not
promise bitwise-identical search decisions across repeated wall-clock runs.
`status.json` tracks progress; `analysis/SUCCESS.json` requires a complete cohort
and final engineering gates. Sources, package versions and hashes are retained.
Only frozen playable inputs are read; no full training states/new checkpoints
are produced.

Gameplay uses a new protocol-specific deal stream with seed 20261013;
preflight uses the unchanged seed 20261012 and the automatic gameplay smoke
uses 21261013. These sets are separate, and smoke/probe outcomes are never added
to the final payoff sample. Within external matchups, search and control use
identical chance/action seeds; mode execution order alternates by pair index.

The primary estimate is the mean search-versus-blueprint value (mbb/hand), first
averaged within each duplicate pair and then within each training seed. Report
all three seed estimates and their descriptive t interval. The secondary
estimate is the equally weighted two-opponent panel's **paired uplift** over
unchanged Exp16, not merely search's raw win rate. Match-level intervals reflect
Monte Carlo uncertainty conditional on saved policies; hands, opponents and
reused checkpoints are not additional independent training runs. Inspect
individual opponents, both seats, latency percentiles and fallback rates.
The report and CSV include every seed/opponent payoff and its conditional
Monte Carlo interval, including the direct-play primary. Separate equal-weight
frozen-cohort Monte Carlo intervals combine independent match streams using
their standard errors and a normal approximation. They quantify sampling noise
for these fixed models, **not** uncertainty over new training runs. The across-
training-seed t intervals remain separate and descriptive with only three seeds;
secondary opponent contrasts are exploratory. A narrow conditional interval
does not remove between-model uncertainty.

Worker seconds for searched and control gameplay are recorded. Batch duration,
bootstrap logs and resource logs retain setup/diagnostic/upload and interrupted-
attempt costs; the most recent attempt wall time alone is not cumulative cost.
The summary also reports the fraction of actually visited flop decisions whose
action distribution changed and their mean total-variation distance. Accepting
a continuation that changes only unvisited hands is not useful search coverage.

Winning this fixed panel does not establish reduced whole-game exploitability
or robustness to unseen exploiters. With only three reused training runs, this
is a development comparison, not an independently confirmed publication claim.
Do not attribute a null outcome to a failed search hypothesis if verification or
timing rarely allowed the search policy to be deployed.
