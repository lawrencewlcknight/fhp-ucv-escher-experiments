# Experiment 23 — paired gameplay with test-time flop re-solving

## Question and frozen design

Does a **500 ms additional public-flop solve** improve the deployed Experiment
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
| Allowance | 0.5 seconds including state/feature preparation, inference, optimisation and verification |
| Trigger | Once at public flop entry, **before either player's first flop action** |
| Continuation | Whole verified range strategy retained for the rest of that hand |
| Primary matchup | Search against its own unchanged blueprint |
| External panel | Five published rule agents; same-label Exp8 48h and Exp9 24h policies |
| Paired controls | Unchanged Exp16 against each external opponent on the identical deals/action seeds |
| Sampling | 5,000 duplicate deal pairs per seed/opponent; both seats |
| Total | 240,000 search hands + 210,000 external-panel control hands |
| Hardware | One `n2-standard-16`, four evaluator processes, four Torch threads each |

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

One job runs the following automatically; no separate Stage A experiment:

1. Reduced-deck independent BR differential and zero-budget simulator tests.
2. Real checkpoint hashes, native scalar/batch/feature-reuse parity, and
   hidden-opponent-card invariance checks.
3. Twenty-four cold-board integration probes per training seed under the same
   four-process allocation as gameplay. At least 10% must produce a changed,
   verified continuation for **each seed**; at most 5% may exceed 500 ms and no
   probe may exceed one second. These are engineering gates, not quality tests.
4. Full paired gameplay and aggregation; report the same timing/acceptance gates
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
`deadline_overrun` denotes the overall 500 ms allowance. Only the latter defines
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
all three seeds under production concurrent load. Start with feasibility mode.

The regression tests additionally check 19,116 full feature rows bitwise against
the native encoder, all 24 suit permutations on representative hands, native
scalar/batched inference, fixed-work solving and independent response recursion.

Both checkouts must contain the new implementation. Preparation is stdlib-only
and uploads a small **source-only, SHA-pinned bundle of both checkouts**, so a stale
`REPO_REF` cannot select the wrong code. Uncommitted source is recorded in the
manifest; review it before submission. No training states are downloaded.

From the UCV repository:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export EXP23_MODE="feasibility"
export RUN_ID="exp23-search-feas-$(date -u '+%Y%m%d-%H%M%S')"

# Review source and job configuration without any cloud calls:
bash gcp/run_exp23_flop_resolving.sh prepare

# Submit that exact prepared bundle:
bash gcp/run_exp23_flop_resolving.sh submit
```

Use the actual existing runner account if yours has another name. It needs
read/write access to this bucket and standard Batch runtime permissions, but
does not create child jobs or alter IAM. `run` combines prepare and submit for
a **new** run ID. `FHP_EVALUATION_SUITE` optionally overrides the sibling path.
The 48-hour Batch cap is a safety limit, not a training duration or runtime
prediction for gameplay mode. The feasibility job has a one-hour safety cap
(including installation/uploads) and a ten-minute cap on the diagnostic Python
phase. Neither mode includes additional training.

**Feasibility mode never starts gameplay**, even if all checks pass. On the same
`n2-standard-16` / four-process / four-thread-per-process allocation, it loads
only the three Exp16 48h checkpoints, checks native correctness, compares legacy
and vectorised preparation on three positions per seed at 16 fixed iterations,
then runs the unchanged 24 timed positions per seed at 500 ms. Legacy/native
feature caches are cleared for the fixed-work comparison, and execution order
is alternated. `preparation_profile.json` contains timings and parity results;
`preflight.json` contains all timed probes. Gate failure retains these outputs
but returns failure, rather than silently declaring a successful experiment.
`FEASIBILITY_SUCCESS.json` means the engineering checks passed, not that gameplay
improvement was established. No `SUCCESS.json` or gameplay shards are generated.

After reviewing a successful cloud feasibility run, start a **new** run ID with
`EXP23_MODE=gameplay` and invoke `run`. That mode repeats the mandatory gates
before paired play. Mode and source are pinned into the prepared bundle: changing
either requires a fresh preparation/run ID. Do not recover the failed original
bundle to obtain these changes. No automatic budget increase is implemented.

Failed infrastructure attempt: once no attempt is active, reuse the original
prepared directory/run ID, set a new `RECOVERY_TAG`, and run the wrapper's
`recover` action. It rechecks gates and resumes complete signed 250-pair shards;
an interrupted shard is rerun. Changed source/configuration requires a new run
ID. Do not use recovery to bypass failed scientific/latency gates.

## Outputs and interpretation

`analysis/manifest.json`, `algebra_checks.json`, `native_checks.json`,
`preflight.json`, `preparation_profile.json` (feasibility mode), `summary.json`,
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

The primary estimate is the mean search-versus-blueprint value (mbb/hand), first
averaged within each duplicate pair and then within each training seed. Report
all three seed estimates and their descriptive t interval. The secondary
estimate is the equally weighted seven-opponent panel's **paired uplift** over
unchanged Exp16, not merely search's raw win rate. Match-level intervals reflect
Monte Carlo uncertainty conditional on saved policies; hands, opponents and
reused checkpoints are not additional independent training runs. Inspect
individual opponents, both seats, latency percentiles and fallback rates.
The summary also reports the fraction of actually visited flop decisions whose
action distribution changed and their mean total-variation distance. Accepting
a continuation that changes only unvisited hands is not useful search coverage.

Winning this fixed panel does not establish reduced whole-game exploitability
or robustness to unseen exploiters. With only three reused training runs, this
is a development comparison, not an independently confirmed publication claim.
Do not attribute a null outcome to a failed search hypothesis if verification or
timing rarely allowed the search policy to be deployed.
