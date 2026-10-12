# Experiments 29–33: additive strategic features

These five independently runnable experiments retain Experiment 10's exact
canonical inputs and hand/board descriptors. They **do not retrain Experiment
10**. Each trains new policies for 24 active hours on seeds 0, 1 and 2, then
automatically evaluates against the archived Experiment 10 24h policies.

Experiment 28 is a separate rare-state audit; its implementation is untouched.

| Experiment | Package added to Experiment 10 | Policy / critic input size | Launcher in `gcp/` |
|---|---|---:|---|
| 29 | A: exact uniform-range relative strength | 219 / 335 | `run_exp29_relative_strength.sh` |
| 30 | A+B: relative strength and opponent-class profiles | 249 / 395 | `run_exp30_opponent_classes.sh` |
| 31 | A+C: relative strength and card-removal descriptors | 241 / 379 | `run_exp31_blocker_features.sh` |
| 32 | D: preflop distribution of eventual FHP equity | 226 / 349 | `run_exp32_preflop_distribution.sh` |
| 33 | A+E: relative strength and betting interactions | 231 / 359 | `run_exp33_strength_betting.sh` |

B, C and E build on A, not on one another. D is tested separately. Comparisons
between these new experiments are deliberately deferred. Thus comparison with
Experiment 10 tests each complete package; it does not isolate B, C or E from A.

## Frozen learning and compute contract

- Baseline: `exp10-features-20261001-161740`, source commit
  `e66d4da515eb212e5026a965ac5c39c86144c901`.
- Three separate `n2-standard-16` VMs, one seed per VM, eight Ray collection
  actors per learner, eight fitting threads; no GPU. Full seed concurrency
  requires 48 N2 vCPUs, plus the small controller allocation.
- Same trajectories, reservoirs, network hidden widths, sampling, learning
  rates and update budgets as Experiment 10. The code checks that **only the
  feature encoder identifier differs** in the learning configuration.
- In particular: 10,000 traversals per player per iteration, 750 regret updates,
  10,000 updates per critic, 2,000 calibration updates; one-million-row buffer
  capacities. Retain cached frozen critic targets, two critics, fixed beta,
  four-fit critic-target averaging and the grouped 192-by-192 policy fit
  with 20,000 updates at learning rate 0.003.
- Added inputs are supplied to policy, regret, calibration and critic networks.
  Each critic perspective receives its own information-safe descriptors;
  opponent actual cards remain in the original critic-only prefix.
- More input columns necessarily add first-layer parameters and replay bytes.
  Changed input dimensions also change initialisation/RNG consumption. This
  is a learning intervention, not an output-preserving optimisation.
- The 24h clock retains the existing active-time convention. Feature computation
  during collection is charged to it; checkpoint policy fitting, serialization,
  uploads and subsequent evaluation add elapsed time. A checkpoint is saved at
  the first complete iteration crossing each threshold, not mid-iteration.

## Feature definitions and information boundaries

**A — six flop features.** Availability, win/tie/loss fractions against all
1,081 legal opponent holdings, equity = win + tie/2, and an unbeatable flag
(no legal holding beats this hand; ties allowed). Before the flop these are
zero, including availability. They use only the acting player's two cards
and public board. This uniform reference range is not a betting-conditioned
opponent posterior and does not use the opponent's actual dealt cards.

**B — thirty additional flop features.** Ten disjoint, prespecified
board-relative classes each supply equity, legal combination mass and support:
unpaired; pair below the board's highest rank; board-top pair with highest
kicker below jack; board-top pair with highest kicker jack or above; overpair;
two pair; trips; straight; flush; full house or better. A class with no legal
holdings has zero equity/mass/support. Mass-weighted class equities recover A's
overall equity. These are interpretable made-hand classes, not fitted opponent
ranges or a reproduction of learned OCHS clusters. In FHP a flop flush/straight
is a final made hand, not a draw to future community cards.

**C — twenty-two additional flop features.** For each of B's classes, the
fraction of board-compatible combinations removed by the player's own cards,
and a board-support flag. Also include the fraction of public-board nuts
combinations removed and their remaining legal mass. The before-removal
denominator is 1,176 holdings; own cards leave 1,081. Class counts describe
card removal, not a claim that a class always calls, folds or bluffs.

**D — thirteen preflop features.** Availability, mean, standard deviation and
ten equal-width histogram bins of the conditional uniform-range equity over
all possible FHP flops. Each of 169 suit-equivalent starting classes is exactly
enumerated: 19,600 legal flops per physical starting hand, with 1,081 legal
opponent holdings on each flop. Implementation sums over 1,755 public-board
suit orbits (22,100 physical boards) and all board-compatible own holdings;
suit multiplicity is retained. This is **not seven-card hold'em equity** and
uses no Monte Carlo. It is a prior distribution, not the actual unrevealed
board. Features are zero after the flop. The approximately 80KB bundled table
has a recorded checksum; incompatible assets are rejected on policy/state
reload. Generation is reproducible via `python3 -m fhp_escher.preflop_strength`
with a new `--output` path; do not replace an asset used by saved policies.

**E — twelve additional flop features.** Pot and call cost in big blinds,
immediate pot odds, remaining-raise fraction, position, next-to-act and whether
check/call closes the final round; equity-minus-pot-odds and products of equity
with pot odds, closing, remaining raises and position. Closing is validated
against legal OpenSpiel transitions. Off-turn critic perspectives mask the
closing indicator. These are explanatory inputs, not action recommendations;
uniform-range equity minus pot odds is not a generally correct decision rule.

No exact card/history input is removed or merged. Grouped fitting remains
grouping by exact canonical information states, not strength buckets.

## Efficiency and validation

Flop statistics use vectorized five-card ranking and per-rank/per-card prefix
sums. They never build a quadratic hand-by-hand payoff matrix. A process-local,
bounded 64-board cache reuses all perspectives and betting histories on each
public board. Preflop features are lookups, not online enumeration. The cloud
smoke saves `feature_validation/feature_preflight.json`, which compares input
construction time with Experiment 10 on diverse legal states and records cache
memory. This is a component diagnostic, not an end-to-end speed estimate.

The common workflow includes:

1. Validate and reload all twelve archived Experiment 10 policies before
   launching training, then run the encoding correctness/cost gate and real
   eight-actor cloud smoke. No baseline training states are downloaded.
2. Cached-versus-uncached fixed-workload equivalence and disk-backed RNG/state
   restart checks, including extension of a **completed** 24h smoke state to 48h.
3. Production-capacity replay allocation and checkpoint round-trip on the
   target VM, before any full training seed is launched.
4. Three full training seeds, aggregation, then baseline evaluation.

Local tests check independent scalar poker rankings, exact legal combination
counts, all suit permutations on representative cases, opponent-card privacy,
unchanged baseline prefixes, unchanged grouped replay targets, saved-policy
reloads, source checksums, contracts and resumable evaluation. Smoke budgets
are deliberately tiny and do not establish a policy-quality improvement.

## Automatic evaluation

The primary endpoint is 24h candidate versus 24h archived Experiment 10, with
50,000 duplicate deal pairs for each of three matched seed labels, both seats.
Secondary diagnostics use the existing five rule agents (10,000 pairs each)
and LBR (1,000 pairs per policy, 4,096 rollouts, ten-pair resumable shards).
Both candidate and baseline are evaluated with the same protocol and deal
seeds; **re-evaluation is not retraining**. All four saved checkpoints are
validated, but this workflow scores only the 24h endpoint. No new-experiment
opponents are included. LBR is a sampled lower-bound diagnostic, not exact
exploitability; lower exploiter payoff is better.

Outputs include seed-level and aggregate direct play, rule, LBR and metric
differences; manifests, CSVs, JSON and `analysis_summary.md`. Training-seed
intervals use three seeds and are exploratory/pointwise, not corrected across
five packages. Individual-seed and individual-opponent effects matter. Stronger
play against the baseline alone is not an equilibrium guarantee. Compare
throughput as well as strength; matched active time does not mean matched nodes
or equal end-to-end wall time. Historical hardware/source differences remain.

## Output retention and continuation

For each initial seed retain all four playable policy checkpoints (6, 12, 18,
24h), analysis, evaluation results, metadata and **only one full training state**
at 24h. It contains learner networks, optimisers, reservoirs, counters, random
states, target histories and all eight actor states. Large intermediate full
states are not written. This trades mid-run crash recovery for lower storage.

Continuation requires the same experiment, feature schema/asset, training
configuration, source commit and pinned runtime. It imports the completed
full state into a new run, preserves old policies, and saves new policies every
six hours plus one new final full state. Temporary imported full states are
excluded from cloud uploads. An extension can add 6–48 active hours at a time.
The original 24h run remains untouched. Extensions do not automatically invoke
the 24h-only evaluator; arrange a horizon-appropriate evaluation separately.

## Launching after committing and pushing

From this repository root, using the existing authorised runner account:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
# Set SA_EMAIL to the existing FHP runner service account.
export REPO_REF="<full-pushed-commit-SHA>"
export PARALLELISM=3
export FEATURE_TOTAL_HOURS=24
unset FEATURE_SOURCE_RUN_ID
export EVAL_MAX_HOURS=48

# Example: experiment 29. Use the corresponding launcher for 30–33.
export RUN_ID="exp29-features-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp29_relative_strength.sh dry-run
bash gcp/run_exp29_relative_strength.sh run
bash gcp/run_exp29_relative_strength.sh status
```

`run` submits a remote controller; after submission the laptop can be closed.
It gates training on cloud smoke, then runs aggregation and evaluation. No
separate local smoke is required. Do not launch all five three-seed experiments
at once without checking quota and cost: they represent 360 seed-training
hours before overhead/evaluation. `PARALLELISM=1` queues seeds on separate VMs
sequentially without changing each seed's configuration.

To extend a successful seed cohort to cumulative 48h, use its exact original
`REPO_REF`, a new `exp29-*` run ID, `FEATURE_TOTAL_HOURS=48`, and
`FEATURE_SOURCE_RUN_ID=<completed-24h-run-id>`, then invoke the same launcher
with `extend`. Do not use the Experiment 10 run as a continuation source.

`resume` restarts a lost controller and reuses existing successful stages;
it never silently retrains a failed training stage. If evaluation fails, use
`evaluate-resume` with the original run ID and pinned commit to reuse validated
completed evaluation tasks. Active overlapping evaluation writers are rejected.

Training outputs are under `$BUCKET/$RUN_ID/analysis` and `workers/`.
Evaluation outputs are under `$BUCKET/$RUN_ID/evaluation/analysis`. For compact
local results, download both analysis directories, excluding `task_results/`;
playable policies and full states remain in the workers directories for later
evaluation/continuation. Nothing in this implementation deletes existing cloud
objects or launches jobs on its own.
