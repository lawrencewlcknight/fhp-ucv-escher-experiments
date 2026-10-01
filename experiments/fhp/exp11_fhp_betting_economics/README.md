# Experiment 11: explicit pot and betting-economics features

**Baseline: FHP Experiment 9**, not Experiment 10. This is an independent test
of additional public betting descriptors. It retains the original exact cards,
suit canonicalisation and betting history. Experiment 10's extra hand/board
features are deliberately absent. No additional states are merged or bucketed.

## Sixteen additive inputs

| Inputs | Definition |
| --- | --- |
| Pot (BB) | Sum of chips already committed by both players, before any outstanding call. |
| Cost to call (BB) | Opponent's total contribution minus this player's, floored at zero. |
| Immediate pot odds | Call cost / (committed pot + call cost); zero if no call is owed. |
| Own/opponent round contributions (2) | Chips committed this round in BB; posted blinds count preflop, contributions reset on the flop. |
| Remaining raises | Raw count from zero to three; posting the big blind does not consume a raise. |
| Position (3) | Button/small-blind seat, acts last this round, next player to act. Seat 0 acts first preflop and last on the flop. |
| Facing a call | Whether this player owes chips to match the opponent. |
| Last action was a raise | Current round only; false before its first action. |
| Last aggressor this round (2) | Self/opponent flags; both zero before any voluntary bet/raise. |
| Last aggressor previous round (2) | Self/opponent flags, retaining preflop initiative on the flop; zero preflop. |
| Opponent just checked | Previous action this round was an opponent check, not a paid call. |

One BB is **100 chips**. Pot and contribution values remain in BB, odds are
fractions, remaining raises are a count, and indicators are binary float32.
The actual initial pot is **1.5 BB**, with **0.5 BB** to call and odds **0.25**.
OpenSpiel's displayed `[Pot: 200]` at that point is not the committed 150-chip
pot; using it would give the wrong denominator. Production features reconstruct
integer chip accounting from the existing public history, without parsing
state strings. A bounded 512-entry history cache reuses the calculation across
card deals. It uses no RNG, equity oracle, exact best response or additional
rollouts. FHP stops after the flop, so there are no assumed future streets.

The player-information input grows **183 -> 199**. The critic's original 263
inputs gain the same descriptors for each player, giving **295**. New features
enter the existing context branch in policy, regret, calibration and critic
networks. The deployed policy never sees opponent-private cards. Retaining the
original prefix makes grouping membership and replay target weights unchanged.

The encoder is independently versioned as `fhp_lossless_betting_economics_v1`.
Earlier encoders and checkpoints remain readable and unchanged. Checkpoint
metadata records feature order, units, conventions and dimensions, and fails
closed on incompatible layouts. As with the original encoder, decision-state
features are the contract; terminal next-state critic values are masked by the
existing `done` flag.

## Frozen experiment design

- Three seeds **0, 1, 2**, each on a separate **n2-standard-16** VM, concurrent.
- **24 active training hours** per seed, ending after the first complete outer
  iteration crossing the boundary; no training-node cap.
- Eight single-threaded Ray traversal actors and eight central learner threads.
  **10,000 total traversals/player/iteration**, not per actor.
- Same cached frozen critic targets, two cross-fitted critics, fixed beta=1,
  four-fit target averaging, disabled predictor and residual calibration.
- Same hidden widths, replay capacities, optimiser settings, learning rates,
  minibatches and update counts. Grouped reset soft-target cross-entropy policy
  fitting still uses 20,000 updates at learning rate 0.003.
- Playable checkpoints at **6, 12, 18 and 24 active hours**; retain analysis,
  diagnostics, evaluation outputs and metadata. Save **one final full resumable
  training state per seed**, no intermediate full states, as in Experiment 9.

Extra input dimensions add first-layer parameters and replay storage (64 bytes
per player-information vector, 128 per full-state vector). They also change
initialisation/RNG consumption. Thus the comparison isolates an additive
feature package, not equal parameter count or bitwise-identical trajectories.
No learning-rate or wider-network intervention is bundled with it.

The hypothesis is improved generalisation and strategically meaningful policy
quality at low preprocessing cost. Compare with Experiment 9 using identical
rule-agent, LBR and duplicate cross-play settings and common evaluation deals.
Report throughput and checkpoint-fitting cost alongside gameplay outcomes.
Historical machine/runtime differences remain. LBR is a lower bound, not exact
exploitability. As in Experiment 9, training does **not** automatically launch
an additional exploiter/head-to-head job; saved policies support that separate
encoder-aware evaluation. Training loss or more nodes alone cannot establish
an improvement in policy quality.

## Run on GCP

After committing and pushing, from this repository root with the usual
`PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` already configured:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP11_RUN_ID="exp11-econ-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP11_RUN_ID"
export PARALLELISM=3
export EXP11_TOTAL_HOURS=24
unset EXP11_SOURCE_RUN_ID
./gcp/run_exp11_betting_economics.sh run
```

The controller runs mandatory cloud smoke, training, then aggregation. The
three training VMs need **48 regional N2 vCPUs**, with **72 active VM-hours**
plus overhead. Completion takes longer than 24 elapsed hours: checkpoint
policy fitting, serialization and uploads are excluded from the active clock
but are still billable. Initial workers have a 36-hour wall limit and zero
automatic training retries. You can disconnect after controller submission.

```bash
RUN_ID="$FHP_EXP11_RUN_ID" ./gcp/run_exp11_betting_economics.sh status
# Same RUN_ID, ref and configuration: recover orchestration/aggregation.
RUN_ID="$FHP_EXP11_RUN_ID" ./gcp/run_exp11_betting_economics.sh resume
```

Playable-only checkpoints cannot resume training. Failure before a final
durable full state requires a fresh run; mid-run full-state recovery is not
provided. The final state retains replay, optimisers, networks, critic-target
history, counters and driver/actor RNGs. Transient critic caches are rebuilt.

## Extend a completed run

Experiment 11 starts fresh: Experiment 9/10 weights have incompatible inputs.
Completed Experiment 11 trajectories can be extended using their original
pinned code/runtime and a new output prefix, for example to 48 cumulative hours:

```bash
export EXP11_SOURCE_RUN_ID="$FHP_EXP11_RUN_ID"
export EXP11_TOTAL_HOURS=48
export RUN_ID="exp11-to48-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp11_betting_economics.sh extend
```

Source configuration, encoder, commit/runtime, seeds and hashes are validated.
Source objects are untouched; earlier playable policies are imported, new
policies are appended at 30/36/42/48 hours, and only the extension's final full
state is retained. Imported full states are temporary and not republished.
Each extension may add 6--48 hours in six-hour increments; wall limit is 72 h.

## Tests and analysis outputs

Tests compare every reachable public decision-state betting sequence with
OpenSpiel's independent chip accounting and actual call/raise legality. They
also cover both player perspectives, round resets, aggression, card-independent
descriptors, cache safety, unchanged old feature prefixes/grouping targets,
all network dimensions, metadata compatibility and cloud-job wiring.

The real eight-worker smoke checks cached/uncached equivalence **within the new
encoder**, policy reload and duplicate play, disk-backed continuation versus
uninterrupted training, final-only retention, and completed-endpoint extension.
Cloud smoke additionally checks production-capacity checkpoint allocation; it
does not prove worst-case fitting memory safety or convergence.

Parallel startup preserves the central learner's post-model-initialisation
Python, NumPy and Torch RNG states (including CUDA when available), so Ray's
port selection and actor startup cannot change its training stream. Actor
seeds are unchanged. The smoke checks identical initial learner/actor states
and RNGs before requiring bitwise-identical cached/uncached training results;
the equivalence checks are not relaxed. Integration tests also inject unequal
startup RNG draws to exercise this protection. This is a reproducibility fix,
not a change to features, targets, sampling rules or training budgets. Existing
jobs remain pinned to their submitted code.
New seeded runs may follow different trajectories from older code because
infrastructure no longer advances the learner's random stream.

```bash
# Optional local validation with the repository dependencies installed:
python -m pytest -q tests/test_betting_economics_features.py \
  tests/test_exp11_betting_economics_experiment.py
./gcp/run_exp11_betting_economics.sh smoke-local
```

Analysis retains the baseline's checkpoint/seed summaries, throughput plot,
parallel timing breakdown, critic-cache counters and continuation provenance.
`analysis/feature_specification.json` additionally records the exact encoder.
To download small analytical outputs without full training states:

```bash
export RUN_ID="$FHP_EXP11_RUN_ID"
FHP_EXP11_BUCKET="${BUCKET%/}"
[[ "$FHP_EXP11_BUCKET" == gs://* ]] || FHP_EXP11_BUCKET="gs://$FHP_EXP11_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_EXP11_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```
