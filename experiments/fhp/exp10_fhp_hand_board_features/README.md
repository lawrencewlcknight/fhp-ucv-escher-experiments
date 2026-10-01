# Experiment 10: explicit hand-strength and board-interaction features

**FHP Experiment 9 is the baseline.** The sole intended learning change is an
additive, information-preserving encoder. Exact suit-canonical cards, betting
history and all original context features remain intact. These are descriptors,
not replacement buckets: no additional information states are merged.

## Feature intervention

The versioned `fhp_lossless_hand_board_v2` encoder appends **30 features** to the
original 183-dimensional player-information vector, giving **213 inputs**.
The full-state critic retains its original 263 values and receives the same
30 descriptors for each player, giving **323 inputs**. Descriptors enter the
existing context branch; exact cards remain in the existing card branch.

| Feature group | Meaning |
| --- | --- |
| Showdown ranks (5) | Ordered ranks within the existing nine-way hand category: pair/trips/quad ranks, two-pair ranks, full-house ranks, straight high card, and relevant descending kickers. Missing slots are zero. |
| Private-card ranks (3) | High rank, low rank and rank gap. Available pre-flop as well as on the flop. |
| Board ranks and texture (9) | Three ordered board ranks; unpaired/one-pair/trips; rainbow/two-tone/monotone. |
| Pocket-pair position (4) | Above all board ranks, below all, between board ranks, or matching a board rank. |
| Hole/board interaction (5) | Matches to highest, middle distinct, or lowest board rank; fraction of hole cards above the board; fraction matching the board. |
| Suit concentration (1) | Largest suit count among the five available cards, divided by five. |
| Straight structure (3) | Maximum distinct ranks covered by a legal five-rank straight window for board/combined cards, and longest combined contiguous rank run. |

All descriptors are deterministic, suit-invariant bounded float32 values.
Ranks use rank/14 (deuce=2, ace=14); a five-high straight uses 5, not 14.
The wheel is supported without allowing spurious wraparound straights.
Made-hand and board features are zero until all three flop cards are present.
FHP ends on the flop: **these are not turn/river draw probabilities**.
On a paired board, the matching flags describe rank relations, not mutually
exclusive hand categories; a trips board may activate both highest and lowest.

Policy, regret and calibration inputs use only the player's own cards and
public information. Critics already observe the full training state and gain
the per-player descriptors. No best response, equity oracle, additional
simulation or opponent-private information is supplied to the deployed policy.

The hypothesis is better generalisation across strategically similar hands,
motivated by Experiment 4's failure to improve held-out accuracy through closer
replay fitting alone. Better performance is not assumed. Grouping membership,
replay targets and objective weights are unchanged, since every extra value is
determined by the original input.

## Frozen design and comparison

- Three seeds **0, 1, 2**, concurrent on separate **n2-standard-16** VMs.
- **24 active training hours** per seed; finish the first complete outer
  iteration crossing the budget, without a node cap.
- Eight single-threaded Ray traversal actors and eight central learner Torch
  threads per seed, with **10,000 total traversals/player/iteration**, not per actor.
- Experiment 9's fixed beta=1, two critics, four-fit critic-target averaging,
  disabled predictor, residual calibration and **cached frozen critic targets**.
- Unchanged hidden widths, optimisers, learning rates, replay capacities,
  sampling, minibatches and update counts. Average-policy fitting remains
  grouped, reset soft-target cross-entropy (20,000 updates, learning rate 0.003).
- Playable checkpoints at **6, 12, 18 and 24 active hours**, all analysis,
  evaluation results and metadata retained. **One final full resumable training
  state per seed**, no intermediate full states, as in Experiment 9.

More input dimensions necessarily add first-layer parameters and float32 replay
storage (120 extra bytes per policy-feature row, 240 per full-state feature
vector, before other fields). Initialisation consumes a different number of
random values. Thus this is a feature-package comparison, not an expectation of
identical Experiment 9/10 trajectories. Existing experiments still use the
original encoder by default; old checkpoints remain readable.

Compare saved policies with Experiment 9 using the same FHP rule-agent, LBR and
duplicate cross-play protocol, ideally with common evaluation deals. LBR is
not exact exploitability. Report policy quality together with throughput and
fitting/evaluation overhead; identical nominal budgets do not eliminate
historical machine/runtime differences. No extra exploiter/head-to-head Batch
job is launched by this training controller. Encoder-aware checkpoint loading
supports the new policies; strategic improvement requires that follow-up
evaluation and cannot be inferred from the throughput plots.

## Launch

After this code has been committed and pushed, from the FHP repository root,
reuse the working `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP10_RUN_ID="exp10-features-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP10_RUN_ID"
export PARALLELISM=3
export EXP10_TOTAL_HOURS=24
unset EXP10_SOURCE_RUN_ID
./gcp/run_exp10_hand_board_features.sh run
```

The controller runs mandatory cloud smoke, three parallel training workers,
then aggregation. The three workers need **48 regional N2 vCPUs**. Expect 72
active VM-hours plus setup, checkpoint fitting, serialization, upload and
aggregation overhead. Initial workers have a 36-hour wall-clock ceiling and
zero automatic training retries. Active training excludes checkpoint fitting,
persistence and uploads; those costs are still billable and recorded.

```bash
./gcp/run_exp10_hand_board_features.sh status
# Recover orchestration/aggregation using the same run, ref and configuration:
./gcp/run_exp10_hand_board_features.sh resume
```

`resume` cannot recover unfinished training from a playable-only checkpoint.
There is no mid-run full-state recovery point. A durable final state includes
learner and actor RNGs, replay, optimisers, networks and critic-target history;
per-fit target caches are rebuilt, not saved.

## Continue a completed Experiment 10 run

This is a new training run, **not a continuation of Experiment 9**: old input
dimensions and weights cannot be resumed as the new encoder. A completed
Experiment 10 run can itself be continued using its original pinned code and
runtime, with a new output prefix. For example, 24 additional active hours:

```bash
export EXP10_SOURCE_RUN_ID="$FHP_EXP10_RUN_ID"
export EXP10_TOTAL_HOURS=48
export RUN_ID="exp10-to48-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp10_hand_board_features.sh extend
```

Configuration, source commit, runtime, encoder metadata, seed and checkpoint
hashes are checked. Source results are untouched. Earlier playable policies
are imported and new ones appended at 30, 36, 42 and 48 hours. The extension
retains only its final full state. Temporary imported training states are not
published. Extensions may add 6--48 hours in six-hour increments and have a
72-hour wall-clock ceiling.

## Validation and outputs

Tests cover all hand categories, kicker ordering, wheel handling, board
relations, global suit permutations, agreement with OpenSpiel showdown returns,
exact preservation of the old feature prefix and grouped replay targets,
old/new checkpoint metadata, cloud contracts and continuation integrity.
The eight-actor smoke checks cached versus uncached learning with the **same new
encoder**, disk-backed continuation versus uninterrupted training, final-only
state retention, policy reload and aggregation. It does not require the feature
intervention to reproduce the old encoder's learning trajectory. Cloud smoke
also runs the production-capacity replay/checkpoint allocation test; this is
not a proof of worst-case training memory safety or convergence.

```bash
# Optional local checks, with the repository dependencies installed:
python -m pytest -q tests/test_hand_board_features.py \
  tests/test_exp10_hand_board_experiment.py
./gcp/run_exp10_hand_board_features.sh smoke-local
```

Analysis retains Experiment 9's checkpoint/seed summaries, nodes-by-time plot,
parallel timing breakdown, cache counters and continuation provenance. It adds
`analysis/feature_specification.json` documenting exact feature ordering and
normalisation. Checkpoints include the new encoder version for replayable
policy evaluation. Download the small analysis folder without full states:

```bash
export RUN_ID="$FHP_EXP10_RUN_ID"
FHP_EXP10_BUCKET="${BUCKET%/}"
[[ "$FHP_EXP10_BUCKET" == gs://* ]] || FHP_EXP10_BUCKET="gs://$FHP_EXP10_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_EXP10_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```
