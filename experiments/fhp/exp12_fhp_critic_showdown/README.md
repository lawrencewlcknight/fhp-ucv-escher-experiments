# Experiment 12: critic-only showdown comparison

**Baseline: FHP Experiment 9.** This is an independent feature ablation, not a
combination with Experiments 10 or 11. The only intended learning change is
three additional full-state critic inputs. Exact cards, suit canonicalisation,
betting history, targets, network hidden widths and optimisation remain intact.

## Feature and information boundary

Once all three flop cards are public, compute both players' exact five-card
hand ranks and append the one-hot vector:

1. Player 0 would win at showdown.
2. The showdown would tie.
3. Player 1 would win at showdown.

The vector is **all zero pre-flop and on partial-board chance states**, so
unavailable information is distinct from a tie. Player order is fixed, not
relative to the acting player. The comparison includes all nine hand classes,
ordered kickers and ace-low straights. It is deterministic, suit-invariant,
float32 and RNG-free. It describes showdown strength, **not who wins after a
fold**, the pot payoff, or an exact action-value target. There is no rollout,
future-card sampling, game-tree enumeration or best-response computation in
feature construction. FHP ends on the flop; this is not a final-outcome feature
for earlier streets in full hold'em.

The versioned encoder is `fhp_lossless_critic_showdown_v1`:

| Input | Experiment 9 | Experiment 12 |
| --- | ---: | ---: |
| Policy and regret | 183 | 183 (bitwise-identical encoding) |
| Information-set calibration | 189 | 189 (unchanged) |
| Full-state critic | 263 | 266 (unchanged 263-value prefix) |

Only the critic's context branch receives these values. Policy, regret and
calibration inputs never receive the actual opponent cards or the comparison.
The two-fold architecture supplies **one held-out critic** per trajectory, so
the existing critic-disagreement scalar entering calibration is identically
zero. Tests check this indirect input path too. Do not extrapolate that guarantee
to a larger critic ensemble without auditing disagreement features again.

Critic-dependent *training targets* can and should change: the control variate
is meant to exploit training-only full-state information. Regret and calibration
inputs remain information-set-based, and policy inference never calls the
full-state encoder. The deployed policy still needs only its own cards and
public information; old checkpoint encoders remain supported.

The hypothesis is that making eventual showdown strength explicit helps the
critic separate it from folding and betting effects, improving the quality of
the variance-reducing control. This neither changes the UCV residual-correction
identity nor guarantees lower variance, stronger play or convergence.

## Frozen run design

- Seeds **0, 1, 2**, each on a separate **n2-standard-16** VM, concurrent.
- **24 active training hours** per seed, ending at a completed outer iteration.
- **Eight single-threaded Ray traversal actors**, eight central Torch fitting
  threads, and sequential critic/calibration fits, exactly as in Experiment 9.
- **10,000 total traversals per player/iteration**, not per actor.
- Fixed beta=1, two cross-fitted critics, four-fit critic-target averaging,
  no predictor, residual-calibrated sampling and per-fit cached critic targets.
- Unchanged replay sizes, learning rates, losses, batches and update counts;
  reset grouped average-policy cross-entropy with 20,000 updates and LR 0.003.
- Playable policies at **6, 12, 18, 24 active hours**, plus all small analytical
  outputs and metadata. **Only one full training state per seed, at the final
  endpoint**; it includes replay, optimisers, critic history and driver/actor RNGs.

The first critic context layer gains three inputs (192 extra weights per critic
at branch width 64). Stored current/next full-state vectors together add 24 bytes
per critic replay row. Other network sizes and policy replay grouping are
unchanged. Different critic initialisations consume different Torch draws, so
identical seed labels do not imply bitwise-identical learning trajectories.
The shared parallel-startup RNG fix is retained; historical Experiment 9 runs
may predate it. Compare selected systems with that provenance caveat.

The cloud controller runs smoke, three training tasks, then aggregation. Three
simultaneous training VMs need **48 regional N2 vCPUs**. Total active VM time is
72 hours; elapsed completion is more than 24 hours because setup, policy fitting,
checkpoint saving and uploads are outside the active training clock. The initial
training wall-clock ceiling is 36 hours, with no automatic training retries.

## Validation

Tests cover named hand classes, kickers, ties, wheels, player swapping and global
suit permutations; 500 random deals are checked against independent OpenSpiel
showdown returns. A stronger hand that folds must still have the showdown-win
indicator. Same-information-set states with different opponent hands must have
identical policy/regret/calibration inputs and grouping targets. Reloaded policy
inference is guarded to fail if it asks for the opponent's private tensor.

The real eight-actor smoke requires bitwise-identical initial and final learning
states for cached/uncached fits, including a production-shaped critic minibatch.
It also checks exact restart continuation, completed-endpoint extension,
final-only state retention, policy reload, duplicate self-play and aggregation.
Tests deliberately disturb startup RNG consumption; strict equality is retained.
Cloud smoke additionally runs the inherited production-capacity checkpoint
allocation test. This is not a proof of worst-case training memory safety.

## Run

The code must first be committed and pushed. From this repository root, retain
the usual `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` settings:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export FHP_EXP12_RUN_ID="exp12-showdown-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$FHP_EXP12_RUN_ID"
export PARALLELISM=3
export EXP12_TOTAL_HOURS=24
unset EXP12_SOURCE_RUN_ID
./gcp/run_exp12_critic_showdown.sh run
```

Once submission succeeds the laptop can disconnect. Check status with:

```bash
RUN_ID="$FHP_EXP12_RUN_ID" ./gcp/run_exp12_critic_showdown.sh status
```

Optional local checks, using the repository Python environment:

```bash
python -m pytest -q tests/test_critic_showdown_features.py \
  tests/test_exp12_critic_showdown_experiment.py
./gcp/run_exp12_critic_showdown.sh smoke-local
```

Set `PYTHON=/path/to/venv/bin/python` for local smoke if needed. The controller's
cloud smoke is mandatory even when local checks pass.

## Outputs and interpretation

Outputs include per-seed checkpoint/loss/estimator tables, memory and cache
diagnostics, `throughput_summary.json`, `critic_cache_summary.csv`,
`feature_specification.json`, `nodes_by_training_time.png` and
`training_time_breakdown.png`. Small analytical files can be downloaded without
the final full training states:

```bash
export RUN_ID="$FHP_EXP12_RUN_ID"
FHP_EXP12_BUCKET="${BUCKET%/}"
[[ "$FHP_EXP12_BUCKET" == gs://* ]] || FHP_EXP12_BUCKET="gs://$FHP_EXP12_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_EXP12_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

As in Experiment 9, training does **not** automatically launch the expensive
strategic evaluation. Compare saved policies with Experiment 9 under the same
rule-agent, restricted-LBR and duplicate cross-play protocol, preferably using
common evaluation deals at each checkpoint. Report all three seeds with
uncertainty, training nodes, active time and fitting overhead. Better critic loss
or throughput alone is not evidence of better play; restricted LBR is not exact
exploitability. The existing encoder-aware checkpoint loader supports these
policies when the evaluation job uses this code revision or newer.

## Continuation and recovery

Use a new run for initial training: Experiment 9 critic weights cannot be
directly resumed into a different input layout. A **completed Experiment 12**
endpoint can be extended with the same pinned code and runtime, for example:

```bash
export EXP12_SOURCE_RUN_ID="$FHP_EXP12_RUN_ID"
export EXP12_TOTAL_HOURS=48
export RUN_ID="exp12-to48-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_exp12_critic_showdown.sh extend
```

Source artifacts are untouched. The extension validates source checksums,
encoder, seed, configuration, commit and runtime, imports earlier playable
policies, and appends 30/36/42/48-hour checkpoints. Only its own final full state
is retained; the imported full state is temporary. Extensions add 6--48 active
hours in six-hour increments and have a 72-hour wall-clock ceiling.

`./gcp/run_exp12_critic_showdown.sh resume` can recover orchestration/aggregation
with the original run ID. It cannot resume interrupted training from a
playable-only intermediate checkpoint, because no intermediate full state is
retained.
