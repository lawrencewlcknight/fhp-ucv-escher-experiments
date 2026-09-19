# Experiment 2: lossless structured FHP UCV-ESCHER

Experiment 2 tests whether an FHP-specific neural representation improves
learning and systems efficiency over the raw OpenSpiel input used by Experiment
1. The UCV estimator, grouped soft-target policy objective, two-member critic,
fixed control-variate coefficient, residual calibration, traversal budget, and
six-hour checkpoint schedule remain unchanged.

The deliberate changes are:

- canonical suit labels, which merge only exact poker symmetries;
- exact private/public card channels plus inexpensive rank, suit, hole-card and
  five-card hand features;
- exact two-round betting sequences without universal-poker padding or unused
  bet-size fields;
- a 183-value player information state instead of the generic 190-value tensor;
- one 263-value full-state critic input instead of two duplicated 190-value
  player tensors;
- card/context branches followed by a CPU-friendly MLP;
- float32 replay storage; and
- 128-unit regret/critic/calibration trunks and a 192-unit average-policy trunk.

No hand bucketing is used. The representation retains the complete FHP
information state modulo a global permutation of suit names, which cannot alter
legal actions or payoffs.

## Production contract

- seeds: `0`, `1`, `2`, one independent VM per seed;
- machine: on-demand `n2-standard-8` per seed;
- effective training time: 24 hours per seed;
- checkpoints: first completed outer iteration after 6, 12, 18, and 24 hours;
- exact continuation state at every checkpoint;
- exact exploitability: disabled;
- evaluation: sampled rule-agent, LBR, temporal cross-play, and direct
  Experiment 1 versus Experiment 2 comparisons.

## Local smoke

```bash
./gcp/run_exp2_lossless_structured.sh smoke-local
```

The smoke executes the same four-checkpoint and restore contract using tiny
buffers and one optimizer step.

## GCP Batch

After completing the shared setup in `docs/GCP_BATCH_EXPERIMENTS.md` and pushing
the tested commit:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp2-fhp-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_exp2_lossless_structured.sh run
```

Monitor or resume with the same `RUN_ID` and `REPO_REF`:

```bash
./gcp/run_exp2_lossless_structured.sh status
./gcp/run_exp2_lossless_structured.sh resume
```

The remote controller requires a successful cloud smoke before launching the
three production seeds. Each training task has one automatic retry and restores
the newest continuation state from Cloud Storage.

## Evaluation

```bash
python -m experiments.fhp.exp2_fhp_lossless_structured_ucv.evaluate_checkpoints \
  --source-run cloud_outputs/RUN_ID/workers/TASK_DIRECTORY
```

Every policy checkpoint records the encoder ID, complete layout and structured
model metadata. `fhp_escher.checkpointing.LoadedFHPPolicy` reconstructs both the
encoder and model; a mismatch fails closed rather than silently evaluating the
wrong representation.
