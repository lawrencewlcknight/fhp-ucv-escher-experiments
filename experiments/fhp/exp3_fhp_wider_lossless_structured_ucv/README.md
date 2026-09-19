# Experiment 3: wider lossless structured FHP UCV-ESCHER

Experiment 3 tests whether additional neural-network capacity improves learning
in FHP. It is a controlled comparison with Experiment 2: representation,
estimator, loss functions, optimisers, learning rates, replay, training work,
seeds, hardware, runtime, checkpoints, persistence, and evaluation are
unchanged.

The only intended change is network width:

| Component | Experiment 2 | Experiment 3 |
|---|---:|---:|
| Card/context branches | 64 | 96 |
| Regret trunks | 2 x 128 | 2 x 192 |
| Critic trunks | 2 x 128 | 2 x 192 |
| Calibration trunk | 2 x 128 | 2 x 192 |
| Average-policy trunk | 2 x 192 | 2 x 256 |

Across the two regret networks, two online critic members, average-policy
network, and calibration network, this increases online trainable parameters
from 310,993 to 611,729 (1.97x). Network depth remains unchanged. The lossless
183-value player input, 263-value critic input, canonical suit handling, compact
betting history, and float32 replay all remain exactly as in Experiment 2.

## Production contract

- seeds: `0`, `1`, `2`, one independent VM per seed;
- machine: on-demand `n2-standard-8` per seed;
- effective training time: 24 hours per seed;
- checkpoints: first completed outer iteration after 6, 12, 18, and 24 hours;
- exact continuation state at every checkpoint;
- exact exploitability: disabled;
- evaluation: the same sampled rule-agent, LBR, temporal cross-play, and direct
  comparison methodology used by Experiment 2.

Because the machine and wall-clock budget are fixed, the result measures useful
policy learning per 24 hours, including any traversal or learner-throughput cost
of the wider networks. Compare policy strength together with nodes, completed
outer iterations, learner timing, and resource diagnostics.

## Local smoke

```bash
./gcp/run_exp3_wider_structured.sh smoke-local
```

## GCP Batch

After completing the shared setup in `docs/GCP_BATCH_EXPERIMENTS.md` and pushing
the tested commit:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp3-fhp-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_exp3_wider_structured.sh run
```

Monitor or resume with the same `RUN_ID` and `REPO_REF`:

```bash
./gcp/run_exp3_wider_structured.sh status
./gcp/run_exp3_wider_structured.sh resume
```

The remote controller requires a successful cloud smoke before launching the
three production seeds. Each task has one automatic retry and restores the
newest valid continuation state from Cloud Storage.

## Evaluation

```bash
python -m experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.evaluate_checkpoints \
  --source-run cloud_outputs/RUN_ID/workers/TASK_DIRECTORY
```

Every checkpoint records its encoder and structured-model metadata. The loader
reconstructs the wider model from that metadata and fails on incompatible
layouts or weights.
