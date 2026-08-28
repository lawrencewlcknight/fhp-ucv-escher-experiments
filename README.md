# FHP ESCHER Architecture Experiments

This repository studies Unbiased Control-Variate ESCHER (UCV-ESCHER) in
two-player flop hold'em poker (FHP). It uses OpenSpiel for the complete game
implementation and preserves the selected UCV-ESCHER training configuration.

## Canonical game

`fhp_escher.game.load_fhp_game()` loads OpenSpiel `universal_poker` with the
VR-Deep FHP contract:

| Setting | Value |
|---|---|
| Players | 2 |
| Betting | Fixed limit |
| Deck | 13 ranks × 4 suits |
| Private cards | 2 per player |
| Public cards | 3-card flop |
| Betting rounds | 2 |
| Blinds | 50 / 100 |
| Raise sizes | 100 / 100 |
| First player | Player 1 / Player 2 |
| Maximum raises | 3 / 3 |

The exact parameter mapping is tested so a silent game-definition change fails
CI. Provenance is recorded in [`fhp_escher/game.py`](fhp_escher/game.py).

## exp1_fhp_ucv_escher_baseline

Experiment 1 transfers the selected UCV-ESCHER architecture and optimisation
settings to FHP and trains by elapsed training time rather than a node budget.
It saves exactly two reloadable average-policy checkpoints: the first at the
first safe trajectory boundary after 6 hours, and the final checkpoint after
12 hours. Policy fitting and checkpoint-writing time is excluded from the
training timer. Training stops immediately after the 12-hour checkpoint.

Initial-policy, early node-threshold, and periodic outer-iteration evaluations
are disabled. Exact tabular exploitability is intentionally not attempted
because enumerating the FHP tree is impractical. The saved policies are designed
for sampled head-to-head evaluation.

Experiment packages, canonical experiment names, output directories, and
numbered checkpoint filenames use an `expN_` prefix (for example,
`exp1_fhp_ucv_escher_baseline`). Google Cloud job IDs use the corresponding
hyphenated `expN-` prefix because underscores are not valid in Batch job names.

Local orchestration smoke test:

```bash
python -m experiments.fhp.exp1_ucv_escher_baseline.run --smoke
```

Full run:

```bash
python -m experiments.fhp.exp1_ucv_escher_baseline.run
```

Outputs include:

- `run_manifest.json`: immutable game, VM, budget, and training configuration;
- `checkpoint_rows.csv`: per-checkpoint losses, estimator diagnostics, nodes,
  and elapsed time;
- `checkpoint_manifest.json`: checkpoint hashes and paths;
- `final_policy_checkpoint.pkl`: stable checkpoint name for later evaluation;
- `summary.json`: throughput, memory use, stop reason, and VM capacity finding.

Reload the final policy:

```python
from fhp_escher.checkpointing import LoadedFHPPolicy
from fhp_escher.game import load_fhp_game

game = load_fhp_game()
policy = LoadedFHPPolicy(game, "outputs/RUN/final_policy_checkpoint.pkl")
```

## Google Cloud Batch

The reference machine remains `n2-standard-8` with 8 vCPUs and 32 GB RAM.
After setting `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL`, submit:

```bash
JOB_NAME="exp1-fhp-escher-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_ucv_escher_baseline.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 50400 8000 32000 100
```

The 14-hour Batch allowance leaves time around the 12 hours of model training
for VM setup, both policy fits, checkpoint verification, teardown, and upload.
The cleanup trap uploads outputs even after a failed run. An independent monitor
writes `resource_snapshots.jsonl` every 15 seconds, including cgroup memory
current/peak/limit values, OOM counters, system memory, disk use, load, and the
largest processes. Compact resource heartbeats also reach Cloud Logging every
minute. On cleanup, `batch_diagnostics.json` preserves the detailed evidence and
`batch_status.json` classifies confirmed cgroup OOM, reported allocator errors,
probable OOM/SIGKILL, timeout/termination, Python exceptions, and other nonzero
exits. The run log also attempts to capture kernel OOM messages when the VM
permits it.

## Verification

```bash
python -m pytest
python -m ruff check .
```

See [`vr_deep_cfr/UPSTREAM.md`](vr_deep_cfr/UPSTREAM.md) for algorithm-code
provenance and redistribution cautions.
