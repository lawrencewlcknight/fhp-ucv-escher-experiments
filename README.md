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

## Experiment 1

Experiment 1 transfers the selected UCV-ESCHER architecture and optimisation
settings unchanged to FHP. It targets 15 million training nodes but uses an
internal 11-hour training budget, leaving one hour inside the 12-hour Batch
limit for final checkpoint verification and upload.

The run saves a reloadable average-policy checkpoint after every completed
outer-iteration policy fit. Initial-policy and early node-threshold evaluations
are disabled. Exact tabular exploitability is intentionally not attempted
because enumerating the FHP tree is impractical. The saved policies are designed
for sampled head-to-head evaluation.

Local orchestration smoke test:

```bash
python -m experiments.fhp.ucv_escher_baseline.run --smoke
```

Full run:

```bash
python -m experiments.fhp.ucv_escher_baseline.run
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
JOB_NAME="fhp-escher-exp1-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.ucv_escher_baseline.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 43200 8000 32000 100
```

The Batch cleanup trap uploads outputs even after a failed run. Resource
snapshots are written every minute so the result can distinguish memory limits
from insufficient training throughput.

## Verification

```bash
python -m pytest
python -m ruff check .
```

See [`vr_deep_cfr/UPSTREAM.md`](vr_deep_cfr/UPSTREAM.md) for algorithm-code
provenance and redistribution cautions.
