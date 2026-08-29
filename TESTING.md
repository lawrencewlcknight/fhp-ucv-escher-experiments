# Testing

Run the unit suite and the end-to-end one-iteration smoke test with:

```bash
python -m pytest
python -m experiments.fhp.exp1_ucv_escher_baseline.run --smoke
python -m experiments.fhp.exp2_ucv_escher_sequential.run --smoke
python -m experiments.fhp.exp3_ucv_escher_parallel.run --smoke
python -m experiments.fhp.exp4_ucv_escher_cpu_optimized.run --smoke
```

The suite verifies the exact OpenSpiel FHP parameters, the transferred UCV
configuration, matched Experiment 2/3 contracts, deterministic parallel budget
partitioning, checkpoint reloadability, and absence of obsolete artifacts. Ray
smoke tests require permission to inspect and start local worker processes.

GCP Batch smoke-test and full-run commands for Experiments 1 through 4 are
documented in [`docs/GCP_BATCH_EXPERIMENTS.md`](docs/GCP_BATCH_EXPERIMENTS.md).
The root README retains a quick-reference copy of the submission commands.
