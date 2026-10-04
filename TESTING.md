# Testing

Experiment 17's frozen critic-budget audit:

```bash
python3 -m pytest -q tests/test_exp17_critic_budget.py
bash gcp/run_exp17_critic_budget.sh smoke-local
RUN_EXP17_RAY_TEST=1 python3 -m pytest -q tests/test_exp17_critic_budget.py -k native_parallel
```

Checks include native short/full fit equivalence (including temporal targets),
unaltered full-path optimiser/RNG state, read-only recursive estimator parity
with the actual DFS, seed-level aggregation, source/output isolation, artifact
checksums and generated Batch scripts. The optional Ray test restores all eight
actors and verifies the audited continuation against the untouched learner.

The four experiment runners below are archived but remain runnable regression
and reproducibility references.

Run the unit suite and the end-to-end one-iteration smoke test with:

```bash
python -m pytest
./gcp/run_exp1_grouped_wide.sh smoke-local
./gcp/run_exp2_lossless_structured.sh smoke-local
./gcp/run_exp3_wider_structured.sh smoke-local
python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run --smoke
python -m experiments.fhp.exp2_archived_ucv_escher_sequential.run --smoke
python -m experiments.fhp.exp3_archived_ucv_escher_parallel.run --smoke
python -m experiments.fhp.exp4_archived_ucv_escher_cpu_optimized.run --smoke
```

The suite verifies the exact OpenSpiel FHP parameters, the transferred UCV
configuration, matched Experiment 2/3 contracts, deterministic parallel budget
partitioning, checkpoint reloadability, and absence of obsolete artifacts. Ray
smoke tests require permission to inspect and start local worker processes.

The active Experiment 2 tests additionally verify suit-permutation invariance,
feature dimensions, exact betting-history separation, structured network and
float32 replay construction, GCP task-array generation, and encoder-aware
checkpoint reloadability.

The active Experiment 3 tests freeze the network-width-only comparison, verify
the 1.97x online parameter count, inspect generated three-VM Batch jobs, and
exercise all four checkpoints plus exact continuation-state restore.

GCP Batch smoke-test and full-run commands for active Experiments 1 through 3
and archived Experiments 1 through 4 are documented in
[`docs/GCP_BATCH_EXPERIMENTS.md`](docs/GCP_BATCH_EXPERIMENTS.md). The root
README retains a quick-reference copy of the active submission commands.
