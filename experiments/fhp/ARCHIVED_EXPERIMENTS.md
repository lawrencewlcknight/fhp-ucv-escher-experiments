# Archived FHP experiments

Experiments 1–4 are frozen historical references. They remain importable and
runnable so future experiments can reuse their game contract, checkpointing,
sequential baseline, Ray backend, replay implementation, and CPU-efficiency
work.

| Experiment | Archived package | Purpose |
|---|---|---|
| 1 | `exp1_archived_ucv_escher_baseline` | Direct transfer of the selected predecessor UCV-ESCHER configuration to FHP |
| 2 | `exp2_archived_ucv_escher_sequential` | Sequential comparison arm on the large reference VM |
| 3 | `exp3_archived_ucv_escher_parallel` | Synchronous 12-worker Ray comparison arm |
| 4 | `exp4_archived_ucv_escher_cpu_optimized` | 28-worker CPU-optimised parallel implementation |

Their numeric `EXPERIMENT_ID` values remain 1–4 for compatibility with saved
manifests and evaluation tooling. New research should use the next experiment
number and should not mutate these archived definitions. Shared solver modules
may be imported or refactored when useful, provided archived regression tests
continue to pass.
