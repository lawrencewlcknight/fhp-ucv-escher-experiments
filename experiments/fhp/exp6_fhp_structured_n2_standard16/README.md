# Experiment 6: Experiment 2 configuration on n2-standard-16

## Purpose

Train the selected lossless structured FHP learner from scratch on a larger VM,
under a dedicated experiment identity. This is the sequential baseline for a
subsequent same-machine parallel-traversal comparison, not the parallel
experiment itself.

Only the training/smoke VM allocation and experiment identifiers change.
`EXPERIMENT_CONFIG` is an independent deep copy of Experiment 2's complete
configuration, with strict equality validation. Seeds, checkpoint schedule,
replay precision/capacity, losses, learning rates, network sizes, reset rules,
total traversals and update counts are unchanged. Critic-target caching from
Experiment 5 is **not enabled**.

## Frozen configuration

- Experiment: `exp6_fhp_structured_n2_standard16`.
- Policy identifier: `lossless_structured_ucv_escher_n2_16`.
- Seeds: 0, 1, 2, one independent on-demand VM per seed; three concurrent by default.
- Training/smoke machine: n2-standard-16, 16 vCPUs / 64 GiB; Batch requests
  16,000 CPU-milli and 62,000 MiB, leaving OS headroom.
- Production disk: 200 GiB pd-balanced, as in Experiment 2.
- Learner: sequential single-collector implementation, eight Torch intra-op
  threads, matching Experiment 2's cloud setting. Local/cloud smoke uses one
  thread and tiny buffers/optimizer budgets.
- 10,000 total traversals per player per iteration; 750 regret updates,
  10,000 critic updates per member and 2,000 calibration updates, as before.
- Two critics, fixed beta=1, no predictive contribution, four-fit averaged
  critic targets; unchanged lossless structured feature encoder.
- Average policy: grouped soft-target cross-entropy, reset fits, shared
  structured network with a 192x192 trunk, 20,000 updates per checkpoint.
- All replay capacities and minibatches remain those of Experiment 2.
- Training time: 24 active hours, completing the outer iteration crossing the
  final boundary. No node cap; the inherited million-iteration ceiling is a
  safety limit, not the intended stopping condition.
- Playable policy checkpoints at 6, 12, 18 and 24 active hours; no full training states.
- Exact exploitability is not computed on FHP.

The active clock pauses for checkpoint average-policy fitting, diagnostics,
serialization and uploads. VM setup and these operations still incur compute
cost. The run is not a hard 24-hour VM lifetime.

## Execution

Use the root README's `gcp/run_exp6_structured_n2_standard16.sh` commands.
The usual project, region, bucket, existing service account and a pushed Git
commit are required. No Experiment 2 data download is needed: this is a fresh
training run, not checkpoint continuation from Experiment 2.

The remote controller runs a cloud smoke, then the seed array, then aggregation.
Training/smoke use n2-standard-16; the controller remains e2-small and lightweight
aggregation n2-standard-8. These smaller auxiliary VMs do not affect training.
The service account needs the same child-job and storage permissions as
Experiment 2; the launcher checks account existence and the local pinned commit
before submission. Permissions are not silently modified.

`PARALLELISM=3` means three concurrent **seeds**, requiring 48 available regional
N2 vCPUs. It does not mean three traversal workers. Use 1 or 2 for a quota-limited
staggered run. The laptop can disconnect after the controller is submitted.

The smoke trains tiny buffers through all four time-checkpoint hooks, reloads a
playable policy, reuses verified completed outputs without extra training, and creates
the analysis. This checks output plumbing, not production-scale memory
or throughput. The inherited raw-input Experiment 1 stress subcommand is not
used: this experiment exercises the structured learner's own checkpoint path.

## Recovery and safety

No replay/optimizer continuation states are retained. `resume` can recover
orchestration/aggregation using verified completed workers, but an interrupted
training worker must restart under a new RUN_ID. The worker refuses to mix
fresh training with earlier partial outputs. Keep the existing RUN_ID and
REPO_REF only when recovering completed-worker orchestration/aggregation.
A separate runtime manifest records the VM contract, visible CPUs, thread counts
and Python/NumPy/Torch versions; mismatched resume runtime versions are rejected.

Production tasks have a 36-hour wall-time limit per attempt and **no automatic
training retries**. Controller limits/retries retain the existing
Experiment 2 orchestration settings. No old run is overwritten. Do not start
two controllers against the same output prefix.

## Results and interpretation

Each worker writes playable checkpoints (no full continuation states),
`checkpoint_rows.csv`, `checkpoint_manifest.json`, `run_manifest.json`,
`runtime_manifest.json`, `summary.json` and `SUCCESS.json`.
The cloud wrapper also saves resource and failure diagnostics.

The small `analysis/` folder contains:

- `experiment_manifest.json`: experiment identity, VM/thread contract and provenance.
- `seed_summaries.csv`: endpoint nodes, iterations, active time and peak process RSS.
- `checkpoint_index.csv`: checkpoint coordinates, paths and checksums.
- `throughput_summary.json`: per-checkpoint mean nodes/actual time, endpoint
  throughput, standard errors across seeds and memory summary.
- `nodes_by_training_time.png`: actual active training time versus nodes;
  individual seeds and mean with one standard error.
- `SUCCESS.json`: aggregation completion marker.

More cores do not automatically accelerate sequential collection. Eight fitting
threads remain fixed so the main comparison does not silently combine a VM
change with a thread-count tuning experiment. Any uplift must be measured.
The plot establishes training throughput, not strategic improvement or
convergence. Historical Experiment 2 comparisons are retrospective because the
current code includes intervening semantics-preserving efficiency changes;
a same-commit old-VM rerun would be required for a hardware-only causal claim.

No extra policy-quality evaluation is launched automatically. The versioned
structured checkpoints remain compatible with
`fhp_escher.checkpointing.LoadedFHPPolicy` and Experiment 2's encoder-aware
evaluation entry point. Use the same rule-agent/LBR/direct-play protocol when
comparing policy quality, recognising that none is exact FHP exploitability.
