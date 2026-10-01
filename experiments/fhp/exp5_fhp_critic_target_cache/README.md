# FHP Experiment 5: frozen critic-target cache

## Question and source

Does computing frozen TD targets once per critic-fitting block reduce fitting
time without changing the algorithm's updates?

Use FHP Experiment 2's final 24-hour states, seeds **0, 1, 2**, from
`exp2-fhp-20260921-093839`. The paired arms start from the same archived online critic,
Adam state, four-fit target history, target critic, frozen regret networks and
occupied critic replay rows. The source learner uses fixed beta=1, two critics,
no instantaneous predictor and a four-fit averaged critic target.
The lossless structured encoder and its network sizes remain those of FHP Experiment 2.
The average-policy network is neither changed nor refitted.

No new trajectories are generated in production. This is an isolated
**critic-fitting efficiency experiment**, not a new convergence experiment.
The final archived state is a reproducible late-training workload; it need
not describe early training's speedup. The synthetic local smoke does generate
a few trajectories to exercise real encodings and solver classes.

## Paired treatment

- **Recompute:** unmodified existing `train_model` path.
- **Cache:** at the beginning of every `train_model` call, evaluate the same
  TD formula in chunks over all occupied replay rows. Keep one float32 scalar
  target per row. Each optimizer step uses the original sampler and gathers
  the matching cached targets, materialising only current histories/actions.
- Two critic folds; 10,000 optimizer steps per fold; minibatch 2,048.
- Three repeated pairs per source seed, restoring the initial fitting state
  before each arm. Alternate arm order across seed/repeat combinations.
- Identical new, deterministic sampling seed within each pair. This controls
  a prospective fitting block; it is not a reconstruction of the source run's
  historical minibatch sequence.
- Eight Torch CPU threads; one source seed per on-demand n2-standard-8 VM;
  three VMs in parallel. Paired arms are sequential on the same hardware.

The cache is local to one call and is discarded before the next fit. No cache
is shared between folds, iterations or changes to the regret/target networks.
The source model's parameters, gates, iteration and replay must remain frozen
during that call. Batch-dependent/stochastic target layers (batch norm or
dropout) are rejected. The final partial cache chunk is padded without
consuming RNG, keeping the production forward-batch shape.

Only an explicit `member.cache_frozen_targets = True` enables caching.
Existing training configurations stay on the original path. The existing
post-fit target synchronization and four-fit parameter averaging still run.

## Measurement and correctness

Measure wall time around the actual critic trainer, including cache
construction and post-fit target averaging. Source download/loading,
state restoration, target audit and equal kernel warm-up are outside both
timed arms. They still consume billed VM time. The scalar cache costs four
bytes per occupied row (about 2 MB for 500,000 rows in one fold), plus bounded
forward-pass temporaries. Record cache bytes/build seconds and peak process
RSS; the latter includes loading and is **not an arm-specific memory delta**.

Before timing, build the cache without consuming the sampling RNG and compare
an independently selected minibatch with direct target recomputation. After
each arm, compare online weights, averaged target snapshot, Adam tensors and
fixed-probe predictions at atol=1e-6, rtol=1e-5. Record bitwise equality
separately, require the same final RNG state and exactly one target-version
increment. Finite-value checks apply throughout. Local tests also compare
sampled batches directly, check terminal rewards, partial chunks and cache
refresh after a changed target network.

A four-update **real-source cloud smoke** must pass before full workers launch.
The local smoke is a fast plumbing/correctness test, not representative timing:
building a full cache can be slower for such a tiny update budget.
Production retains failed equivalence flags for diagnosis rather than hiding
them. Do not enable caching in normal training unless **every numerical gate
passes** and the observed timing improvement is worthwhile. Passing tolerance
does not imply bit-for-bit reproducibility; the reports distinguish these.

For each source seed, sum the two fold times within each repeat and take the
median of its three repeated two-critic timings. Report the per-seed ratio
recompute/cache, time reduction, and mean speedup with standard deviation,
standard error and a descriptive 95% t interval across the three source seeds.
Folds and timing repeats are not independent training-seed replicates.
End-to-end training speedup depends on the fraction of training spent fitting
critics; do not multiply these component ratios into a convergence claim.

## Execution, provenance and recovery

Use the root README's `gcp/run_frozen_critic_target_efficiency.sh` commands.
`run` launches a remote controller, smoke, three workers and aggregation.
`dry-run` writes job specifications without GCP submission. `smoke-cloud`
runs only the source-state smoke. There is no new service-account type:
reuse the existing runner with source read/output write, log-writing,
Batch child-job creation/view permissions and permission to act as itself.
Local preflight requires the account, pinned commit and all three completed
source states to exist. IAM failures surface when submitting child jobs;
permissions are not silently changed.

The VMs fetch only each selected endpoint's full training state and small
provenance sidecars. They verify state SHA-256, schema, experiment/seed,
checkpoint and the selected critic configuration, then reconstruct only the
critic replay/networks plus frozen regret models. Other large state arrays
are released. There is no need to download these archives to the laptop.

Workers upload each completed timing repeat and small manifests/summaries.
No new policy or full-training-state checkpoints are produced; fetched source
states are inputs only and are never copied into the output prefix.
`resume` reuses complete repeats under the same source/code contract; an
interrupted repeat is rerun. It is not mid-optimizer-step recovery. Changing
source, code, Torch version or protocol requires a new run ID, preventing a
mixed-version benchmark. Keep RUN_ID/SOURCE_RUN_ID/REPO_REF unchanged for
recovery and do not launch two controllers against one output prefix.

There are no automatic worker retries. Hard task limits: real-source smoke
2 hours; each worker 12 hours; aggregation 1 hour; controller 18 hours.
These are safety timeouts, **not runtime estimates**. The three-worker cap
alone is 36 n2-standard-8 VM-hours; smoke, aggregation, controller, storage and
data transfer are additional. Queueing can extend elapsed time. The controller
failing does not automatically cancel a running child; check `status`.

Outputs in `analysis/`:

- `summary.json`: source-seed timings, speedup statistics and aggregate
  correctness gate.
- `per_seed_timing.csv`: timings, ratio, reduction, correctness and peak RSS.
- `critic_cache_efficiency.png`: component times and source-seed speedups.
- `detailed_results.json`: all repeat/fold timings and numerical comparisons.
- `experiment_metadata.json`, `source_manifests.json`: source identity,
  checksum, configuration, code hash, Git ref and runtime versions.
- `README.md`: interpretation limits.

Intentional output-convention deviation: this is not a new policy training
run. No exploitability, head-to-head, node/time convergence curves or playable
policies are generated. Source states and original experimental outputs are
never modified; this study establishes efficiency and numerical equivalence,
not stronger strategic play.
