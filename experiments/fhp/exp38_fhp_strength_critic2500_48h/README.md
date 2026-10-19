# Experiment 38 — continue Experiment 36 to 48 active hours

Continue, rather than restart, the three original Experiment 36 seeds from
`exp36-rich2500-20261018-205125`. This tests the effect of additional training
without changing the learning algorithm. No evaluation is scheduled by this
workflow; compare frozen policies separately after training completes.

## Frozen training contract

| Setting | Experiment 38 |
|---|---|
| Starting states | Experiment 36 complete 24h states, seeds 0, 1, 2 |
| Training target | 48 cumulative active hours per seed |
| Critic budget | Unchanged: 2,500 updates per critic member per fit |
| Representation | Unchanged richer AE strength/betting features |
| Parallelism | Eight Ray traversal workers and eight fitting threads per seed |
| Hardware | Standard CPU `n2-standard-16`, one VM per seed |
| Default seed scheduling | Three concurrent tasks, 48 N2 vCPUs total |
| Historical policy exports | Original 6/12/18/24h files copied unchanged |
| New policy exports | 30/36/42/48 cumulative active hours |
| Full training state | Final 48h state for every seed |
| Automatic evaluation / further continuation | Neither |

All other settings are inherited from Experiment 36. The cumulative clock stops
at the first completed iteration reaching 48 hours. The original states already
slightly exceed 24 hours, so the extra active time is approximately, rather than
exactly, 24 hours. Setup, downloads, policy extraction, checkpoint saving and
uploads add wall-clock time. The inherited 72h per-task Batch ceiling is generous
operational headroom, not a request for 72 hours of learning.

### Preserve the native learning state

The learner runs at the original Experiment 36 revision
`6ab85bbcd03627281e411243419d64ce9e9236d5`, with Python 3.11.16, Torch 2.7.0 CPU,
NumPy 1.26.4, Ray 2.51.2 and OpenSpiel 1.6.3. The learning configuration digest is
`c9dbd0be75986bda4e34c060da47b937f4feb657d9888ccdee6a506cfc1854ed`.
`REPO_REF` selects the new **workflow** revision, not a replacement learner.

Native worker manifests and checkpoint payloads deliberately retain Experiment
36 identity, just as earlier continuation workflows retained their source
identity. Experiment 38 owns the new run prefix, Batch jobs, source receipt and
analysis. This avoids renaming or partially converting saved optimisers, replay
buffers or actor states. The source run is read-only throughout.

The complete final checkpoint includes online and deployed networks, optimiser
states, replay/reservoir contents, target histories, learner and actor random
states, counters and cumulative training time. It is not merely a playable
policy export. Future continuation uses the native Experiment 36 `stage-source`
and `worker` commands at the original learner revision, a new output directory,
and a horizon beyond 48h. A new workflow can therefore extend these states to
72h without starting from scratch. Nothing starts that continuation automatically.

## Launch from this repository root

First commit and push the Experiment 38 files. A workflow revision without these
files cannot be used. The launcher extracts its builders, validation and embedded
reporting from the selected Git revision, so uncommitted edits cannot silently
change a submitted job.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp_ucv_escher/fhp-ucv-escher-experiments

export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp38-rich48-$(date -u +%Y%m%d-%H%M%S)"
export EXP38_PARALLELISM=3

# Prepare job specifications without accessing cloud services.
bash gcp/run_exp38_strength_critic2500_48h.sh dry-run

# Recommended first cloud check; wait for this job to succeed.
bash gcp/run_exp38_strength_critic2500_48h.sh smoke-cloud
bash gcp/run_exp38_strength_critic2500_48h.sh status
```

After smoke reports `SUCCEEDED`, retain the same environment and run ID:

```bash
bash gcp/run_exp38_strength_critic2500_48h.sh run
```

The controller reuses the successful smoke, launches three continuation tasks,
waits for completion, and aggregates their outputs. You may disconnect the
laptop. Calling `run` without a prior smoke is also supported: the controller
runs smoke before launching training. `EXP38_PARALLELISM=1` or `2` reduces
concurrent seed VMs but increases total elapsed time; it does not change the
eight Ray actors within each seed.

Preflight reads small source manifests and object metadata, confirming all three
states exist and match the pinned contract before paid training is submitted.
On each learner VM, native staging verifies full payload checksums and the
restore code verifies runtime/configuration compatibility before learning. Cloud
smoke checks capacity, cached-target equivalence, bitwise state restoration,
a tiny 24→48 continuation, and an additional tiny 48→72 continuation. These
short schedule analogues test execution, not policy quality or production runtime.

## Outputs and recovery

Results are stored under `$BUCKET/$RUN_ID/`, separately from Experiment 36:

- `workers/task_.../checkpoints/`: original and new playable policies.
- `workers/task_.../training_states/time_48h.pt`: complete final restart state.
- `experiment38_contract.json`: pinned source metadata, hashes and object generations.
- `analysis/restart_inventory.json`: each final state URI, checksum, size and learner revision.
- `analysis/continuation_progress.csv`: per-seed added time, nodes and iterations.
- `analysis/analysis_summary.md`: continuation summary; no policy-strength claims.
- `analysis/learner_*.json`: preserved native aggregation identity alongside the
  Experiment 38 workflow reports.

Aggregation verifies original policy hashes, continuation lineage and uploaded
48h state sizes before marking workflow analysis complete. Full payload hashes
are retained for validation on subsequent download/restore. Large training states
are not downloaded again just to produce analysis.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Keep the remote full states: downloading analysis alone does not download the
resumable models. Intermediate 30/36/42h policy files are **not** full training
states. Final-only retention matches Experiment 36 and avoids repeated large
serialisations; a failure before the 48h state is saved may require repeating
the continuation from the original 24h state.

The inherited training lifecycle retains resource diagnostics, worker logs and
failure uploads. No paid training stage retries automatically. If only reporting
fails after training succeeds, keep the same `RUN_ID` and `REPO_REF` and use:

```bash
bash gcp/run_exp38_strength_critic2500_48h.sh aggregate
```

If the controller stops but its child jobs have finished, `resume` starts a new
controller and reuses succeeded stages. It does not restart a failed training
job. Inspect failures before launching another training run; the original 24h
source remains available. An existing non-smoke output prefix cannot be reused
with `run`, and the launcher rejects simultaneous jobs under that prefix.

## Local validation

```bash
python -m pytest tests/test_exp38_strength_critic2500_48h.py -q
RUN_EXP38_RAY_TESTS=1 python -m pytest \
  tests/test_exp38_strength_critic2500_48h.py::test_real_native_24_to_48_to_72_restart -q
```

The opt-in integration test starts real local Ray actors and requires process
and socket permissions. The deployed cloud smoke uses the exact pinned learner
revision and production runtime. Neither local tests nor source preflight launch
production training.
