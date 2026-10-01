"""Same-hardware parallel-collection counterpart to Experiment 6."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.fhp.exp2_fhp_lossless_structured_ucv import config as _base

EXPERIMENT_ID = 7
EXPERIMENT_NAME = "exp7_fhp_parallel_structured_n2_standard16"
ALGORITHM_ID = "parallel_structured_ucv_escher_n2_16"
ALGORITHM_LABEL = "UCV-ESCHER (eight traversal workers, n2-standard-16)"
BATCH_TIMEOUT_SECONDS = _base.BATCH_TIMEOUT_SECONDS
CHECKPOINT_HOURS = _base.CHECKPOINT_HOURS
PRODUCTION_SEEDS = _base.PRODUCTION_SEEDS
SMOKE_SEEDS = _base.SMOKE_SEEDS
LEARNER_THREADS = 8
PARALLEL_WORKERS = 8
COLLECTION_CHUNK_SIZE = 1200


def parallel_settings(*, smoke: bool = False) -> dict:
    return {
        "parallel_num_workers": PARALLEL_WORKERS,
        "parallel_collection_chunk_size": 8 if smoke else COLLECTION_CHUNK_SIZE,
        "parallel_ray_object_store_memory": (128 * 1024**2 if smoke else 4 * 1024**3),
        # Isolate collection speedup; retain Experiment 6's learner ordering.
        "parallelize_independent_learners": False,
        "parallel_cache_actor_snapshots": True,
    }

# No learning hyperparameters, representation, replay sizes or target-cache
# flags change. Eight intra-op threads also match the old cloud bootstrap.
EXPERIMENT_CONFIG = deepcopy(_base.EXPERIMENT_CONFIG)
REFERENCE_VM = deepcopy(_base.REFERENCE_VM)
REFERENCE_VM.update(machine_type="n2-standard-16", cpu_milli=16_000, memory_mib=64_000)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    return _base.checkpoint_schedule(smoke=smoke)


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[tuple[str, int], ...]:
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


def smoke_config() -> dict:
    config = _base.smoke_config()
    # Exercise all eight actors and snapshot reuse over two chunks per player.
    config["num_traversals"] = 16
    return config


def validate_contract(*, seeds: Sequence[int], schedule: Sequence[Mapping],
                      config: Mapping, smoke: bool) -> None:
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(int(seed) for seed in seeds) != expected_seeds:
        raise ValueError(f"Seeds must be {expected_seeds}")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen contract")
    expected = smoke_config() if smoke else EXPERIMENT_CONFIG
    if dict(config) != expected:
        changed = sorted(key for key in set(config) | set(expected)
                         if config.get(key) != expected.get(key))
        raise ValueError("Experiment 7 configuration differs at " + ", ".join(changed))


def contract_manifest() -> dict:
    manifest = deepcopy(_base.contract_manifest())
    manifest.update(
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID,
        algorithm_label=ALGORITHM_LABEL,
        training_config=deepcopy(EXPERIMENT_CONFIG),
        reference_vm=deepcopy(REFERENCE_VM),
        baseline="exp6_fhp_structured_n2_standard16",
        baseline_reference_vm=deepcopy(REFERENCE_VM),
        single_intended_change="synchronous_parallel_traversal_collection",
        traversal_execution="ray_parallel",
        parallel_traversal_workers=PARALLEL_WORKERS,
        parallel_settings=parallel_settings(),
        traversal_budget_scope="10000 total per player per iteration, not per worker",
        learner_intraop_threads=LEARNER_THREADS,
        frozen_critic_target_cache=False,
        training_state_retention="none",
        expected_speedup="not_assumed; measure completed iterations and training nodes",
        historical_comparison_caveat=(
            "Compare with Experiment 6 on the same commit, VM and thread count. "
            "Actor RNG streams change realised samples, not the learning rule; "
            "serial/parallel trajectories are not expected to be bitwise equal."
        ),
    )
    return manifest
