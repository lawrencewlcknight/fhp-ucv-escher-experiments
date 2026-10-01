"""Experiment 7 with per-fit target caching and a final continuation state."""
from copy import deepcopy
from typing import Mapping, Sequence

from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16 import config as base

EXPERIMENT_ID = 9
EXPERIMENT_NAME = "exp9_fhp_cached_parallel_24h"
ALGORITHM_ID = "cached_parallel_structured_ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER (eight traversal workers, cached critic targets)"
DEFAULT_TOTAL_HOURS = 24
CHECKPOINT_HOURS = base.CHECKPOINT_HOURS
BATCH_TIMEOUT_SECONDS = base.BATCH_TIMEOUT_SECONDS
PRODUCTION_SEEDS = base.PRODUCTION_SEEDS
SMOKE_SEEDS = base.SMOKE_SEEDS
LEARNER_THREADS = base.LEARNER_THREADS
PARALLEL_WORKERS = base.PARALLEL_WORKERS
COLLECTION_CHUNK_SIZE = base.COLLECTION_CHUNK_SIZE
REFERENCE_VM = deepcopy(base.REFERENCE_VM)
EXPERIMENT_CONFIG = deepcopy(base.EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG["cache_frozen_critic_targets"] = True
parallel_settings = base.parallel_settings


def smoke_config():
    config = base.smoke_config()
    config["cache_frozen_critic_targets"] = True
    return config


def validate_total_hours(total_hours):
    if (isinstance(total_hours, bool) or int(total_hours) != total_hours
            or total_hours < 24 or total_hours % 6):
        raise ValueError("Total active hours must be an integer multiple of six, at least 24")
    return int(total_hours)


def checkpoint_schedule(*, smoke=False, total_hours=DEFAULT_TOTAL_HOURS):
    total_hours = validate_total_hours(total_hours)
    return tuple({
        "checkpoint_id": f"smoke_time_{i:02d}" if smoke else f"time_{hour:02d}h",
        "target_training_seconds": float(i * .001 if smoke else hour * 3600),
        "target_training_hours": float(i * .001 / 3600 if smoke else hour),
    } for i, hour in enumerate(range(6, total_hours + 1, 6), 1))


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS):
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


def validate_contract(*, seeds: Sequence[int], schedule: Sequence[Mapping],
                      config: Mapping, smoke: bool, total_hours=DEFAULT_TOTAL_HOURS):
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(seeds) != expected_seeds:
        raise ValueError(f"Seeds must be {expected_seeds}")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(
            smoke=smoke, total_hours=total_hours):
        raise ValueError("Checkpoint schedule differs from the frozen contract")
    expected = smoke_config() if smoke else EXPERIMENT_CONFIG
    if dict(config) != expected:
        raise ValueError("Experiment 9 learning configuration must equal Experiment 7 plus caching")


def contract_manifest(*, total_hours=DEFAULT_TOTAL_HOURS):
    schedule = checkpoint_schedule(total_hours=total_hours)
    manifest = deepcopy(base.contract_manifest())
    manifest.update(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        baseline=base.EXPERIMENT_NAME, training_config=deepcopy(EXPERIMENT_CONFIG),
        single_intended_change="cache_frozen_critic_targets_once_per_fit",
        frozen_critic_target_cache=True,
        checkpoint_schedule=list(schedule),
        checkpoint_training_seconds=[r["target_training_seconds"] for r in schedule],
        checkpoint_hours=[r["target_training_hours"] for r in schedule],
        training_duration_seconds=total_hours * 3600, total_active_hours=total_hours,
        training_hours_per_seed=total_hours, training_state_retention="final",
        batch_timeout_seconds=(36 if total_hours == 24 else 72) * 3600,
        continuation="Final endpoint only: full driver and actor state, same commit/runtime, new output prefix",
        cache_lifetime="one critic fit; rebuild after replay, targets or iteration change",
        historical_comparison_caveat=(
            "Same-seed historical efficiency follow-up to Experiment 7, not a new architecture. "
            "Identical VM class does not eliminate hardware/runtime variability. "
            "Caching equivalence is tested at fixed iterations, not equal wall time. "
            "Throughput does not establish lower exploitability or convergence."
        ),
    )
    return manifest
