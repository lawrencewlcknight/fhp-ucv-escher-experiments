"""Experiment 7's learner with a 48-hour horizon and explicit continuation."""
from copy import deepcopy
from typing import Mapping, Sequence

from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16 import config as _base

EXPERIMENT_ID = 8
EXPERIMENT_NAME = "exp8_fhp_parallel_48h"
ALGORITHM_ID = "parallel_structured_ucv_escher_48h"
ALGORITHM_LABEL = "UCV-ESCHER (eight traversal workers, 48-hour horizon)"
DEFAULT_TOTAL_HOURS = 48
CHECKPOINT_HOURS = tuple(range(6, 49, 6))
BATCH_TIMEOUT_SECONDS = 72 * 3600
PRODUCTION_SEEDS = _base.PRODUCTION_SEEDS
SMOKE_SEEDS = _base.SMOKE_SEEDS
LEARNER_THREADS = _base.LEARNER_THREADS
PARALLEL_WORKERS = _base.PARALLEL_WORKERS
COLLECTION_CHUNK_SIZE = _base.COLLECTION_CHUNK_SIZE
EXPERIMENT_CONFIG = deepcopy(_base.EXPERIMENT_CONFIG)
REFERENCE_VM = deepcopy(_base.REFERENCE_VM)
parallel_settings = _base.parallel_settings
smoke_config = _base.smoke_config


def validate_total_hours(total_hours):
    if isinstance(total_hours, bool) or int(total_hours) != total_hours or total_hours < 48 or total_hours % 6:
        raise ValueError("Total active hours must be an integer multiple of six, at least 48")
    return int(total_hours)


def checkpoint_schedule(*, smoke=False, total_hours=DEFAULT_TOTAL_HOURS):
    total_hours = validate_total_hours(total_hours)
    hours = tuple(range(6, total_hours + 1, 6))
    return tuple({
        "checkpoint_id": f"smoke_time_{i:02d}" if smoke else f"time_{hour:02d}h",
        "target_training_seconds": float(i * .001 if smoke else hour * 3600),
        "target_training_hours": float(i * .001 / 3600 if smoke else hour),
    } for i, hour in enumerate(hours, 1))


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS):
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


def validate_contract(*, seeds: Sequence[int], schedule: Sequence[Mapping],
                      config: Mapping, smoke: bool, total_hours=DEFAULT_TOTAL_HOURS):
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(seeds) != expected_seeds:
        raise ValueError(f"Seeds must be {expected_seeds}")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke, total_hours=total_hours):
        raise ValueError("Checkpoint schedule differs from the frozen contract")
    expected = smoke_config() if smoke else EXPERIMENT_CONFIG
    if dict(config) != expected:
        raise ValueError("Experiment 8 learning configuration must equal Experiment 7")


def contract_manifest(*, total_hours=DEFAULT_TOTAL_HOURS):
    manifest = deepcopy(_base.contract_manifest())
    schedule = checkpoint_schedule(total_hours=total_hours)
    manifest.update(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        baseline=_base.EXPERIMENT_NAME,
        single_intended_change="active_training_horizon",
        checkpoint_schedule=list(schedule),
        checkpoint_training_seconds=[r["target_training_seconds"] for r in schedule],
        checkpoint_hours=[r["target_training_hours"] for r in schedule],
        training_duration_seconds=total_hours * 3600,
        total_active_hours=total_hours,
        training_state_retention="final",
        training_hours_per_seed=total_hours,
        batch_timeout_seconds=BATCH_TIMEOUT_SECONDS,
        continuation="Final endpoint only: full driver and actor state; extend in a new run prefix with the same commit/runtime",
        historical_comparison_caveat="Long-horizon follow-up, not a new architecture or a convergence guarantee.",
    )
    return manifest
