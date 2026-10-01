"""Frozen VM-only replication of the Experiment 2 learning configuration."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.fhp.exp2_fhp_lossless_structured_ucv import config as _base

EXPERIMENT_ID = 6
EXPERIMENT_NAME = "exp6_fhp_structured_n2_standard16"
ALGORITHM_ID = "lossless_structured_ucv_escher_n2_16"
ALGORITHM_LABEL = "UCV-ESCHER (Experiment 2 configuration, n2-standard-16)"
BATCH_TIMEOUT_SECONDS = _base.BATCH_TIMEOUT_SECONDS
CHECKPOINT_HOURS = _base.CHECKPOINT_HOURS
PRODUCTION_SEEDS = _base.PRODUCTION_SEEDS
SMOKE_SEEDS = _base.SMOKE_SEEDS
LEARNER_THREADS = 8

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
    return _base.smoke_config()


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
        raise ValueError("Experiment 6 configuration differs at " + ", ".join(changed))


def contract_manifest() -> dict:
    manifest = deepcopy(_base.contract_manifest())
    manifest.update(
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID,
        algorithm_label=ALGORITHM_LABEL,
        training_config=deepcopy(EXPERIMENT_CONFIG),
        reference_vm=deepcopy(REFERENCE_VM),
        baseline="exp2_fhp_lossless_structured_ucv",
        baseline_reference_vm=deepcopy(_base.REFERENCE_VM),
        single_intended_change="vm_size",
        traversal_execution="sequential_single_collector",
        parallel_traversal_workers=0,
        learner_intraop_threads=LEARNER_THREADS,
        frozen_critic_target_cache=False,
        training_state_retention="none",
        expected_speedup="not_assumed; measure completed iterations and training nodes",
        historical_comparison_caveat=(
            "Learning configuration matches Experiment 2. Historical results may "
            "also reflect intervening semantics-preserving runtime optimisations; "
            "a same-commit n2-standard-8 rerun is needed to isolate hardware alone."
        ),
    )
    return manifest
