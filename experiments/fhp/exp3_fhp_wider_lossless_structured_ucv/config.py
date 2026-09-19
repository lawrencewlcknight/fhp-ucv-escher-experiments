"""Frozen contract for FHP Experiment 3."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.fhp.exp2_fhp_lossless_structured_ucv import config as _base
from fhp_escher.features import ENCODER_ID, FULL_STATE_LAYOUT, POLICY_LAYOUT


EXPERIMENT_ID = 3
EXPERIMENT_NAME = "exp3_fhp_wider_lossless_structured_ucv"
ALGORITHM_ID = "wider_lossless_structured_ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER (wider lossless structured FHP network)"
BATCH_TIMEOUT_SECONDS = _base.BATCH_TIMEOUT_SECONDS
CHECKPOINT_HOURS = _base.CHECKPOINT_HOURS
PRODUCTION_SEEDS = _base.PRODUCTION_SEEDS
SMOKE_SEEDS = _base.SMOKE_SEEDS
REFERENCE_VM = deepcopy(_base.REFERENCE_VM)

BASELINE_ONLINE_PARAMETER_COUNT = 310_993
ONLINE_PARAMETER_COUNT = 611_729
PARAMETER_MULTIPLIER = ONLINE_PARAMETER_COUNT / BASELINE_ONLINE_PARAMETER_COUNT

EXPERIMENT_CONFIG = deepcopy(_base.EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "num_layers": 2,
        "num_hiddens": 192,
        "average_policy_network_layers": (256, 256),
        "structured_branch_width": 96,
    }
)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    return _base.checkpoint_schedule(smoke=smoke)


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


def smoke_config() -> dict:
    config = _base.smoke_config()
    config.update(
        {
            "num_layers": 2,
            "num_hiddens": 192,
            "average_policy_network_layers": (256, 256),
            "structured_branch_width": 96,
        }
    )
    return config


def validate_contract(
    *,
    seeds: Sequence[int],
    schedule: Sequence[Mapping],
    config: Mapping,
    smoke: bool,
) -> None:
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(int(seed) for seed in seeds) != expected_seeds:
        raise ValueError(f"Seeds must be {expected_seeds}")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen contract")
    expected_config = smoke_config() if smoke else EXPERIMENT_CONFIG
    if dict(config) != dict(expected_config):
        changed = sorted(
            key
            for key in set(config) | set(expected_config)
            if config.get(key) != expected_config.get(key)
        )
        raise ValueError(
            "Experiment 3 configuration differs from the frozen contract at "
            + ", ".join(changed)
        )


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "production_seeds": list(PRODUCTION_SEEDS),
        "training_hours_per_seed": 24,
        "checkpoint_hours": list(CHECKPOINT_HOURS),
        "seed_execution": "parallel_gcp_batch_task_array_one_vm_per_seed",
        "checkpoint_boundary": "first_completed_outer_iteration_after_threshold",
        "training_config": dict(EXPERIMENT_CONFIG),
        "representation": {
            "encoder_id": ENCODER_ID,
            "policy_feature_size": POLICY_LAYOUT.total_size,
            "critic_feature_size": FULL_STATE_LAYOUT.total_size,
            "lossless_modulo_suit_symmetry": True,
            "replay_dtype": "float32",
        },
        "network_capacity": {
            "structured_branch_width": 96,
            "regret_critic_calibration_layers": [192, 192],
            "average_policy_layers": [256, 256],
            "baseline_online_parameters": BASELINE_ONLINE_PARAMETER_COUNT,
            "online_parameters": ONLINE_PARAMETER_COUNT,
            "parameter_multiplier": PARAMETER_MULTIPLIER,
        },
        "baseline": "exp2_fhp_lossless_structured_ucv",
        "single_intended_change": "network_width",
        "exact_exploitability": False,
    }


__all__ = [
    "ALGORITHM_ID",
    "ALGORITHM_LABEL",
    "BASELINE_ONLINE_PARAMETER_COUNT",
    "BATCH_TIMEOUT_SECONDS",
    "CHECKPOINT_HOURS",
    "EXPERIMENT_CONFIG",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "ONLINE_PARAMETER_COUNT",
    "PARAMETER_MULTIPLIER",
    "PRODUCTION_SEEDS",
    "REFERENCE_VM",
    "SMOKE_SEEDS",
    "checkpoint_schedule",
    "contract_manifest",
    "smoke_config",
    "task_schedule",
    "validate_contract",
]
