"""Frozen contract for the new FHP Experiment 1 baseline."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from unbiased_escher.grouped_wide_solver import (
    GROUPED_SOFT_TARGET_CROSS_ENTROPY,
)


EXPERIMENT_ID = 1
EXPERIMENT_NAME = "exp1_fhp_grouped_wide_ucv_baseline"
ALGORITHM_ID = "grouped_wide_ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER (Experiment 35 grouped-wide policy)"
PRODUCTION_SEEDS = (0, 1, 2)
SMOKE_SEEDS = (0,)
CHECKPOINT_HOURS = (6, 12, 18, 24)
CHECKPOINT_TRAINING_SECONDS = tuple(hours * 60 * 60 for hours in CHECKPOINT_HOURS)
SMOKE_CHECKPOINT_SECONDS = (0.001, 0.002, 0.003, 0.004)
ITERATION_SAFETY_CAP = 1_000_000
BATCH_TIMEOUT_SECONDS = 36 * 60 * 60

# Exact training-algorithm transfer from Leduc Experiment 35.  The FHP game
# identifier and impractical whole-tree evaluation controls are the only
# game-scale adaptations.  Leduc-only paired/exact diagnostics are not part of
# the training algorithm and are deliberately excluded.
EXPERIMENT_35_CONFIG = {
    "game_name": "FHP",
    "advantage_buffer_size": 1_000_000,
    "ave_policy_buffer_size": 1_000_000,
    "baseline_buffer_size": 1_000_000,
    "learning_rate": 1e-3,
    "num_traversals": 10_000,
    "advantage_network_train_steps": 750,
    "ave_policy_network_train_steps": 5_000,
    "baseline_network_train_steps": 10_000,
    "advantage_batch_size": 2_048,
    "ave_policy_batch_size": 2_048,
    "baseline_batch_size": 2_048,
    "num_layers": 3,
    "num_hiddens": 64,
    "reinitialize_advantage_networks": False,
    "reinitialize_imm_regret_networks": True,
    "use_regret_matching_argmax": True,
    "epsilon": 0.6,
    "fit_advantage": True,
    "use_baseline": True,
    "alpha": 2.3,
    "gamma": 2.0,
    "device": "cpu",
    "evaluation_frequency": 0,
    "max_num_iterations": ITERATION_SAFETY_CAP,
    "preserve_evaluation_rng": True,
    "evaluate_initial_policy": False,
    "early_evaluation_node_thresholds": (),
    "q_gradient_clip_norm": 10.0,
    "q_ensemble_size": 2,
    "beta_min": 0.0,
    "beta_max": 2.0,
    "beta_ridge": 1e-4,
    "fixed_control_variate_beta": 1.0,
    "force_prediction_gate_zero": True,
    "use_instantaneous_predictor": False,
    "sampling_uniform_floor_mass": 0.2,
    "calibration_buffer_size": 1_000_000,
    "calibration_batch_size": 2_048,
    "calibration_train_steps": 2_000,
    "calibration_learning_rate": 1e-3,
    "calibration_minimum_variance": 1e-5,
    "prediction_gate_ema_decay": 0.9,
    "prediction_gate_initial": 0.0,
    "use_residual_calibration": True,
    "regret_network_type": "mlp",
    "regret_residual_width": 64,
    "regret_residual_blocks": 4,
    "regret_policy_gradient_clip_norm": None,
    "anneal_start_nodes": None,
    "anneal_end_nodes": None,
    "anneal_final_learning_rate": None,
    "critic_target_average_window": 4,
    "average_policy_loss": GROUPED_SOFT_TARGET_CROSS_ENTROPY,
    "average_policy_reset_each_fit": True,
    "average_policy_network_layers": (136, 136, 136),
    "average_policy_learning_rate": 3e-3,
    "average_policy_train_steps": 20_000,
    "paired_legacy_network_layers": (64, 64, 64),
    "paired_legacy_learning_rate": 1e-3,
    "paired_legacy_train_steps": 5_000,
}

REFERENCE_VM = {
    "machine_type": "n2-standard-8",
    "cpu_milli": 8_000,
    "memory_mib": 32_000,
    "boot_disk_gib": 200,
    "boot_disk_type": "pd-balanced",
}


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    seconds = SMOKE_CHECKPOINT_SECONDS if smoke else CHECKPOINT_TRAINING_SECONDS
    return tuple(
        {
            "checkpoint_id": (
                f"smoke_time_{index:02d}"
                if smoke
                else f"time_{CHECKPOINT_HOURS[index - 1]:02d}h"
            ),
            "target_training_seconds": float(target),
            "target_training_hours": float(target) / 3600.0,
        }
        for index, target in enumerate(seconds, start=1)
    )


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


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
    required = {
        "game_name": "FHP",
        "q_ensemble_size": 2,
        "critic_target_average_window": 4,
        "fixed_control_variate_beta": 1.0,
        "use_instantaneous_predictor": False,
        "use_residual_calibration": True,
        "average_policy_loss": GROUPED_SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
        "average_policy_network_layers": (136, 136, 136),
        "average_policy_learning_rate": 3e-3,
        "average_policy_train_steps": 1 if smoke else 20_000,
        "paired_legacy_network_layers": (64, 64, 64),
        "evaluation_frequency": 0,
        "evaluate_initial_policy": False,
        "early_evaluation_node_thresholds": (),
    }
    for key, value in required.items():
        observed = config.get(key)
        if key == "average_policy_network_layers":
            observed = tuple(observed)
        if observed != value:
            raise ValueError(f"Frozen configuration requires {key}={value!r}")


def smoke_config() -> dict:
    config = deepcopy(EXPERIMENT_35_CONFIG)
    for key in (
        "advantage_buffer_size",
        "ave_policy_buffer_size",
        "baseline_buffer_size",
        "calibration_buffer_size",
    ):
        config[key] = 256
    for key in (
        "advantage_batch_size",
        "ave_policy_batch_size",
        "baseline_batch_size",
        "calibration_batch_size",
    ):
        config[key] = 2
    for key in (
        "advantage_network_train_steps",
        "ave_policy_network_train_steps",
        "baseline_network_train_steps",
        "calibration_train_steps",
        "average_policy_train_steps",
    ):
        config[key] = 1
    config["num_traversals"] = 4
    return config


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
        "training_config": dict(EXPERIMENT_35_CONFIG),
        "source": "Leduc ESCHER architecture Experiment 35",
        "excluded_leduc_diagnostics": [
            "exact_tabular_exploitability",
            "exact_tabular_average_policy",
            "paired_legacy_row_policy_fit",
        ],
    }


__all__ = [
    "ALGORITHM_ID",
    "ALGORITHM_LABEL",
    "BATCH_TIMEOUT_SECONDS",
    "CHECKPOINT_HOURS",
    "CHECKPOINT_TRAINING_SECONDS",
    "EXPERIMENT_35_CONFIG",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "ITERATION_SAFETY_CAP",
    "PRODUCTION_SEEDS",
    "REFERENCE_VM",
    "SMOKE_SEEDS",
    "checkpoint_schedule",
    "contract_manifest",
    "smoke_config",
    "task_schedule",
    "validate_contract",
]
