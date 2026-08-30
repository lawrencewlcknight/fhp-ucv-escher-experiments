"""Configuration for FHP Experiment 1."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


EXPERIMENT_ID = 1
EXPERIMENT_NAME = "exp1_fhp_ucv_escher_baseline"
ALGORITHM_ID = "ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER"
DEFAULT_SEED = 0
CHECKPOINT_TRAINING_SECONDS = (6 * 60 * 60, 12 * 60 * 60)
TRAINING_DURATION_SECONDS = CHECKPOINT_TRAINING_SECONDS[-1]
ITERATION_SAFETY_CAP = 1_000_000
BATCH_TIMEOUT_SECONDS = 14 * 60 * 60

# Experiment 1 preserves every algorithm and optimiser setting from the final
# predecessor UCV-ESCHER configuration. The required game identifier is FHP,
# and the two whole-game-era evaluation triggers are disabled. Runtime, node,
# and checkpoint controls are orchestration safeguards, not update-rule changes.
BEST_UCV_CONFIG = {
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
    "q_ensemble_size": 3,
    "beta_min": 0.0,
    "beta_max": 2.0,
    "beta_ridge": 1e-4,
    "sampling_uniform_floor_mass": 0.2,
    "calibration_buffer_size": 1_000_000,
    "calibration_batch_size": 2_048,
    "calibration_train_steps": 2_000,
    "calibration_learning_rate": 1e-3,
    "calibration_minimum_variance": 1e-5,
    "prediction_gate_ema_decay": 0.9,
    "prediction_gate_initial": 0.0,
}

# SHA-256 of the canonical JSON for every setting above except ``game_name``.
# This makes an accidental optimisation or architecture change visible.
BEST_UCV_TRAINING_CONFIG_SHA256 = (
    "42a1c60051502d7cf44d6a368d144588b7b470910d4a20c602b2667118431846"
)

REFERENCE_VM = {
    "machine_type": "n2-standard-8",
    "cpu_milli": 8_000,
    "memory_mib": 32_000,
    "boot_disk_gib": 100,
    "boot_disk_type": "pd-balanced",
}


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("game_name") != "FHP":
        raise ValueError("Experiment 1 must use the canonical FHP loader")
    if int(config["q_ensemble_size"]) != 3:
        raise ValueError("The transferred UCV configuration uses three Q folds")
    if int(config["evaluation_frequency"]) != 0:
        raise ValueError("Periodic outer-iteration checkpoints are disabled for FHP")
    if int(config["max_num_iterations"]) != ITERATION_SAFETY_CAP:
        raise ValueError("The time-based experiment requires a nonbinding safety cap")
    if bool(config["evaluate_initial_policy"]):
        raise ValueError("Initial-policy evaluation is disabled for FHP")
    if tuple(config["early_evaluation_node_thresholds"]):
        raise ValueError("Early node-threshold evaluation is disabled for FHP")
    if not bool(config["preserve_evaluation_rng"]):
        raise ValueError("Checkpoint fitting must not perturb training RNG state")


def smoke_config() -> dict:
    """Return a tiny orchestration test while retaining the same mechanisms."""
    config = deepcopy(BEST_UCV_CONFIG)
    config.update(
        {
            "advantage_buffer_size": 128,
            "ave_policy_buffer_size": 128,
            "baseline_buffer_size": 192,
            "num_traversals": 2,
            "advantage_network_train_steps": 1,
            "ave_policy_network_train_steps": 1,
            "baseline_network_train_steps": 1,
            "advantage_batch_size": 2,
            "ave_policy_batch_size": 2,
            "baseline_batch_size": 2,
            "max_num_iterations": ITERATION_SAFETY_CAP,
            "calibration_buffer_size": 128,
            "calibration_batch_size": 2,
            "calibration_train_steps": 1,
        }
    )
    return config
