import hashlib
import json

from experiments.fhp.ucv_escher_baseline.config import (
    BEST_UCV_CONFIG,
    BEST_UCV_TRAINING_CONFIG_SHA256,
    BATCH_TIMEOUT_SECONDS,
    CHECKPOINT_TRAINING_SECONDS,
    ITERATION_SAFETY_CAP,
    REFERENCE_VM,
    TRAINING_DURATION_SECONDS,
    validate_config,
)


def test_experiment_1_budget_and_reference_vm():
    validate_config(BEST_UCV_CONFIG)
    assert CHECKPOINT_TRAINING_SECONDS == (21_600, 43_200)
    assert TRAINING_DURATION_SECONDS == 43_200
    assert BATCH_TIMEOUT_SECONDS == 50_400
    assert REFERENCE_VM == {
        "machine_type": "n2-standard-8",
        "cpu_milli": 8_000,
        "memory_mib": 32_000,
        "boot_disk_gib": 100,
    }


def test_selected_ucv_configuration_is_pinned():
    assert BEST_UCV_CONFIG["game_name"] == "FHP"
    assert BEST_UCV_CONFIG["num_traversals"] == 10_000
    assert BEST_UCV_CONFIG["max_num_iterations"] == ITERATION_SAFETY_CAP
    assert BEST_UCV_CONFIG["q_ensemble_size"] == 3
    assert BEST_UCV_CONFIG["calibration_train_steps"] == 2_000
    assert BEST_UCV_CONFIG["prediction_gate_ema_decay"] == 0.9
    assert BEST_UCV_CONFIG["evaluation_frequency"] == 0
    assert BEST_UCV_CONFIG["evaluate_initial_policy"] is False
    assert BEST_UCV_CONFIG["early_evaluation_node_thresholds"] == ()
    payload = {
        key: value for key, value in BEST_UCV_CONFIG.items() if key != "game_name"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == BEST_UCV_TRAINING_CONFIG_SHA256
