"""Experiment 9 plus additive, information-preserving hand/board features."""
from copy import deepcopy

from experiments.fhp.exp9_fhp_cached_parallel_24h import config as base
from fhp_escher.hand_board_features import ENCODER_ID, FHPHandBoardFeatureEncoder

EXPERIMENT_ID = 10
EXPERIMENT_NAME = "exp10_fhp_hand_board_features"
ALGORITHM_ID = "hand_board_cached_parallel_ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER (cached parallel, explicit hand/board features)"
DEFAULT_TOTAL_HOURS = base.DEFAULT_TOTAL_HOURS
CHECKPOINT_HOURS = base.CHECKPOINT_HOURS
BATCH_TIMEOUT_SECONDS = base.BATCH_TIMEOUT_SECONDS
PRODUCTION_SEEDS = base.PRODUCTION_SEEDS
SMOKE_SEEDS = base.SMOKE_SEEDS
LEARNER_THREADS = base.LEARNER_THREADS
PARALLEL_WORKERS = base.PARALLEL_WORKERS
COLLECTION_CHUNK_SIZE = base.COLLECTION_CHUNK_SIZE
REFERENCE_VM = deepcopy(base.REFERENCE_VM)
EXPERIMENT_CONFIG = deepcopy(base.EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG["feature_encoder_id"] = ENCODER_ID
parallel_settings = base.parallel_settings
validate_total_hours = base.validate_total_hours
checkpoint_schedule = base.checkpoint_schedule


def smoke_config():
    return {**base.smoke_config(), "feature_encoder_id": ENCODER_ID}


def task_schedule(seeds=PRODUCTION_SEEDS):
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


def validate_contract(*, seeds, schedule, config, smoke,
                      total_hours=DEFAULT_TOTAL_HOURS):
    expected = smoke_config() if smoke else EXPERIMENT_CONFIG
    if dict(config) != expected:
        raise ValueError("Experiment 10 must equal Experiment 9 plus the hand/board encoder")
    inherited = dict(config)
    inherited.pop("feature_encoder_id")
    base.validate_contract(seeds=seeds, schedule=schedule, config=inherited,
                           smoke=smoke, total_hours=total_hours)


def contract_manifest(*, total_hours=DEFAULT_TOTAL_HOURS):
    manifest = deepcopy(base.contract_manifest(total_hours=total_hours))
    encoder = FHPHandBoardFeatureEncoder()
    manifest["representation"].update(
        encoder_id=encoder.encoder_id,
        policy_feature_size=encoder.policy_layout.total_size,
        critic_feature_size=encoder.full_state_layout.total_size,
        additive_features_only=True,
    )
    manifest.update(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        baseline=base.EXPERIMENT_NAME, training_config=deepcopy(EXPERIMENT_CONFIG),
        single_intended_change="additive_hand_strength_and_board_interaction_features",
        feature_encoder=encoder.metadata(),
        affected_networks=["average_policy", "regret", "critic", "calibration"],
        historical_comparison_caveat=(
            "Compare to Experiment 9 using the same seed labels, 24-active-hour budget, "
            "eight actors and VM class. Historical hardware/runtime differences remain. "
            "Hidden widths are unchanged; extra inputs increase first-layer parameters "
            "and replay bytes. Different input dimensions change initial weights/RNG "
            "consumption, so this is a representation-package test, not a bitwise paired trajectory. "
            "Throughput alone does not establish stronger play; evaluate saved policies "
            "with the same FHP rule/LBR/cross-play protocol."
        ),
    )
    return manifest
