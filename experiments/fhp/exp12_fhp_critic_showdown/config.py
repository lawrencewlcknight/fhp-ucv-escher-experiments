"""Experiment 9 plus a critic-only comparison of the completed showdown hands."""
from copy import deepcopy

from experiments.fhp.exp9_fhp_cached_parallel_24h import config as base
from fhp_escher.critic_showdown_features import ENCODER_ID, FHPCriticShowdownFeatureEncoder

EXPERIMENT_ID = 12
EXPERIMENT_NAME = "exp12_fhp_critic_showdown"
ALGORITHM_ID = "critic_showdown_cached_parallel_ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER (cached parallel, explicit critic-only showdown)"
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
        raise ValueError("Experiment 12 must equal Experiment 9 plus the critic-showdown encoder")
    inherited = dict(config)
    inherited.pop("feature_encoder_id")
    base.validate_contract(seeds=seeds, schedule=schedule, config=inherited,
                           smoke=smoke, total_hours=total_hours)


def contract_manifest(*, total_hours=DEFAULT_TOTAL_HOURS):
    manifest = deepcopy(base.contract_manifest(total_hours=total_hours))
    encoder = FHPCriticShowdownFeatureEncoder()
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
        single_intended_change="critic_only_exact_showdown_comparison",
        feature_encoder=encoder.metadata(),
        affected_networks=["critic"],
        information_boundary={
            "unchanged_inputs": ["average_policy", "regret", "calibration"],
            "privileged_comparison": "full_state_critic_only",
            "availability": "completed_three_card_flop_only",
            "calibration_disagreement": "zero_with_one_heldout_critic_of_two_folds",
            "no_best_response_or_future_card_sampling": True,
        },
        historical_comparison_caveat=(
            "Compare to Experiment 9 using the same seed labels, 24-active-hour budget, "
            "eight actors and VM class. Historical hardware/runtime differences remain. "
            "Policy/regret/calibration inputs and all hidden widths are unchanged; "
            "three critic-only inputs increase critic parameters and replay bytes. "
            "Different critic input dimensions change initial weights/RNG "
            "consumption, so this is a representation-package test, not a bitwise paired trajectory. "
            "Throughput alone does not establish stronger play; evaluate saved policies "
            "with the same FHP rule/LBR/cross-play protocol."
        ),
    )
    return manifest
