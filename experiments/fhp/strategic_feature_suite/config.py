"""Frozen contracts: no baseline retraining or cross-candidate evaluation."""
from copy import deepcopy
from types import SimpleNamespace

from experiments.fhp.exp10_fhp_hand_board_features import config as base

from .registry import SPECS, BASELINE_RUN_ID, BASELINE_REF


def get_experiment(number):
    number = int(number)
    slug, kind, label = SPECS[number]
    return make_experiment(number, slug, kind, label)


def make_experiment(number, slug, kind, label, *, critic_updates=None):
    """Reuse the feature/checkpoint contract without changing archived experiments."""
    from fhp_escher.strategic_features import ENCODER_IDS, StrategicFeatureEncoder
    config = deepcopy(base.EXPERIMENT_CONFIG)
    config["feature_encoder_id"] = ENCODER_IDS[kind]
    if critic_updates is not None:
        if isinstance(critic_updates, bool) or not isinstance(critic_updates, int) or critic_updates <= 0:
            raise ValueError("critic_updates must be a positive integer")
        config["baseline_network_train_steps"] = critic_updates
    exp = SimpleNamespace(
        EXPERIMENT_ID=number, EXPERIMENT_NAME=f"exp{number}_fhp_{slug}",
        ALGORITHM_ID=(f"strategic_{kind.lower()}_ucv_escher" if critic_updates is None
                      else f"strategic_{kind.lower()}_critic{critic_updates}_ucv_escher"), ALGORITHM_LABEL=label,
        FEATURE_PACKAGE=kind, SLUG=slug, BASELINE_RUN_ID=BASELINE_RUN_ID, BASELINE_REF=BASELINE_REF,
        DEFAULT_TOTAL_HOURS=24, CHECKPOINT_HOURS=base.CHECKPOINT_HOURS,
        BATCH_TIMEOUT_SECONDS=base.BATCH_TIMEOUT_SECONDS, PRODUCTION_SEEDS=base.PRODUCTION_SEEDS,
        SMOKE_SEEDS=base.SMOKE_SEEDS, LEARNER_THREADS=base.LEARNER_THREADS,
        PARALLEL_WORKERS=base.PARALLEL_WORKERS, COLLECTION_CHUNK_SIZE=base.COLLECTION_CHUNK_SIZE,
        REFERENCE_VM=deepcopy(base.REFERENCE_VM), EXPERIMENT_CONFIG=config,
        parallel_settings=base.parallel_settings, validate_total_hours=base.validate_total_hours,
        checkpoint_schedule=base.checkpoint_schedule)
    def smoke_config():
        result = dict(base.smoke_config(), feature_encoder_id=ENCODER_IDS[kind])
        if critic_updates is not None:
            result["baseline_network_train_steps"] = 2
        return result
    exp.smoke_config = smoke_config
    exp.task_schedule = lambda seeds=exp.PRODUCTION_SEEDS: tuple((exp.ALGORITHM_ID, int(s)) for s in seeds)
    exp.encoder = lambda: StrategicFeatureEncoder(kind)

    def validate_contract(*, seeds, schedule, config, smoke, total_hours=24):
        expected = exp.smoke_config() if smoke else exp.EXPERIMENT_CONFIG
        if dict(config) != expected:
            raise ValueError("Only the prespecified feature encoder and critic budget may differ from Experiment 10")
        inherited = dict(config, feature_encoder_id=base.EXPERIMENT_CONFIG["feature_encoder_id"])
        inherited["baseline_network_train_steps"] = (
            base.smoke_config() if smoke else base.EXPERIMENT_CONFIG)["baseline_network_train_steps"]
        base.validate_contract(seeds=seeds, schedule=schedule, config=inherited,
                               smoke=smoke, total_hours=total_hours)

    def contract_manifest(*, total_hours=24):
        result = deepcopy(base.contract_manifest(total_hours=total_hours))
        encoder = exp.encoder()
        result.update(experiment_id=number, experiment_name=exp.EXPERIMENT_NAME,
            algorithm_id=exp.ALGORITHM_ID, algorithm_label=label, baseline=base.EXPERIMENT_NAME,
            baseline_run_id=BASELINE_RUN_ID, baseline_commit=BASELINE_REF,
            baseline_retrained=False, feature_package=kind, training_config=deepcopy(config),
            feature_encoder=encoder.metadata(), single_intended_change=f"additive_feature_package_{kind}",
            initialisation="fresh training, never warm-start from Experiment 10",
            automatic_evaluation="saved 24h policy versus historical Experiment 10 24h; no cross-candidate comparisons",
            evaluation_output="evaluation/analysis", training_state_retention="final",
            historical_comparison_caveat="Historical source commits and matched seed labels do not isolate runtime or RNG differences. More input parameters and feature cost are part of the package. No causal attribution to one descriptor or automatic promotion.")
        result["representation"].update(encoder_id=encoder.encoder_id,
            policy_feature_size=encoder.policy_size, critic_feature_size=encoder.full_state_size)
        if critic_updates is not None:
            result.pop("single_intended_change")
            result.update(
                intended_changes=[f"additive_feature_package_{kind}", f"critic_updates_{critic_updates}"],
                critic_updates_per_member=critic_updates,
                automatic_evaluation=False,
                evaluation_comparators=[20, 33, 10], primary_comparator=20,
                historical_comparison_caveat="Two changes relative to Experiment 20: richer inputs and fewer critic updates. "
                    "Experiment 33 holds features fixed but uses 10,000 updates. Historical comparisons are not a factorial interaction test.")
        return result
    exp.validate_contract, exp.contract_manifest = validate_contract, contract_manifest
    return exp
