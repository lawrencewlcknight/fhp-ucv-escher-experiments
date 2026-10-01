"""FHP Experiment 5: correctness-gated frozen target caching."""
from experiments.critic_target_cache_cli import main
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _make_solver
from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import smoke_config

SPEC = {
    "experiment_id": 5,
    "training_state_retention": "none",
    "experiment_name": "exp5_fhp_critic_target_cache",
    "game": "fhp",
    "seeds": [0, 1, 2],
    "source_run_default": "exp2-fhp-20260921-093839",
    "source_experiment": "exp2_fhp_lossless_structured_ucv",
    "source_algorithm": "lossless_structured_ucv_escher",
    "checkpoint": "time_24h",
    "state_type": "exp2_fhp_lossless_structured_full_training_state",
    "manifest": "checkpoint_manifest.json",
    "arms": ["recompute", "cache"],
    "timing_repeats": 3,
    "production_critic_updates": 10000,
    "production_batch_size": 2048,
    "production_threads": 8,
}


def make_smoke(seed):
    config = smoke_config()
    config.update(evaluation_frequency=0, evaluate_initial_policy=False, early_evaluation_node_thresholds=())
    solver = _make_solver(seed, config)
    for _ in range(2):
        solver.iteration()
    return solver


if __name__ == "__main__":
    main(SPEC, _make_solver, make_smoke)
