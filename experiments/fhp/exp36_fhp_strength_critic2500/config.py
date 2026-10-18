"""Two prespecified changes to Experiment 20; all other learning settings fixed."""
from experiments.fhp.strategic_feature_suite.config import make_experiment

EXPERIMENT = make_experiment(36, "strength_critic2500", "AE",
    "Relative strength and betting interactions; 2,500 critic updates", critic_updates=2500)
EXPERIMENT.REMOTE_TASK_ENV = "EXP36_REMOTE_TASK_URI"
EXPERIMENT.REQUIRE_CRITIC_BUDGET_DIAGNOSTICS = True
_feature_manifest = EXPERIMENT.contract_manifest


def _manifest(*, total_hours=24):
    result = _feature_manifest(total_hours=total_hours)
    result.update(
        baseline="exp20_fhp_half_critic_updates",
        baseline_run_id="exp20-critic24-20261006-164357",
        baseline_commit="7020ef71094e65b04eceec810b86441cf79c1ba2",
        feature_source_experiment=33,
        feature_source_run_id="exp33-features-20261012-143026",
        initialisation="fresh seeds; never warm-start from a differently shaped Experiment 20 model",
    )
    return result


EXPERIMENT.contract_manifest = _manifest
globals().update(vars(EXPERIMENT))
