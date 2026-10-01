"""Four architectures, matched data/budgets, equal validation-only tuning."""

from copy import deepcopy
from experiments.fhp.exp13_fhp_policy_capacity.config import contract as capacity_contract
from fhp_escher.card_policy import MODEL_TYPE

ARCHITECTURES = {
    "standard": {"network_layers": [192, 192], "branch_width": 64, "parameters": 74243},
    "dense": {"model_type": MODEL_TYPE, "kind": "dense", "parameters": 133486},
    "deepsets": {"model_type": MODEL_TYPE, "kind": "deepsets", "parameters": 133478},
    "attention": {"model_type": MODEL_TYPE, "kind": "attention", "parameters": 133094},
}
COMPARISONS = [["deepsets", "dense"], ["attention", "dense"],
               ["dense", "standard"], ["deepsets", "standard"], ["attention", "standard"]]


def contract(smoke=False):
    config = capacity_contract(smoke)
    config.update(
        experiment_id=14, experiment_name="exp14_fhp_card_architecture",
        architectures=deepcopy(ARCHITECTURES), comparisons=deepcopy(COMPARISONS),
        remote_worker_env="EXP14_REMOTE_WORKER", policy_id_prefix="card_architecture",
        deduplicate_identical_evaluation=True,
        initialisation_seed_base=202610140, sampler_seed_base=202610141,
        evaluation_seed_base=202610142, split_seed=20261014,
        split_unit="canonical_visible_private_and_public_cards",
        split_strata=["round", "made_hand_category", "card_configuration_frequency_1_2to4_5plus"],
        analysis_title="Frozen FHP card-architecture audit",
        interpretation=("Three reused source trajectories, not six seeds. Primary contrast: Deep Sets minus "
                        "parameter-matched dense residual at 20k. Attention, 60k and tuned contrasts are secondary. "
                        "Intervals are exploratory. LBR is not exact exploitability. No end-to-end promotion is automatic."),
        analysis_intro=("Four output-policy architectures reuse identical Experiment 2 replay. "
                        "Diagnostic splits keep all betting histories and player labels for the same canonical "
                        "visible private/public cards together. Fixed fits share Adam 0.003 and minibatch sequences; "
                        "each architecture gets the same four-recipe validation-only screen. "
                        "Test scores belong to diagnostic fits; gameplay uses separate full-replay refits. "
                        "The exact canonical baseline remains trainable in every residual candidate. "
                        "Dense/Deep Sets/attention candidates differ by less than 0.3% in parameter count. "),
    )
    return config
