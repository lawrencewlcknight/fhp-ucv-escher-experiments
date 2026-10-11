"""Prespecified development protocol, not an adaptive-training experiment."""
from experiments.fhp.exp22_fhp_critic_objectives.config import contract as previous

STAGES = ("one_hour", "six_hours", "late_24h_plus_one_iteration")
UPDATES = (0, 100, 500, 1000, 2500, 5000, 10000)
SEED = 0


def contract(smoke=False, stage="one_hour"):
    if stage not in STAGES:
        raise ValueError("Unknown pre-fit stage")
    result = previous(smoke)
    result.update(experiment_id=24, linked_experiment_id=25,
        experiment_name="exp24_fhp_critic_budget_stages", stage=stage,
        updates=[0, 1, 2, 3] if smoke else list(UPDATES),
        random_seed=240110 + STAGES.index(stage)*1000, native_replay_rng=True,
        fixed_replay_probe_rows=8 if smoke else 4096,
        fixed_fit_panel_diagnostics=True,
        phase="before_ensemble_fit_after_collection_regret_and_calibration",
        deployment="zero: unchanged target; positive: one append to common native parameter-average history",
        primary_question="ordinary replay TD budget at pre-fit boundaries",
        secondary_question="fixed-panel generalisation of matched TD / weighted TD / corrected-regret variance",
        inference="one source seed; stages, folds and roots are not independent seed replicates",
        selection="minimum deployed selection variance; exact ties prefer shorter budget; final assessment locked out",
        limitation="late stage is a historical 24h state plus one iteration, not the same longitudinal trajectory",
        interpretation="conditional within-history variance only; no policy-quality or adaptive-controller claim")
    return result
