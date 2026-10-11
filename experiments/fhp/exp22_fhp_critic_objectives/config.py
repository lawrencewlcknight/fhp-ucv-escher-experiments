"""Prespecified late-stage screen. Changing this contract requires a new run."""
from experiments.fhp.exp17_fhp_critic_budget.config import SOURCE_RUN_ID, TRAINING_REF

OBJECTIVES = ("replay_td", "matched_td", "weighted_td", "regret_variance")


def contract(smoke=False):
    return dict(
        experiment_id=22, experiment_name="exp22_fhp_critic_objectives",
        source_run_id=SOURCE_RUN_ID, source_commit=TRAINING_REF, smoke=smoke,
        objectives=list(OBJECTIVES), updates=[1, 2, 3] if smoke else [2500, 5000, 10000],
        # Each member sees equal numbers of roots for the two traversers.
        split_roots={"fit": 4 if smoke else 256, "selection": 4 if smoke else 64,
                     "assessment": 4 if smoke else 128},
        split_repeats={"fit": 4 if smoke else 32, "selection": 4 if smoke else 64,
                       "assessment": 4 if smoke else 128},
        pair_batch_size=2 if smoke else 32, evaluation_batch_size=64,
        fit_threads=1 if smoke else 8, diagnostic_threads=1, random_seed=2200411,
        max_trace_decisions=64, stage="frozen_24h_only",
        estimator="native recursive beta=1 control variate, then policy centring",
        weighting="weighted_td: 1/(number_of_legal_actions * root_action_probability)",
        variance_loss="half squared difference of independent paired root advantages, mean over legal actions",
        fitting="online candidate; frozen policies/proposals/old TD targets; no target refresh within fit",
        deployment="one hypothetical native completed fit appended to common parameter-average history",
        selection="minimum selection-panel deployed variance within objective; shorter budget breaks ties",
        inference="source seeds are units; folds/histories/continuations are not independent seeds",
        interpretation="conditional within-history probe variance, not global CFR variance or exploitability",
        native_control="replay_td retains native replay minibatch size; other arms share paired-root minibatches",
        compute="equal update counts are not equal compute; charge each method its required data cost",
    )
