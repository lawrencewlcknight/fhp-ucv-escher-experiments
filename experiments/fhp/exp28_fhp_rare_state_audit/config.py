"""Prespecified development audit, not a new full training run."""
from experiments.fhp.exp16_fhp_hand_board_48h.contract import TRAINING_REF, worker_name

EXPERIMENT = "exp28_fhp_rare_state_audit"
SOURCES = {24: "exp10-features-20261001-161740", 48: "exp16-feat48-20261004-182051"}
PRIOR_AUDIT = "fhp-strat16-dev-20261010-194045"
SEEDS = (0, 1, 2)
ARMS = ("ordinary", "targeted", "random_control")
PANEL_SEED = 2026101228


def contract(smoke=False):
    return dict(experiment_id=28, experiment_name=EXPERIMENT, version=1,
        smoke=bool(smoke), seeds=[0] if smoke else list(SEEDS), source_runs=SOURCES,
        updates=[1, 2] if smoke else [1000, 5000], batch_size=64 if smoke else 2048,
        continuation_iterations=1 if smoke else 10, continuation_cap_seconds=7200,
        fit_replicates=1 if smoke else 2, targeted_mass_fraction=.20,
        minimum_target_count=2, minimum_target_ess=1.5, maximum_target_fold=.05,
        # This bound is by prefix, with deterministic selection, not worst-case loss.
        controls_per_board=2 if smoke else 12, hands_per_category=1 if smoke else 4,
        additional_dev_boards=2 if smoke else 8, assessment_boards=2 if smoke else 8,
        probe_positions=2 if smoke else 24, probe_hidden_hands=2 if smoke else 32,
        gameplay_deal_pairs=2 if smoke else 2000, random_seed=PANEL_SEED,
        fit_target="retained iteration-weighted replay only; no oracle labels",
        scientific_unit="source training seed; fitting replicates nested within seed",
        panel_scope="terminal call/fold; no future cards or decisions",
        selection="no winner selection; all arms and both prefixes assessed after fitting lock",
        untouched="new board suit classes absent from the previous audit, excluded from all refit replay",
        historical_limits="retained replay is not a visit log or an exact historical average",
        continuation="native learning config/order; instrumentation does not supply training rows")
