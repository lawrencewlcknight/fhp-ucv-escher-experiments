"""Prespecified diagnostic budget; independent units are source training seeds."""
from experiments.fhp.exp16_fhp_hand_board_48h.contract import (
    SOURCE_RUN_ID, TRAINING_REF, CONFIG_SHA256, SEEDS, RUNTIME, worker_name,
)

EXPERIMENT_ID = 17
EXPERIMENT_NAME = "exp17_fhp_critic_budget"


def contract(smoke=False):
    return dict(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        source_run_id=SOURCE_RUN_ID, source_commit=TRAINING_REF,
        source_config_sha256=CONFIG_SHA256, seeds=[0] if smoke else list(SEEDS),
        smoke=smoke, updates=[1, 2] if smoke else [5000, 10000],
        boundaries=2 if smoke else 3,
        probes_per_player_fold=1 if smoke else 16,
        repeats_per_probe=4 if smoke else 128,
        training_probe_rows=16 if smoke else 8192,
        diagnostic_seed=1700401, learner_threads=1 if smoke else 8,
        continuation="unchanged 10000-update control; three consecutive post-24h iterations",
        pairing="one native fit; shared initial model/Adam, replay, targets and minibatch prefix",
        variance_estimand="mean within-full-history legal-action advantage variance; equal probe weights",
        sampling="fresh independent trajectories; two players x two held-out critic folds",
        heldout_error="sampled-action frozen-TD MSE, not oracle action-value error",
        inference="average boundaries/folds/probes within each source seed first",
        retention="JSON diagnostics and small critic-only weights; no full replay/training states",
        acceptance="screening only; no automatic promotion or claim of equivalent policy strength",
    )
