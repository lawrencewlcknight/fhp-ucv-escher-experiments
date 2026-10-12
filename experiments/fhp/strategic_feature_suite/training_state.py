"""Full learner, replay, RNG and eight-actor states, guarded by feature identity."""
from experiments.fhp.exp10_fhp_hand_board_features import training_state as base

read_training_state = base.read_training_state
save_training_state = base.save_training_state


def build_training_state(*args, experiment, **kwargs):
    payload = base.build_training_state(*args, **kwargs)
    payload["type"] = experiment.EXPERIMENT_NAME + "_full_training_state"
    payload["strategic_feature_contract"] = experiment.encoder().metadata()
    return payload


def restore_training_state(solver, payload, *, experiment, **kwargs):
    if (payload.get("type") != experiment.EXPERIMENT_NAME + "_full_training_state"
            or payload.get("strategic_feature_contract") != experiment.encoder().metadata()):
        raise ValueError("Incompatible strategic feature continuation contract")
    return base.restore_training_state(solver, dict(payload, type=base.STATE_TYPE), **kwargs)
