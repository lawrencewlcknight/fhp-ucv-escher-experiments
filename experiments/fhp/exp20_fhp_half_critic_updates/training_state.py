"""Full driver/actor state with a distinct Experiment 20 identity and strict resume."""
from experiments.fhp.exp10_fhp_hand_board_features import training_state as base

SCHEMA_VERSION = base.SCHEMA_VERSION
STATE_TYPE = "exp20_fhp_half_critic_updates_full_training_state"
read_training_state = base.read_training_state
save_training_state = base.save_training_state


def build_training_state(*args, **kwargs):
    payload = base.build_training_state(*args, **kwargs)
    payload["type"] = STATE_TYPE
    return payload


def restore_training_state(solver, payload, **kwargs):
    if payload.get("type") != STATE_TYPE:
        raise ValueError("Unsupported Experiment 20 full-training-state schema")
    return base.restore_training_state(solver, dict(payload, type=base.STATE_TYPE), **kwargs)
