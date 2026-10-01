"""Experiment 8's complete driver/actor state, with a strict cache contract."""
from experiments.fhp.exp8_fhp_parallel_48h import training_state as base
from .diagnostics import CACHE_COUNTERS

SCHEMA_VERSION = base.SCHEMA_VERSION
STATE_TYPE = "exp11_fhp_betting_economics_full_training_state"
read_training_state = base.read_training_state
save_training_state = base.save_training_state


def _cache_contract(solver):
    return {"member_flags": [m.cache_frozen_targets for m in solver.q_value_trainer.members],
            "lifetime": "this_fit_only"}


def build_training_state(*args, **kwargs):
    solver = args[0] if args else kwargs["solver"]
    payload = base.build_training_state(*args, **kwargs)
    payload["type"] = STATE_TYPE
    payload["critic_cache"] = {
        "contract": _cache_contract(solver),
        "counters": {name: getattr(solver, name) for name in CACHE_COUNTERS},
    }
    # Do not persist computed targets: their lifetime is one completed fit.
    return payload


def restore_training_state(solver, payload, **kwargs):
    if payload.get("type") != STATE_TYPE:
        raise ValueError("Unsupported Experiment 11 full-training-state schema")
    if payload["critic_cache"]["contract"] != _cache_contract(solver):
        raise ValueError("Continuation critic-cache contract differs")
    compatible = dict(payload, type=base.STATE_TYPE)
    captured = base.restore_training_state(solver, compatible, **kwargs)
    for name in CACHE_COUNTERS:
        setattr(solver, name, payload["critic_cache"]["counters"][name])
    return captured
