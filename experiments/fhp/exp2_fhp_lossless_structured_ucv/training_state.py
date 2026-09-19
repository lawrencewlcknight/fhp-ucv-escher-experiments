"""Continuation-state schema for FHP Experiment 2."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline import training_state as _base


SCHEMA_VERSION = _base.SCHEMA_VERSION
STATE_TYPE = "exp2_fhp_lossless_structured_full_training_state"


def build_training_state(*args, **kwargs):
    solver = args[0] if args else kwargs["solver"]
    payload = _base.build_training_state(*args, **kwargs)
    payload["type"] = STATE_TYPE
    payload["feature_encoder"] = solver.feature_encoder.metadata()
    payload["replay_rng"] = {
        "average_policy": deepcopy(
            solver.ave_policy_trainer.buffer.rng.bit_generator.state
        ),
        "q_members": [
            deepcopy(member.buffer.rng.bit_generator.state)
            for member in solver.q_value_trainer.members
        ],
    }
    return payload


def restore_training_state(solver, payload: Mapping, **kwargs):
    if payload.get("type") != STATE_TYPE:
        raise ValueError("Unsupported Experiment 2 full-training-state schema")
    if payload.get("feature_encoder") != solver.feature_encoder.metadata():
        raise ValueError("Training-state feature encoder differs from the solver")
    compatible = dict(payload)
    compatible["type"] = _base.STATE_TYPE
    captured = _base.restore_training_state(solver, compatible, **kwargs)
    replay_rng = payload["replay_rng"]
    solver.ave_policy_trainer.buffer.rng.bit_generator.state = deepcopy(
        replay_rng["average_policy"]
    )
    if len(replay_rng["q_members"]) != len(solver.q_value_trainer.members):
        raise ValueError("Training-state critic replay RNG count differs")
    for member, state in zip(solver.q_value_trainer.members, replay_rng["q_members"]):
        member.buffer.rng.bit_generator.state = deepcopy(state)
    return captured


read_training_state = _base.read_training_state
save_training_state = _base.save_training_state


__all__ = [
    "SCHEMA_VERSION",
    "STATE_TYPE",
    "build_training_state",
    "read_training_state",
    "restore_training_state",
    "save_training_state",
]
