"""Continuation-state schema for FHP Experiment 3."""

from __future__ import annotations

from typing import Mapping

from experiments.fhp.exp2_fhp_lossless_structured_ucv import training_state as _base


SCHEMA_VERSION = _base.SCHEMA_VERSION
STATE_TYPE = "exp3_fhp_wider_lossless_structured_full_training_state"


def build_training_state(*args, **kwargs):
    payload = _base.build_training_state(*args, **kwargs)
    payload["type"] = STATE_TYPE
    return payload


def restore_training_state(solver, payload: Mapping, **kwargs):
    if payload.get("type") != STATE_TYPE:
        raise ValueError("Unsupported Experiment 3 full-training-state schema")
    compatible = dict(payload)
    compatible["type"] = _base.STATE_TYPE
    return _base.restore_training_state(solver, compatible, **kwargs)


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
