"""Continuation-state schema for FHP Experiment 7."""

from __future__ import annotations

from typing import Mapping
from copy import deepcopy

from experiments.fhp.exp2_fhp_lossless_structured_ucv import training_state as _base
from .diagnostics import COUNTERS


SCHEMA_VERSION = _base.SCHEMA_VERSION
STATE_TYPE = "exp7_fhp_parallel_structured_n2_standard16_full_training_state"


def build_training_state(*args, **kwargs):
    solver = args[0] if args else kwargs["solver"]
    payload = _base.build_training_state(*args, **kwargs)
    payload["type"] = STATE_TYPE
    payload["parallel"] = {
        "contract": _actor_contract(solver),
        "actors": solver._ray.get([w.continuation_state.remote() for w in solver._workers]),
        "counters": {name: getattr(solver, name, 0) for name in COUNTERS},
        # Regret rows are transient, but each compact sampler's private RNG
        # persists across reset_buffer() and is required for exact continuation.
        "regret_replay_rng": [deepcopy(t.buffer.rng.bit_generator.state)
                              for t in solver.regret_trainers],
    }
    return payload


def _actor_contract(solver):
    return {key: getattr(solver, key) for key in (
        "_parallel_num_workers", "_parallel_run_seed", "_parallel_collection_chunk_size",
        "_parallel_cache_actor_snapshots", "_parallelize_independent_learners",
    )}


def restore_training_state(solver, payload: Mapping, **kwargs):
    if payload.get("type") != STATE_TYPE:
        raise ValueError("Unsupported Experiment 7 full-training-state schema")
    state = payload["parallel"]
    if state["contract"] != _actor_contract(solver):
        raise ValueError("Parallel continuation contract differs")
    if len(state["actors"]) != len(solver._workers):
        raise ValueError("Parallel continuation actor count differs")
    compatible = dict(payload)
    compatible["type"] = _base.STATE_TYPE
    captured = _base.restore_training_state(solver, compatible, **kwargs)
    solver._ray.get([worker.restore_continuation_state.remote(actor_state)
                     for worker, actor_state in zip(solver._workers, state["actors"])])
    for name in COUNTERS:
        setattr(solver, name, state["counters"][name])
    for trainer, rng in zip(solver.regret_trainers, state["regret_replay_rng"], strict=True):
        trainer.buffer.rng.bit_generator.state = deepcopy(rng)
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
