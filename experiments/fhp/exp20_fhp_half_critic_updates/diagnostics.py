"""Retain native timers and expose the actual per-member fitting configuration."""
from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import (
    CACHE_COUNTERS, install_cache, execution_diagnostics as base_diagnostics,
)


def execution_diagnostics(solver):
    result = base_diagnostics(solver)
    result["critic_train_steps_per_fit"] = [int(m.train_steps) for m in solver.q_value_trainer.members]
    return result
