"""Keep the cached parallel solver's established timing and memory counters."""
from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import (
    CACHE_COUNTERS, execution_diagnostics as _execution_diagnostics, install_cache,
)


def execution_diagnostics(solver):
    result = _execution_diagnostics(solver)
    result["critic_train_steps_per_fit"] = [int(m.train_steps) for m in solver.q_value_trainer.members]
    return result
