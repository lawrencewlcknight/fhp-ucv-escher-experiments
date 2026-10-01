"""Passive per-fit cache instrumentation; all existing parallel timers retained."""
import time

from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.diagnostics import (
    execution_diagnostics as parallel_diagnostics,
)

CACHE_COUNTERS = (
    "_cumulative_critic_fit_seconds", "_cumulative_critic_cache_build_seconds",
    "_cumulative_critic_fit_calls", "_cumulative_cached_critic_fit_calls",
    "_cumulative_cached_critic_rows", "_peak_critic_cache_bytes",
)


def install_cache(solver, *, enabled):
    for name in CACHE_COUNTERS:
        setattr(solver, name, 0)
    for member in solver.q_value_trainer.members:
        member.cache_frozen_targets = enabled
        _instrument_member(solver, member)


def _instrument_member(solver, member):
    original = member.train_model

    def timed_fit(iteration):
        started = time.perf_counter()
        eligible = (len(member.buffer) > 0 and member.train_steps > 0
                    and (member.batch_size < 0 or len(member.buffer) >= member.batch_size))
        try:
            result = original(iteration)
            solver._cumulative_critic_fit_calls += 1
            if member.cache_frozen_targets and eligible:
                stats = member.last_target_cache_stats
                if stats["cache_lifetime"] != "this_fit_only":
                    raise RuntimeError("Frozen targets were not rebuilt for this fit")
                solver._cumulative_cached_critic_fit_calls += 1
                solver._cumulative_cached_critic_rows += int(stats["cached_rows"])
                solver._cumulative_critic_cache_build_seconds += float(stats["cache_build_seconds"])
                solver._peak_critic_cache_bytes = max(solver._peak_critic_cache_bytes,
                                                     int(stats["cache_bytes"]))
            return result
        finally:
            solver._cumulative_critic_fit_seconds += time.perf_counter() - started

    member.train_model = timed_fit


def execution_diagnostics(solver):
    result = parallel_diagnostics(solver)
    result.update({name.lstrip("_"): getattr(solver, name) for name in CACHE_COUNTERS})
    result["parallel_frozen_critic_target_cache"] = all(
        m.cache_frozen_targets for m in solver.q_value_trainer.members)
    result["critic_timer_interpretation"] = (
        "critic_fit_seconds is a subset of parallel learner time; "
        "cache_build_seconds is a subset of critic_fit_seconds, not additional time"
    )
    return result
