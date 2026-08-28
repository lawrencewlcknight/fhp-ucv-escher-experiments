"""Pure helpers for deterministic UCV-ESCHER worker orchestration."""

from __future__ import annotations


WORKER_SEED_STRIDE = 1_000_003


def worker_seed(run_seed: int, worker_index: int) -> int:
    """Return one deterministic, distinct traversal-worker seed."""
    worker_index = int(worker_index)
    if worker_index < 0:
        raise ValueError("worker_index must be non-negative")
    return int(run_seed) + WORKER_SEED_STRIDE * (worker_index + 1)


def partition_total(total: int, parts: int) -> list[int]:
    """Split an integer budget exactly with at most one item of imbalance."""
    total = int(total)
    parts = int(parts)
    if total < 0:
        raise ValueError("total must be non-negative")
    if parts <= 0:
        raise ValueError("parts must be positive")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]
