"""Validate and aggregate the three Experiment 3 seed workers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from experiments.fhp.exp2_fhp_lossless_structured_ucv.aggregate import (
    aggregate_workers as _aggregate_structured_workers,
    task_name as _structured_task_name,
)

from .config import (
    ALGORITHM_ID,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PRODUCTION_SEEDS,
    contract_manifest,
)


def task_name(index: int, seed: int) -> str:
    return _structured_task_name(index, seed, algorithm_id=ALGORITHM_ID)


def aggregate_workers(
    output_root: Path,
    *,
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> Path:
    return _aggregate_structured_workers(
        output_root,
        seeds=seeds,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID,
        contract_manifest_fn=contract_manifest,
        experiment_label="Experiment 3",
    )


__all__ = ["aggregate_workers", "task_name"]
