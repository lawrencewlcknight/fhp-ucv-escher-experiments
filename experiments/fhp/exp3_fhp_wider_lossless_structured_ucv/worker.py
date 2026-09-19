"""Train one seed of FHP Experiment 3."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import (
    _make_solver as _make_structured_solver,
    run_worker as _run_structured_worker,
)

from .config import (
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PRODUCTION_SEEDS,
    REFERENCE_VM,
    SMOKE_SEEDS,
    validate_contract,
)
from .training_state import (
    build_training_state,
    read_training_state,
    restore_training_state,
    save_training_state,
)


def _make_solver(seed: int, config: Mapping):
    return _make_structured_solver(seed, config)


def run_worker(
    *,
    seed: int,
    schedule: Sequence[Mapping],
    worker_dir: Path,
    config: Mapping,
    smoke: bool,
    resume: bool,
) -> dict:
    return _run_structured_worker(
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        config=config,
        smoke=smoke,
        resume=resume,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID,
        algorithm_label=ALGORITHM_LABEL,
        production_seeds=PRODUCTION_SEEDS,
        smoke_seeds=SMOKE_SEEDS,
        reference_vm=REFERENCE_VM,
        contract_validator=validate_contract,
        training_state_builder=build_training_state,
        training_state_reader=read_training_state,
        training_state_restorer=restore_training_state,
        training_state_saver=save_training_state,
        remote_task_env="EXP3_REMOTE_TASK_URI",
        experiment_label="Experiment 3",
    )


__all__ = ["_make_solver", "run_worker"]
