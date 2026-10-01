"""Train one seed of FHP Experiment 7."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import json
import os
import platform

import numpy as np
import torch
import ray

from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import (
    _solver_kwargs,
    run_worker as _run_structured_worker,
)
from unbiased_escher.grouped_parallel_solver import ParallelStructuredGroupedUCVEscher

from .config import (
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    LEARNER_THREADS,
    PRODUCTION_SEEDS,
    REFERENCE_VM,
    SMOKE_SEEDS,
    validate_contract,
    parallel_settings,
)
from .diagnostics import execution_diagnostics
from .training_state import (
    build_training_state,
    read_training_state,
    restore_training_state,
    save_training_state,
)


def _make_solver(seed: int, config: Mapping, *, smoke: bool = False):
    solver = ParallelStructuredGroupedUCVEscher(
        **_solver_kwargs(seed, config), **parallel_settings(smoke=smoke),
        parallel_run_seed=seed,
    )
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(config["early_evaluation_node_thresholds"])
    solver.target_nodes_touched = None
    solver.max_wall_clock_seconds = None
    return solver


def run_worker(
    *,
    seed: int,
    schedule: Sequence[Mapping],
    worker_dir: Path,
    config: Mapping,
    smoke: bool,
    resume: bool,
) -> dict:
    # Keep Experiment 2's fitting concurrency, not one thread per VM vCPU.
    torch.set_num_threads(1 if smoke else LEARNER_THREADS)
    worker_dir = Path(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "reference_vm": dict(REFERENCE_VM),
        "logical_cpus_visible": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "traversal_execution": "ray_parallel",
        "parallel_settings": parallel_settings(smoke=smoke),
        "ray_version": ray.__version__,
        "frozen_critic_target_cache": False,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    runtime_path = worker_dir / "runtime_manifest.json"
    if resume and runtime_path.exists():
        previous = json.loads(runtime_path.read_text())
        for key in ("experiment_id", "reference_vm", "torch_intraop_threads",
                    "torch_version", "numpy_version", "python_version",
                    "ray_version", "parallel_settings"):
            if previous[key] != runtime[key]:
                raise ValueError(f"Resume runtime differs at {key}")
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    solvers = []

    def factory(seed, config):
        solver = _make_solver(seed, config, smoke=smoke)
        solvers.append(solver)
        return solver

    try:
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
            remote_task_env="EXP7_REMOTE_TASK_URI",
            experiment_label="Experiment 7",
            training_state_retention="none",
            solver_factory=factory,
            execution_backend="ray_parallel",
            extra_diagnostics_fn=execution_diagnostics,
        )
    finally:
        for solver in solvers:
            solver.close()


__all__ = ["_make_solver", "run_worker"]
