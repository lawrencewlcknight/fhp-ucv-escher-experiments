"""Train one seed of FHP Experiment 11."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
from functools import partial
import json
import os
import platform

import numpy as np
import torch
import ray
from fhp_escher.checkpointing import sha256_file
from fhp_escher.betting_economics_features import FHPBettingEconomicsFeatureEncoder

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
    DEFAULT_TOTAL_HOURS,
)
from .diagnostics import execution_diagnostics, install_cache
from .training_state import (
    build_training_state,
    read_training_state,
    restore_training_state,
    save_training_state,
)


def _make_solver(seed: int, config: Mapping, *, smoke: bool = False):
    kwargs = _solver_kwargs(seed, config)
    cache_enabled = kwargs.pop("cache_frozen_critic_targets")
    if not isinstance(cache_enabled, bool):
        raise ValueError("cache_frozen_critic_targets must be a boolean")
    solver = ParallelStructuredGroupedUCVEscher(
        **kwargs, **parallel_settings(smoke=smoke),
        parallel_run_seed=seed,
    )
    install_cache(solver, enabled=cache_enabled)
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
    total_hours: int = DEFAULT_TOTAL_HOURS,
) -> dict:
    # Keep Experiment 2's fitting concurrency, not one thread per VM vCPU.
    torch.set_num_threads(1 if smoke else LEARNER_THREADS)
    worker_dir = Path(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    prior_path = worker_dir / "run_manifest.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior["checkpoint_schedule"] != list(schedule):
            raise ValueError("Use a new run directory to extend a completed run")
        if not resume:
            raise ValueError("Refusing to overwrite an existing run without resume")
    lineage_path = worker_dir / "continuation_source.json"
    initial_resume_state = None
    if lineage_path.exists() and not prior_path.exists():
        lineage = json.loads(lineage_path.read_text())
        if lineage.get("local_state_path") != "continuation_inputs/source_state.pt":
            raise ValueError("Invalid temporary continuation input path")
        initial_resume_state = worker_dir / lineage["local_state_path"]
        if sha256_file(initial_resume_state) != lineage["source_state_sha256"]:
            raise ValueError("Imported continuation state is corrupt")
    if total_hours != DEFAULT_TOTAL_HOURS:
        if not lineage_path.exists() or not resume:
            raise ValueError("An extended horizon requires an imported continuation state")
        if json.loads(lineage_path.read_text())["total_hours"] != total_hours:
            raise ValueError("Extension horizon differs from imported source contract")
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "reference_vm": dict(REFERENCE_VM),
        "logical_cpus_visible": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "traversal_execution": "ray_parallel",
        "parallel_settings": parallel_settings(smoke=smoke),
        "ray_version": ray.__version__,
        "frozen_critic_target_cache": True,
        "feature_encoder": FHPBettingEconomicsFeatureEncoder().metadata(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    runtime_path = worker_dir / "runtime_manifest.json"
    if resume and runtime_path.exists():
        previous = json.loads(runtime_path.read_text())
        for key in ("experiment_id", "reference_vm", "torch_intraop_threads",
                    "torch_version", "numpy_version", "python_version",
                    "ray_version", "parallel_settings", "frozen_critic_target_cache",
                    "feature_encoder"):
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
            contract_validator=partial(validate_contract, total_hours=total_hours),
            training_state_builder=build_training_state,
            training_state_reader=read_training_state,
            training_state_restorer=restore_training_state,
            training_state_saver=save_training_state,
            remote_task_env="EXP11_REMOTE_TASK_URI",
            experiment_label="Experiment 11",
            training_state_retention="final",
            initial_resume_state=initial_resume_state,
            solver_factory=factory,
            execution_backend="ray_parallel",
            extra_diagnostics_fn=execution_diagnostics,
            require_resume_state=(lineage_path.exists()
                                  or (worker_dir / "checkpoint_manifest.json").exists()),
        )
    finally:
        for solver in solvers:
            solver.close()
        if initial_resume_state is not None and initial_resume_state.is_file():
            # This is a validated private copy, never the archived source.
            initial_resume_state.unlink()
            initial_resume_state.parent.rmdir()


__all__ = ["_make_solver", "run_worker"]
