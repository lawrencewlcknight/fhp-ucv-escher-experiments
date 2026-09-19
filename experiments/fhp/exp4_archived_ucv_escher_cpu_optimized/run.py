"""Run archived Experiment 4: CPU-optimised parallel FHP UCV-ESCHER."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path

from experiments.fhp.exp1_archived_ucv_escher_baseline.run import (
    run_experiment as run_time_bound_experiment,
)
from unbiased_escher.efficient_parallel_solver import (
    EfficientParallelUnbiasedControlVariateEscher,
)

from .config import (
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    CHECKPOINT_TRAINING_SECONDS,
    DEFAULT_SEED,
    EFFICIENCY_CHANGES,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PARALLEL_COLLECTION_CHUNK_SIZE,
    PARALLEL_LEARNER_INTRAOP_THREADS,
    PARALLEL_LEARNER_THREADS,
    PARALLEL_NUM_WORKERS,
    PARALLEL_RAY_OBJECT_STORE_MEMORY,
    REFERENCE_VM,
    UCV_CONFIG,
    smoke_config,
)


def _parallel_solver_kwargs(seed: int, smoke: bool = False) -> dict:
    return {
        "parallel_num_workers": 2 if smoke else PARALLEL_NUM_WORKERS,
        "parallel_run_seed": int(seed),
        "parallel_log_to_driver": False,
        "parallel_ray_object_store_memory": (
            256 * 1024 * 1024
            if smoke
            else PARALLEL_RAY_OBJECT_STORE_MEMORY
        ),
        "parallelize_independent_learners": True,
        "parallel_learner_threads": 2 if smoke else PARALLEL_LEARNER_THREADS,
        "parallel_learner_intraop_threads": (
            1 if smoke else PARALLEL_LEARNER_INTRAOP_THREADS
        ),
        "parallel_collection_chunk_size": (
            2 if smoke else PARALLEL_COLLECTION_CHUNK_SIZE
        ),
        "parallel_cache_actor_snapshots": True,
    }


def run_experiment(
    *,
    seed: int,
    config,
    checkpoint_training_seconds,
    output_root: Path,
    smoke: bool = False,
) -> Path:
    return run_time_bound_experiment(
        seed=seed,
        config=config,
        checkpoint_training_seconds=checkpoint_training_seconds,
        output_root=output_root,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID,
        algorithm_label=ALGORITHM_LABEL,
        reference_vm=REFERENCE_VM,
        solver_class=EfficientParallelUnbiasedControlVariateEscher,
        solver_extra_kwargs=_parallel_solver_kwargs(seed, smoke),
        checkpoint_prefix="exp4_archived_fhp_ucv_escher_cpu_optimized",
        execution_backend="ray_parallel_cpu_optimized",
        implementation_provenance=EFFICIENCY_CHANGES,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        nargs=2,
        metavar=("FIRST", "FINAL"),
        default=CHECKPOINT_TRAINING_SECONDS,
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    config = smoke_config() if args.smoke else deepcopy(UCV_CONFIG)
    checkpoint_seconds = (
        (1e-6, 2e-6)
        if args.smoke and tuple(args.checkpoint_seconds) == CHECKPOINT_TRAINING_SECONDS
        else tuple(args.checkpoint_seconds)
    )
    run_dir = run_experiment(
        seed=args.seed,
        config=config,
        checkpoint_training_seconds=checkpoint_seconds,
        output_root=args.output_root,
        smoke=args.smoke,
    )
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
