"""Run archived Experiment 2: time-bound sequential FHP UCV-ESCHER."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path

from experiments.fhp.exp1_archived_ucv_escher_baseline.run import (
    run_experiment as run_time_bound_experiment,
)
from unbiased_escher import UnbiasedControlVariateEscher

from .config import (
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    CHECKPOINT_TRAINING_SECONDS,
    DEFAULT_SEED,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    REFERENCE_VM,
    UCV_CONFIG,
    smoke_config,
)


def run_experiment(
    *,
    seed: int,
    config,
    checkpoint_training_seconds,
    output_root: Path,
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
        solver_class=UnbiasedControlVariateEscher,
        checkpoint_prefix="exp2_archived_fhp_ucv_escher_sequential",
        execution_backend="sequential",
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
    )
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
