"""CLI for FHP Experiment 7 training, aggregation, and smoke validation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path

from .aggregate import aggregate_workers, task_name
from .config import (
    EXPERIMENT_CONFIG,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    contract_manifest,
    smoke_config,
    task_schedule,
    validate_contract,
)
from .worker import run_worker


def _run_task(*, task_index: int, output_root: Path, smoke: bool, resume: bool):
    seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    schedule = checkpoint_schedule(smoke=smoke)
    config = smoke_config() if smoke else deepcopy(EXPERIMENT_CONFIG)
    validate_contract(seeds=seeds, schedule=schedule, config=config, smoke=smoke)
    tasks = task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} is outside 0..{len(tasks) - 1}")
    _, seed = tasks[task_index]
    worker_dir = Path(output_root) / "workers" / task_name(task_index, seed)
    return run_worker(
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        config=config,
        smoke=smoke,
        resume=resume,
    )


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--no-resume", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    capacity = subparsers.add_parser("capacity-preflight")
    capacity.add_argument("--output-root", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-root", type=Path, required=True)
    compare.add_argument("--parallel-root", type=Path, required=True)
    compare.add_argument("--output-root", type=Path, required=True)
    subparsers.add_parser("contract")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        from .smoke import verify_continuation
        verify_continuation(args.output_root / "restart_validation")
        result = _run_task(
            task_index=0,
            output_root=args.output_root,
            smoke=True,
            resume=not args.no_resume,
        )
        resumed = _run_task(
            task_index=0, output_root=args.output_root, smoke=True, resume=True,
        )
        if (not resumed.get("reused_completed_outputs")
                or resumed["final_nodes_touched"] != result["final_nodes_touched"]):
            raise RuntimeError("Smoke reuse changed the completed checkpoint")
        aggregate_workers(args.output_root, seeds=SMOKE_SEEDS)
        print(json.dumps(result, indent=2))
    elif args.command == "compare":
        from .compare import compare_runs
        print(json.dumps(compare_runs(args.baseline_root, args.parallel_root, args.output_root), indent=2))
    elif args.command == "capacity-preflight":
        from .preflight import capacity_preflight
        print(json.dumps(capacity_preflight(args.output_root), indent=2))
    elif args.command == "worker":
        result = _run_task(
            task_index=args.task_index,
            output_root=args.output_root,
            smoke=False,
            resume=not args.no_resume,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "aggregate":
        print(aggregate_workers(args.output_root))
    else:
        print(json.dumps(contract_manifest(), indent=2))


if __name__ == "__main__":
    main()
