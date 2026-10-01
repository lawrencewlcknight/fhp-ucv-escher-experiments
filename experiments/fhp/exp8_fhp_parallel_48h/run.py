"""CLI for FHP Experiment 8 training, aggregation, and smoke validation."""

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
    DEFAULT_TOTAL_HOURS,
)
from .worker import run_worker


def _run_task(*, task_index: int, output_root: Path, smoke: bool, resume: bool,
              total_hours=DEFAULT_TOTAL_HOURS, source_worker=None):
    seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    schedule = checkpoint_schedule(smoke=smoke, total_hours=total_hours)
    config = smoke_config() if smoke else deepcopy(EXPERIMENT_CONFIG)
    validate_contract(seeds=seeds, schedule=schedule, config=config, smoke=smoke,
                      total_hours=total_hours)
    tasks = task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} is outside 0..{len(tasks) - 1}")
    _, seed = tasks[task_index]
    worker_dir = Path(output_root) / "workers" / task_name(task_index, seed)
    if source_worker is not None:
        from .continuation import stage_source
        stage_source(str(source_worker), worker_dir, total_hours=total_hours,
                     seed=seed, smoke=smoke)
    return run_worker(
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        config=config,
        smoke=smoke,
        resume=resume,
        total_hours=total_hours,
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
    worker.add_argument("--total-hours", type=int, default=DEFAULT_TOTAL_HOURS)
    worker.add_argument("--source-worker", type=Path)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--total-hours", type=int, default=DEFAULT_TOTAL_HOURS)
    capacity = subparsers.add_parser("capacity-preflight")
    capacity.add_argument("--output-root", type=Path, required=True)
    source = subparsers.add_parser("stage-source")
    source.add_argument("--source", required=True)
    source.add_argument("--destination", type=Path, required=True)
    source.add_argument("--seed", type=int, required=True)
    source.add_argument("--total-hours", type=int, required=True)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--total-hours", type=int, default=DEFAULT_TOTAL_HOURS)
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
        if (not resumed["resumed_from_training_state"]
                or resumed["final_nodes_touched"] != result["final_nodes_touched"]):
            raise RuntimeError("Smoke continuation changed the completed checkpoint")
        aggregate_workers(args.output_root, seeds=SMOKE_SEEDS)
        # Explicitly test extending a COMPLETED final checkpoint, not merely
        # restarting an interrupted run or reloading a policy for inference.
        extended_root = args.output_root / "extension_smoke"
        extended = _run_task(
            task_index=0, output_root=extended_root, smoke=True, resume=True,
            total_hours=72,
            source_worker=args.output_root / "workers" / task_name(0, 0))
        if (not extended["resumed_from_training_state"]
                or extended["final_iteration"] <= result["final_iteration"]
                or extended["checkpoint_count"] != 12):
            raise RuntimeError("Completed-endpoint extension smoke failed")
        aggregate_workers(extended_root, seeds=SMOKE_SEEDS, total_hours=72)
        print(json.dumps(result, indent=2))
    elif args.command == "stage-source":
        from .continuation import stage_source
        print(json.dumps(stage_source(args.source, args.destination,
                                     total_hours=args.total_hours, seed=args.seed), indent=2))
    elif args.command == "capacity-preflight":
        from .preflight import capacity_preflight
        print(json.dumps(capacity_preflight(args.output_root), indent=2))
    elif args.command == "worker":
        result = _run_task(
            task_index=args.task_index,
            output_root=args.output_root,
            smoke=False,
            resume=not args.no_resume,
            total_hours=args.total_hours,
            source_worker=args.source_worker,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "aggregate":
        print(aggregate_workers(args.output_root, total_hours=args.total_hours))
    else:
        print(json.dumps(contract_manifest(total_hours=args.total_hours), indent=2))


if __name__ == "__main__":
    main()
