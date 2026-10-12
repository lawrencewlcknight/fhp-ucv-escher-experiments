"""CLI for FHP strategic-feature experiment training, aggregation, and smoke validation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path

from .aggregate import aggregate_workers, task_name
from .worker import run_worker


def _run_task(*, task_index: int, output_root: Path, smoke: bool, resume: bool,
              total_hours=24, source_worker=None, experiment):
    seeds = experiment.SMOKE_SEEDS if smoke else experiment.PRODUCTION_SEEDS
    schedule = experiment.checkpoint_schedule(smoke=smoke, total_hours=total_hours)
    config = experiment.smoke_config() if smoke else deepcopy(experiment.EXPERIMENT_CONFIG)
    experiment.validate_contract(seeds=seeds, schedule=schedule, config=config, smoke=smoke,
                      total_hours=total_hours)
    tasks = experiment.task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} is outside 0..{len(tasks) - 1}")
    _, seed = tasks[task_index]
    worker_dir = Path(output_root) / "workers" / task_name(task_index, seed, experiment=experiment)
    if source_worker is not None:
        from .continuation import stage_source
        stage_source(str(source_worker), worker_dir, total_hours=total_hours,
                     seed=seed, smoke=smoke, experiment=experiment)
    return run_worker(
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        config=config,
        smoke=smoke,
        resume=resume,
        total_hours=total_hours, experiment=experiment,
    )


def main(argv=None, *, experiment) -> None:
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
    worker.add_argument("--total-hours", type=int, default=24)
    worker.add_argument("--source-worker", type=Path)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--total-hours", type=int, default=24)
    capacity = subparsers.add_parser("capacity-preflight")
    capacity.add_argument("--output-root", type=Path, required=True)
    baseline_check = subparsers.add_parser("baseline-preflight")
    baseline_check.add_argument("--baseline-run", type=Path, required=True)
    baseline_check.add_argument("--output-root", type=Path, required=True)
    source = subparsers.add_parser("stage-source")
    source.add_argument("--source", required=True)
    source.add_argument("--destination", type=Path, required=True)
    source.add_argument("--seed", type=int, required=True)
    source.add_argument("--total-hours", type=int, required=True)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--total-hours", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        from .feature_preflight import run_feature_preflight
        from .smoke import verify_continuation, verify_cache_equivalence
        run_feature_preflight(args.output_root / "feature_validation", experiment=experiment)
        verify_cache_equivalence(args.output_root / "cache_validation", experiment=experiment)
        verify_continuation(args.output_root / "restart_validation", experiment=experiment)
        result = _run_task(
            task_index=0, experiment=experiment,
            output_root=args.output_root,
            smoke=True,
            resume=not args.no_resume,
        )
        resumed = _run_task(
            task_index=0, experiment=experiment, output_root=args.output_root, smoke=True, resume=True,
        )
        if (not resumed["resumed_from_training_state"]
                or resumed["final_nodes_touched"] != result["final_nodes_touched"]):
            raise RuntimeError("Smoke continuation changed the completed checkpoint")
        aggregate_workers(args.output_root, seeds=experiment.SMOKE_SEEDS, experiment=experiment)
        # Explicitly test extending a COMPLETED final checkpoint, not merely
        # restarting an interrupted run or reloading a policy for inference.
        extended_root = args.output_root / "extension_smoke"
        extended = _run_task(
            task_index=0, experiment=experiment, output_root=extended_root, smoke=True, resume=True,
            total_hours=48,
            source_worker=args.output_root / "workers" / task_name(0, 0, experiment=experiment))
        if (not extended["resumed_from_training_state"]
                or extended["final_iteration"] <= result["final_iteration"]
                or extended["checkpoint_count"] != 8):
            raise RuntimeError("Completed-endpoint extension smoke failed")
        aggregate_workers(extended_root, seeds=experiment.SMOKE_SEEDS, total_hours=48, experiment=experiment)
        print(json.dumps(result, indent=2))
    elif args.command == "stage-source":
        from .continuation import stage_source
        print(json.dumps(stage_source(args.source, args.destination,
                                     total_hours=args.total_hours, seed=args.seed, experiment=experiment), indent=2))
    elif args.command == "capacity-preflight":
        from .preflight import capacity_preflight
        print(json.dumps(capacity_preflight(args.output_root, experiment=experiment), indent=2))
    elif args.command == "baseline-preflight":
        from .evaluate import discover
        records = discover(args.baseline_run, experiment, control=True)
        report = dict(status="passed", baseline_run_id=experiment.BASELINE_RUN_ID,
                      baseline_retrained=False, policy_count=len(records), checkpoints=records)
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "baseline_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Validated {len(records)} archived Experiment 10 policies; no baseline training.")
    elif args.command == "worker":
        result = _run_task(
            task_index=args.task_index, experiment=experiment,
            output_root=args.output_root,
            smoke=False,
            resume=not args.no_resume,
            total_hours=args.total_hours,
            source_worker=args.source_worker,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "aggregate":
        print(aggregate_workers(args.output_root, total_hours=args.total_hours, experiment=experiment))
    else:
        print(json.dumps(experiment.contract_manifest(total_hours=args.total_hours), indent=2))
