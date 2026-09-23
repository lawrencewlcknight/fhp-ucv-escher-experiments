"""CLI for FHP Experiment 1 training, aggregation, and smoke validation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import shutil

from .aggregate import aggregate_workers, task_name
from .config import (
    EXPERIMENT_35_CONFIG,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    contract_manifest,
    smoke_config,
    task_schedule,
    validate_contract,
)
from .worker import run_worker
from .worker import _make_solver
from .training_state import (
    build_training_state,
    read_sharded_training_state,
    save_sharded_training_state,
)


def _run_task(*, task_index: int, output_root: Path, smoke: bool, resume: bool):
    seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    schedule = checkpoint_schedule(smoke=smoke)
    config = smoke_config() if smoke else deepcopy(EXPERIMENT_35_CONFIG)
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


def _run_checkpoint_stress(output_root: Path) -> dict:
    """Exercise production-capacity replay checkpointing without training."""
    output_root = Path(output_root).resolve()
    result_dir = output_root / "checkpoint_stress"
    state_path = result_dir / "production_capacity.state"
    result_dir.mkdir(parents=True, exist_ok=True)
    solver = _make_solver(0, deepcopy(EXPERIMENT_35_CONFIG))

    average = solver.ave_policy_trainer.buffer
    average.cur_id = int(average.buffer_size)
    for member in solver.q_value_trainer.members:
        member.buffer.size = int(member.buffer.buffer_size)
        member.buffer.cur_id = 0
    calibration = solver.calibration_trainer
    if calibration is not None:
        calibration.buffer.size = int(calibration.buffer.capacity)
        calibration.buffer.cursor = 0

    payload = build_training_state(
        solver,
        seed=0,
        checkpoint_id="production_capacity_stress",
        repository_commit="checkpoint-stress",
        config=EXPERIMENT_35_CONFIG,
        captured_checkpoints=(),
    )
    storage = save_sharded_training_state(state_path, payload)
    del payload
    loaded = read_sharded_training_state(state_path, verify=False)
    average_rows = int(
        loaded["average_policy_trainer"]["buffer"]["infostate"].shape[0]
    )
    critic_rows = int(
        sum(
            member["buffer"]["history"].shape[0]
            for member in loaded["q_ensemble"]["members"]
        )
    )
    calibration_rows = int(loaded["calibration"]["buffer"]["features"].shape[0])
    if (
        average_rows != int(EXPERIMENT_35_CONFIG["ave_policy_buffer_size"])
        or critic_rows != int(EXPERIMENT_35_CONFIG["baseline_buffer_size"])
        or calibration_rows != int(EXPERIMENT_35_CONFIG["calibration_buffer_size"])
        or int(storage["size_bytes"]) < 3_000_000_000
    ):
        raise RuntimeError("Production-capacity checkpoint stress contract failed")
    result = {
        "status": "complete",
        "format": storage["format"],
        "serialized_size_bytes": int(storage["size_bytes"]),
        "average_policy_rows": average_rows,
        "critic_rows": critic_rows,
        "calibration_rows": calibration_rows,
    }
    # The stress state proves serialization at production capacity but is not
    # useful experimental output, so avoid uploading several synthetic GiB.
    del loaded, solver
    gc.collect()
    shutil.rmtree(state_path)
    (result_dir / "SUCCESS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


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

    checkpoint_smoke = subparsers.add_parser("checkpoint-smoke")
    checkpoint_smoke.add_argument("--output-root", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)

    subparsers.add_parser("contract")
    args = parser.parse_args(argv)

    if args.command == "smoke":
        summary = _run_task(
            task_index=0,
            output_root=args.output_root,
            smoke=True,
            resume=not args.no_resume,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "worker":
        summary = _run_task(
            task_index=args.task_index,
            output_root=args.output_root,
            smoke=False,
            resume=not args.no_resume,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "checkpoint-smoke":
        print(json.dumps(_run_checkpoint_stress(args.output_root), indent=2))
    elif args.command == "aggregate":
        output = aggregate_workers(args.output_root)
        print(output)
    else:
        print(json.dumps(contract_manifest(), indent=2))


if __name__ == "__main__":
    main()
