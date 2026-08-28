"""Run FHP Experiment 1 and save reloadable policy checkpoints."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import resource
import shutil
import sys
import traceback
from typing import Mapping

import numpy as np
import psutil

from fhp_escher.checkpointing import (
    LoadedFHPPolicy,
    load_checkpoint_payload,
    save_policy_checkpoint,
    sha256_file,
)
from fhp_escher.game import load_fhp_game, serialisable_game_definition
from unbiased_escher import UnbiasedControlVariateEscher
from vr_deep_cfr.logger import Logger

from .config import (
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    BEST_UCV_CONFIG,
    BEST_UCV_TRAINING_CONFIG_SHA256,
    CHECKPOINT_TRAINING_SECONDS,
    DEFAULT_SEED,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    REFERENCE_VM,
    smoke_config,
    validate_config,
)


LOGGER = logging.getLogger(__name__)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, allow_nan=True)


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _peak_rss_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _solver_kwargs(seed: int, config: Mapping[str, object]) -> dict:
    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {
        key: value
        for key, value in deepcopy(dict(config)).items()
        if key not in control_fields
    }
    kwargs.update(
        {
            "num_episodes": (
                2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
            ),
            "seed": int(seed),
            "logger": Logger(verbose=True),
        }
    )
    return kwargs


def _training_config_sha256(config: Mapping[str, object]) -> str:
    payload = {key: value for key, value in config.items() if key != "game_name"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _capacity_assessment(summary: Mapping[str, object]) -> str:
    memory_fraction = float(summary["peak_rss_mib"]) / float(
        summary["reference_vm"]["memory_mib"]
    )
    if memory_fraction >= 0.9:
        return "memory_headroom_is_insufficient"
    if (
        summary["stop_reason"] == "training_time_budget"
        and int(summary["checkpoint_count"]) == 2
    ):
        final_target_seconds = float(summary["checkpoint_training_seconds"][-1])
        if np.isclose(final_target_seconds, 12 * 60 * 60):
            return "reference_vm_completed_the_12_hour_training_schedule"
        return "checkpoint_schedule_completed"
    return "inconclusive_check_failure_or_iteration_cap"


def run_experiment(
    *,
    seed: int,
    config: Mapping[str, object],
    checkpoint_training_seconds,
    output_root: Path,
    experiment_id: int = EXPERIMENT_ID,
    experiment_name: str = EXPERIMENT_NAME,
    algorithm_id: str = ALGORITHM_ID,
    algorithm_label: str = ALGORITHM_LABEL,
    reference_vm: Mapping[str, object] = REFERENCE_VM,
    solver_class=UnbiasedControlVariateEscher,
    solver_extra_kwargs: Mapping[str, object] | None = None,
    checkpoint_prefix: str = "exp1_fhp_ucv_escher",
    execution_backend: str = "sequential",
    implementation_provenance: Mapping[str, object] | None = None,
) -> Path:
    validate_config(config)
    checkpoint_training_seconds = tuple(
        float(value) for value in checkpoint_training_seconds
    )
    if len(checkpoint_training_seconds) != 2:
        raise ValueError("Time-bound FHP experiments require exactly two checkpoints")
    if any(
        left >= right
        for left, right in zip(
            checkpoint_training_seconds, checkpoint_training_seconds[1:]
        )
    ):
        raise ValueError("Time checkpoints must be strictly increasing")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{experiment_name}_{timestamp}_seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "experiment_id": int(experiment_id),
        "experiment_name": str(experiment_name),
        "algorithm_id": str(algorithm_id),
        "algorithm_label": str(algorithm_label),
        "execution_backend": str(execution_backend),
        "seed": int(seed),
        "checkpoint_training_seconds": list(checkpoint_training_seconds),
        "training_duration_seconds": checkpoint_training_seconds[-1],
        "stopping_rule": "final_training_time_checkpoint",
        "reference_vm": dict(reference_vm),
        "game": serialisable_game_definition(),
        "training_config": dict(config),
        "training_config_sha256": _training_config_sha256(config),
        "selected_full_config_sha256": BEST_UCV_TRAINING_CONFIG_SHA256,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if solver_extra_kwargs:
        manifest["execution_config"] = dict(solver_extra_kwargs)
    if implementation_provenance:
        manifest["implementation_provenance"] = dict(implementation_provenance)
    _write_json(run_dir / "run_manifest.json", manifest)

    active_solver_kwargs = _solver_kwargs(seed, config)
    active_solver_kwargs.update(dict(solver_extra_kwargs or {}))
    try:
        solver = solver_class(**active_solver_kwargs)
    except BaseException as exc:
        _write_json(
            run_dir / "failure.json",
            {
                "phase": "solver_initialization",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "peak_rss_mib": _peak_rss_mib(),
            },
        )
        raise
    solver.target_nodes_touched = None
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.max_wall_clock_seconds = None
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )
    solver.training_time_checkpoint_seconds = checkpoint_training_seconds
    solver.stop_after_final_training_time_checkpoint = True

    checkpoint_rows = []

    def capture_checkpoint(active_solver, raw_checkpoint):
        raw_checkpoint = dict(raw_checkpoint)
        raw_checkpoint.update(
            {
                "experiment_id": int(experiment_id),
                "experiment_name": str(experiment_name),
                "algorithm_id": str(algorithm_id),
                "execution_backend": str(execution_backend),
            }
        )
        index = len(checkpoint_rows)
        kind = str(raw_checkpoint.get("checkpoint_kind", "checkpoint"))
        filename = (
            f"{checkpoint_prefix}_{index:03d}_iter_{active_solver.num_iteration:04d}_"
            f"{kind}.pkl"
        )
        checkpoint_path = save_policy_checkpoint(
            active_solver,
            checkpoints_dir / filename,
            seed=seed,
            config=config,
            checkpoint_row=raw_checkpoint,
        )
        payload = load_checkpoint_payload(checkpoint_path)
        if int(payload["nodes_touched"]) != int(active_solver.nodes_touched):
            raise RuntimeError("Saved checkpoint node count does not match solver state")
        checkpoint_rows.append(
            {
                "checkpoint_index": index,
                "experiment_id": int(experiment_id),
                "experiment_name": str(experiment_name),
                "execution_backend": str(execution_backend),
                "checkpoint_kind": kind,
                "outer_iteration": int(active_solver.num_iteration),
                "episode": int(active_solver.episode),
                "nodes_touched": int(active_solver.nodes_touched),
                "wall_clock_seconds": float(
                    raw_checkpoint.get("wall_clock_seconds", float("nan"))
                ),
                "training_elapsed_seconds": float(
                    raw_checkpoint.get("training_elapsed_seconds", float("nan"))
                ),
                "checkpoint_target_seconds": float(
                    raw_checkpoint.get("checkpoint_target_seconds", float("nan"))
                ),
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            }
        )
        _write_json(run_dir / "checkpoint_manifest.json", checkpoint_rows)

    try:
        raw_rows = solver.solve(post_checkpoint_callback=capture_checkpoint)
        actual_targets = tuple(
            row["checkpoint_target_seconds"] for row in checkpoint_rows
        )
        if actual_targets != checkpoint_training_seconds:
            raise RuntimeError(
                f"Captured time checkpoints {actual_targets}, expected "
                f"{checkpoint_training_seconds}"
            )
        if solver.stop_reason != "training_time_budget":
            raise RuntimeError(
                f"Training stopped for {solver.stop_reason!r}, not after the final "
                "training-time checkpoint"
            )

        final_source = Path(checkpoint_rows[-1]["path"])
        final_checkpoint = run_dir / "final_policy_checkpoint.pkl"
        shutil.copyfile(final_source, final_checkpoint)
        final_payload = load_checkpoint_payload(final_checkpoint)
        game = load_fhp_game()
        loaded_policy = LoadedFHPPolicy(game, final_checkpoint)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        probabilities = loaded_policy.action_probabilities(state)
        if not np.isclose(sum(probabilities.values()), 1.0):
            raise RuntimeError("Reloaded final policy probabilities do not sum to one")

        _write_json(run_dir / "checkpoint_rows.json", raw_rows)
        _write_csv(run_dir / "checkpoint_rows.csv", raw_rows)
        _write_csv(run_dir / "checkpoint_manifest.csv", checkpoint_rows)
        final_wall_clock = float(raw_rows[-1].get("wall_clock_seconds", 0.0))
        final_training_elapsed = float(
            raw_rows[-1].get("training_elapsed_seconds", 0.0)
        )
        summary = {
            "experiment_id": int(experiment_id),
            "experiment_name": str(experiment_name),
            "algorithm_id": str(algorithm_id),
            "algorithm_label": str(algorithm_label),
            "execution_backend": str(execution_backend),
            "seed": int(seed),
            "stop_reason": str(solver.stop_reason),
            "checkpoint_training_seconds": list(checkpoint_training_seconds),
            "final_nodes_touched": int(solver.nodes_touched),
            "final_outer_iteration": int(solver.num_iteration),
            "final_episode": int(solver.episode),
            "wall_clock_seconds": final_wall_clock,
            "training_elapsed_seconds": final_training_elapsed,
            "nodes_per_second": (
                float(solver.nodes_touched) / final_training_elapsed
                if final_training_elapsed > 0.0
                else float("nan")
            ),
            "peak_rss_mib": _peak_rss_mib(),
            "current_rss_mib": (
                psutil.Process().memory_info().rss / (1024.0 * 1024.0)
            ),
            "reference_vm": dict(reference_vm),
            "checkpoint_count": len(checkpoint_rows),
            "final_policy_checkpoint": str(final_checkpoint.resolve()),
            "final_policy_checkpoint_sha256": sha256_file(final_checkpoint),
            "final_checkpoint_nodes": int(final_payload["nodes_touched"]),
        }
        summary["execution_metrics"] = {
            key: value
            for key, value in raw_rows[-1].items()
            if key.startswith("parallel_") or key.startswith("cumulative_parallel_")
        }
        summary["capacity_assessment"] = _capacity_assessment(summary)
        _write_json(run_dir / "summary.json", summary)
        return run_dir
    except BaseException as exc:
        _write_json(
            run_dir / "failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "nodes_touched": int(getattr(solver, "nodes_touched", 0)),
                "outer_iteration": int(getattr(solver, "num_iteration", 0)),
                "peak_rss_mib": _peak_rss_mib(),
            },
        )
        raise
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()


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
    config = smoke_config() if args.smoke else deepcopy(BEST_UCV_CONFIG)
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
