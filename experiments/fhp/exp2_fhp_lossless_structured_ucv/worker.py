"""Train one seed of FHP Experiment 2."""

from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Mapping, Sequence

import numpy as np

from fhp_escher.checkpointing import (
    LoadedFHPPolicy,
    load_checkpoint_payload,
    save_policy_checkpoint,
    sha256_file,
)
from fhp_escher.game import load_fhp_game, serialisable_game_definition
from unbiased_escher.fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
from vr_deep_cfr.logger import Logger

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


LOGGER = logging.getLogger(__name__)
REMOTE_SYNC_ATTEMPTS = 5
REMOTE_SYNC_MAX_DELAY_SECONDS = 30.0


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, allow_nan=True)
        handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _repository_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _peak_rss_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def _config_sha256(config: Mapping) -> str:
    encoded = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _solver_kwargs(seed: int, config: Mapping) -> dict:
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
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=True),
    )
    return kwargs


def _make_solver(seed: int, config: Mapping):
    solver = StructuredFHPGroupedWideUCVEscher(**_solver_kwargs(seed, config))
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )
    solver.target_nodes_touched = None
    solver.max_wall_clock_seconds = None
    return solver


def _state_paths(worker_dir: Path, schedule: Sequence[Mapping], seed: int):
    return [
        worker_dir
        / "training_states"
        / f"{ALGORITHM_ID}_seed_{seed}_{row['checkpoint_id']}.pt"
        for row in reversed(schedule)
    ]


def _sync_remote(worker_dir: Path) -> None:
    remote = os.environ.get("EXP2_REMOTE_TASK_URI")
    if not remote:
        return
    if (
        not remote.startswith("gs://")
        or remote != remote.strip()
        or any(ord(character) < 32 for character in remote)
    ):
        raise ValueError(f"Invalid Experiment 2 remote task URI: {remote!r}")
    command = [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        str(worker_dir.resolve()),
        remote.rstrip("/"),
    ]
    last_result = None
    for attempt in range(1, REMOTE_SYNC_ATTEMPTS + 1):
        last_result = subprocess.run(command, check=False)
        if last_result.returncode == 0:
            return
        if attempt < REMOTE_SYNC_ATTEMPTS:
            delay = min(2.0 ** (attempt - 1), REMOTE_SYNC_MAX_DELAY_SECONDS)
            LOGGER.warning(
                "Checkpoint upload attempt %s/%s failed with exit code %s; "
                "retrying in %.1f seconds",
                attempt,
                REMOTE_SYNC_ATTEMPTS,
                last_result.returncode,
                delay,
            )
            time.sleep(delay)
    assert last_result is not None
    raise subprocess.CalledProcessError(last_result.returncode, command)


def _restore_latest(
    solver,
    *,
    worker_dir: Path,
    schedule: Sequence[Mapping],
    seed: int,
    commit: str,
    config: Mapping,
) -> list[dict]:
    manifest_rows = []
    manifest_path = worker_dir / "checkpoint_manifest.json"
    if manifest_path.is_file():
        manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in _state_paths(worker_dir, schedule, seed):
        if not path.is_file():
            continue
        recorded = next(
            (
                row
                for row in manifest_rows
                if row.get("training_state_path") == str(path.relative_to(worker_dir))
            ),
            None,
        )
        if recorded is not None and sha256_file(path) != recorded.get(
            "training_state_sha256"
        ):
            LOGGER.warning("Skipping corrupt continuation state: %s", path)
            continue
        try:
            payload = read_training_state(path)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
            LOGGER.exception("Skipping unreadable continuation state: %s", path)
            continue
        captured = restore_training_state(
            solver,
            payload,
            seed=seed,
            repository_commit=commit,
            config=config,
        )
        for record in captured:
            checkpoint = worker_dir / record["path"]
            if not checkpoint.is_file() or sha256_file(checkpoint) != record["sha256"]:
                raise ValueError(f"Restored policy checkpoint is missing or corrupt: {checkpoint}")
            state_path = worker_dir / "training_states" / (
                f"{ALGORITHM_ID}_seed_{seed}_{record['checkpoint_id']}.pt"
            )
            if state_path.is_file():
                record["training_state_path"] = str(state_path.relative_to(worker_dir))
                record["training_state_sha256"] = sha256_file(state_path)
                record["training_state_size_bytes"] = int(state_path.stat().st_size)
        LOGGER.info(
            "Resuming seed %s from %s at iteration %s and %.2f training hours",
            seed,
            path.name,
            solver.num_iteration,
            solver._resume_training_elapsed_seconds / 3600.0,
        )
        return [dict(row) for row in captured]
    return []


def _validate_loaded_policy(checkpoint_path: Path) -> None:
    game = load_fhp_game()
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    loaded = LoadedFHPPolicy(game, checkpoint_path)
    probabilities = loaded.action_probabilities(state)
    if not probabilities or not np.isclose(sum(probabilities.values()), 1.0):
        raise RuntimeError("Reloaded structured policy is not a valid distribution")


def _write_incremental_manifests(worker_dir: Path, records: Sequence[Mapping]) -> None:
    rows = [dict(row) for row in records]
    _write_json(worker_dir / "checkpoint_manifest.json", rows)
    _write_csv(worker_dir / "checkpoint_manifest.csv", rows)


def run_worker(
    *,
    seed: int,
    schedule: Sequence[Mapping],
    worker_dir: Path,
    config: Mapping,
    smoke: bool,
    resume: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    contract_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    validate_contract(
        seeds=contract_seeds, schedule=schedule, config=config, smoke=smoke
    )
    if int(seed) not in contract_seeds:
        raise ValueError(f"Seed {seed} is not part of the frozen contract")
    commit = _repository_commit()
    started = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "execution_backend": "sequential_seed_worker",
        "seed": int(seed),
        "smoke": bool(smoke),
        "checkpoint_schedule": [dict(row) for row in schedule],
        "checkpoint_training_seconds": [
            float(row["target_training_seconds"]) for row in schedule
        ],
        "training_duration_seconds": float(schedule[-1]["target_training_seconds"]),
        "checkpoint_boundary": "completed_outer_iteration",
        "stopping_rule": "final_training_time_checkpoint",
        "game": serialisable_game_definition(),
        "feature_encoder": StructuredFHPGroupedWideUCVEscher.__name__,
        "training_config": dict(config),
        "training_config_sha256": _config_sha256(config),
        "reference_vm": dict(REFERENCE_VM),
        "repository_commit": commit,
        "started_utc": started.isoformat(),
    }
    manifest_path = worker_dir / "run_manifest.json"
    if manifest_path.is_file() and resume:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("seed", "training_config_sha256", "repository_commit"):
            if prior.get(key) != manifest.get(key):
                raise ValueError(f"Existing run manifest differs at {key}")
        manifest["started_utc"] = prior["started_utc"]
    _write_json(manifest_path, manifest)

    solver = _make_solver(seed, config)
    manifest["feature_encoder"] = solver.feature_encoder.metadata()
    _write_json(manifest_path, manifest)
    records = (
        _restore_latest(
            solver,
            worker_dir=worker_dir,
            schedule=schedule,
            seed=seed,
            commit=commit,
            config=config,
        )
        if resume
        else []
    )
    captured = {str(row["checkpoint_id"]): dict(row) for row in records}
    solver.training_time_checkpoint_seconds = tuple(
        float(row["target_training_seconds"]) for row in schedule
    )
    solver.stop_after_final_training_time_checkpoint = True

    def capture_checkpoint(active_solver, raw_checkpoint):
        raw_checkpoint = dict(raw_checkpoint)
        target_seconds = float(raw_checkpoint["checkpoint_target_seconds"])
        target = next(
            row
            for row in schedule
            if np.isclose(float(row["target_training_seconds"]), target_seconds)
        )
        checkpoint_id = str(target["checkpoint_id"])
        if checkpoint_id in captured:
            return
        checkpoints_dir = worker_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / (
            f"{ALGORITHM_ID}_seed_{seed}_{checkpoint_id}.pkl"
        )
        raw_checkpoint.update(
            experiment_id=EXPERIMENT_ID,
            experiment_name=EXPERIMENT_NAME,
            algorithm_id=ALGORITHM_ID,
            execution_backend="sequential_seed_worker",
        )
        save_policy_checkpoint(
            active_solver,
            checkpoint_path,
            seed=seed,
            config=config,
            checkpoint_row=raw_checkpoint,
        )
        payload = load_checkpoint_payload(checkpoint_path)
        if payload.get("feature_encoder") != active_solver.feature_encoder.metadata():
            raise RuntimeError("Saved policy omitted the Experiment 2 encoder contract")
        record = {
            "checkpoint_index": len(captured),
            "checkpoint_id": checkpoint_id,
            "checkpoint_target_seconds": target_seconds,
            "checkpoint_target_hours": float(target["target_training_hours"]),
            "actual_training_elapsed_seconds": float(
                raw_checkpoint["training_elapsed_seconds"]
            ),
            "outer_iteration": int(active_solver.num_iteration),
            "episode": int(active_solver.episode),
            "nodes_touched": int(active_solver.nodes_touched),
            "path": str(checkpoint_path.relative_to(worker_dir)),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": int(checkpoint_path.stat().st_size),
        }
        captured[checkpoint_id] = record
        ordered = [
            captured[str(row["checkpoint_id"])]
            for row in schedule
            if str(row["checkpoint_id"]) in captured
        ]
        _write_incremental_manifests(worker_dir, ordered)
        state_path = worker_dir / "training_states" / (
            f"{ALGORITHM_ID}_seed_{seed}_{checkpoint_id}.pt"
        )
        record["training_state_path"] = str(state_path.relative_to(worker_dir))
        state_payload = build_training_state(
            active_solver,
            seed=seed,
            checkpoint_id=checkpoint_id,
            repository_commit=commit,
            config=config,
            captured_checkpoints=ordered,
        )
        save_training_state(state_path, state_payload)
        del state_payload
        record["training_state_sha256"] = sha256_file(state_path)
        record["training_state_size_bytes"] = int(state_path.stat().st_size)
        _write_incremental_manifests(worker_dir, ordered)
        _sync_remote(worker_dir)

    try:
        checkpoint_rows = (
            list(solver.checkpoint_rows)
            if len(captured) == len(schedule)
            else solver.solve(post_checkpoint_callback=capture_checkpoint)
        )
        if len(captured) == len(schedule):
            solver.stop_reason = "training_time_budget"
        ordered = [captured[str(row["checkpoint_id"])] for row in schedule]
        if len(ordered) != 4 or solver.stop_reason != "training_time_budget":
            raise RuntimeError("Experiment 2 did not complete all four checkpoints")
        _write_csv(worker_dir / "checkpoint_rows.csv", checkpoint_rows)
        _write_incremental_manifests(worker_dir, ordered)
        final_checkpoint = worker_dir / ordered[-1]["path"]
        final_policy = worker_dir / "final_policy_checkpoint.pkl"
        shutil.copy2(final_checkpoint, final_policy)
        _validate_loaded_policy(final_policy)
        finished = datetime.now(timezone.utc)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": ALGORITHM_ID,
            "seed": int(seed),
            "smoke": bool(smoke),
            "stop_reason": solver.stop_reason,
            "checkpoint_count": len(ordered),
            "checkpoint_hours": [
                float(row["checkpoint_target_hours"]) for row in ordered
            ],
            "final_iteration": int(solver.num_iteration),
            "final_episode": int(solver.episode),
            "final_nodes_touched": int(solver.nodes_touched),
            "final_training_elapsed_seconds": float(
                ordered[-1]["actual_training_elapsed_seconds"]
            ),
            "feature_encoder": solver.feature_encoder.metadata(),
            "peak_rss_mib": _peak_rss_mib(),
            "reference_vm": dict(REFERENCE_VM),
            "final_policy_path": final_policy.name,
            "final_policy_sha256": sha256_file(final_policy),
            "resumed_from_training_state": bool(
                solver._resume_training_elapsed_seconds > 0.0
            ),
            "started_utc": manifest["started_utc"],
            "finished_utc": finished.isoformat(),
        }
        _write_json(worker_dir / "summary.json", summary)
        _write_json(
            worker_dir / "SUCCESS.json",
            {
                "status": "complete",
                "seed": int(seed),
                "checkpoints": ordered,
                "summary_sha256": sha256_file(worker_dir / "summary.json"),
            },
        )
        _sync_remote(worker_dir)
        return summary
    except BaseException as exc:
        _write_json(
            worker_dir / "failure.json",
            {
                "status": "failed",
                "seed": int(seed),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "phase": "training_or_checkpoint",
                "outer_iteration": int(solver.num_iteration),
                "episode": int(solver.episode),
                "nodes_touched": int(solver.nodes_touched),
                "training_elapsed_seconds": float(solver._training_elapsed_seconds()),
                "peak_rss_mib": _peak_rss_mib(),
            },
        )
        try:
            _sync_remote(worker_dir)
        except BaseException:
            LOGGER.exception("Failure artifact upload also failed")
        raise


__all__ = ["_make_solver", "run_worker"]
