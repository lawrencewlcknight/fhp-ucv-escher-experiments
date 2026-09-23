"""Exact, memory-conscious continuation states for FHP Experiment 1."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1
STATE_TYPE = "exp1_fhp_grouped_wide_full_training_state"
SHARDED_SCHEMA_VERSION = 1
SHARDED_STATE_TYPE = "exp1_fhp_sharded_training_state"
SHARDED_MANIFEST = "manifest.json"
SHARDED_METADATA = "metadata.pt"
_ARRAY_MARKER = "__exp1_numpy_shard__"


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _reservoir_state(buffer) -> dict[str, Any]:
    size = min(int(buffer.cur_id), int(buffer.buffer_size))
    return {
        "cur_id": int(buffer.cur_id),
        "size": size,
        # Views avoid doubling FHP's large replay allocation before torch.save.
        "infostate": np.asarray(buffer.infostate_buf[:size]),
        "q_value": np.asarray(buffer.q_value_buf[:size]),
        "q_value_mask": np.asarray(buffer.q_value_mask_buf[:size]),
        "iteration": np.asarray(buffer.iteration_buf[:size]),
    }


def _load_reservoir_state(buffer, state: Mapping[str, Any]) -> None:
    size = int(state["size"])
    if size > int(buffer.buffer_size):
        raise ValueError("Saved reservoir exceeds configured capacity")
    buffer.infostate_buf[:size] = state["infostate"]
    buffer.q_value_buf[:size] = state["q_value"]
    buffer.q_value_mask_buf[:size] = state["q_value_mask"]
    buffer.iteration_buf[:size] = state["iteration"]
    buffer.cur_id = int(state["cur_id"])


def _circular_state(buffer) -> dict[str, Any]:
    occupied = (
        int(buffer.buffer_size)
        if int(buffer.size) == int(buffer.buffer_size)
        else int(buffer.size)
    )
    return {
        "cur_id": int(buffer.cur_id),
        "size": int(buffer.size),
        "history": np.asarray(buffer.history_buf[:occupied]),
        "action": np.asarray(buffer.action_buf[:occupied]),
        "next_history": np.asarray(buffer.next_history_buf[:occupied]),
        "next_state": np.asarray(buffer.next_state_buf[:occupied]),
        "next_legal_actions_mask": np.asarray(
            buffer.next_legal_actions_mask_buf[:occupied]
        ),
        "next_player": np.asarray(buffer.next_player_buf[:occupied]),
        "done": np.asarray(buffer.done_buf[:occupied]),
        "reward": np.asarray(buffer.reward_buf[:occupied]),
    }


def _load_circular_state(buffer, state: Mapping[str, Any]) -> None:
    occupied = len(state["action"])
    if occupied > int(buffer.buffer_size) or int(state["size"]) > int(buffer.buffer_size):
        raise ValueError("Saved circular replay exceeds configured capacity")
    buffer.history_buf[:occupied] = state["history"]
    buffer.action_buf[:occupied] = state["action"]
    buffer.next_history_buf[:occupied] = state["next_history"]
    buffer.next_state_buf[:occupied] = state["next_state"]
    buffer.next_legal_actions_mask_buf[:occupied] = state["next_legal_actions_mask"]
    buffer.next_player_buf[:occupied] = state["next_player"]
    buffer.done_buf[:occupied] = state["done"]
    buffer.reward_buf[:occupied] = state["reward"]
    buffer.cur_id = int(state["cur_id"])
    buffer.size = int(state["size"])


def _calibration_state(buffer) -> dict[str, Any]:
    occupied = (
        int(buffer.capacity)
        if int(buffer.size) == int(buffer.capacity)
        else int(buffer.size)
    )
    return {
        "cursor": int(buffer.cursor),
        "size": int(buffer.size),
        "features": np.asarray(buffer.features[:occupied]),
        "targets": np.asarray(buffer.targets[:occupied]),
    }


def _load_calibration_state(buffer, state: Mapping[str, Any]) -> None:
    occupied = len(state["targets"])
    if occupied > int(buffer.capacity) or int(state["size"]) > int(buffer.capacity):
        raise ValueError("Saved calibration replay exceeds configured capacity")
    buffer.features[:occupied] = state["features"]
    buffer.targets[:occupied] = state["targets"]
    buffer.cursor = int(state["cursor"])
    buffer.size = int(state["size"])


def _trainer_state(trainer, *, include_buffer: bool) -> dict[str, Any]:
    state = {
        "model": _cpu_state_dict(trainer.model),
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
    }
    if hasattr(trainer, "target_model"):
        state["target_model"] = _cpu_state_dict(trainer.target_model)
    if include_buffer:
        state["buffer"] = _reservoir_state(trainer.buffer)
    return state


def _load_trainer_state(trainer, state: Mapping[str, Any]) -> None:
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    if "target_model" in state:
        trainer.target_model.load_state_dict(state["target_model"])
    if "buffer" in state:
        _load_reservoir_state(trainer.buffer, state["buffer"])


def build_training_state(
    solver,
    *,
    seed: int,
    checkpoint_id: str,
    repository_commit: str,
    config: Mapping[str, Any],
    captured_checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rng_state = solver._checkpoint_resume_rng_state
    if rng_state is None:
        rng_state = solver._capture_rng_state()
    calibration = solver.calibration_trainer
    return {
        "schema_version": SCHEMA_VERSION,
        "type": STATE_TYPE,
        "seed": int(seed),
        "checkpoint_id": str(checkpoint_id),
        "repository_commit": str(repository_commit),
        "config": dict(config),
        "captured_checkpoints": [dict(row) for row in captured_checkpoints],
        "solver": {
            "num_iteration": int(solver.num_iteration),
            "episode": int(solver.episode),
            "nodes_touched": int(solver.nodes_touched),
            "checkpoint_rows": list(solver.checkpoint_rows),
            "training_elapsed_seconds": float(solver._training_elapsed_seconds()),
            "architecture_stats": dict(solver._architecture_stats),
            "minimum_sample_probability": float(solver._minimum_sample_probability),
            "last_average_policy_loss": float(solver._last_average_policy_loss),
        },
        "rng": deepcopy(rng_state),
        "regret_trainers": [
            _trainer_state(trainer, include_buffer=False)
            for trainer in solver.regret_trainers
        ],
        "average_policy_trainer": _trainer_state(
            solver.ave_policy_trainer, include_buffer=True
        ),
        "q_ensemble": {
            "active_fold": int(solver.q_value_trainer.active_fold),
            "active_add_count": int(solver.q_value_trainer._active_add_count),
            "members": [
                {
                    "model": _cpu_state_dict(member.model),
                    "target_model": _cpu_state_dict(member.target_model),
                    "optimizer": deepcopy(member.optimizer.state_dict()),
                    "buffer": _circular_state(member.buffer),
                    "target_version": int(member.target_version),
                    "target_history": [
                        {
                            name: value.detach().cpu().clone()
                            for name, value in snapshot.items()
                        }
                        for snapshot in member.target_history
                    ],
                    "last_target_update_l2": float(member.last_target_update_l2),
                }
                for member in solver.q_value_trainer.members
            ],
        },
        "calibration": (
            None
            if calibration is None
            else {
                "model": _cpu_state_dict(calibration.model),
                "target_model": _cpu_state_dict(calibration.target_model),
                "optimizer": deepcopy(calibration.optimizer.state_dict()),
                "buffer": _calibration_state(calibration.buffer),
                "target_version": int(calibration.target_version),
            }
        ),
        "gate_controller": {
            "gates": np.asarray(solver.gate_controller.gates),
            "prediction_mse": np.asarray(solver.gate_controller.prediction_mse),
            "zero_mse": np.asarray(solver.gate_controller.zero_mse),
            "relative_skill": np.asarray(solver.gate_controller.relative_skill),
        },
        "logger": {
            "pending": dict(solver.logger._pending),
            "history": list(solver.logger.history),
        },
    }


def save_training_state(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the legacy monolithic format used by FHP Experiments 2 and 3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def read_training_state(path: Path) -> dict[str, Any]:
    """Read the legacy monolithic format used by FHP Experiments 2 and 3."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _externalise_numpy_arrays(
    value: Any,
    *,
    state_root: Path,
    shards: list[dict[str, Any]],
) -> Any:
    """Replace NumPy arrays with independently streamed ``.npy`` shards."""
    if isinstance(value, np.ndarray):
        relative = Path("arrays") / f"array_{len(shards):04d}.npy"
        destination = state_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Passing the real file handle lets NumPy stream contiguous replay
        # views directly instead of constructing one multi-gigabyte pickle.
        with open(destination, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        record = {
            "path": relative.as_posix(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "size_bytes": int(destination.stat().st_size),
            "sha256": _sha256_file(destination),
        }
        shards.append(record)
        return {_ARRAY_MARKER: relative.as_posix()}
    if isinstance(value, dict):
        return {
            key: _externalise_numpy_arrays(
                item, state_root=state_root, shards=shards
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _externalise_numpy_arrays(item, state_root=state_root, shards=shards)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _externalise_numpy_arrays(item, state_root=state_root, shards=shards)
            for item in value
        )
    return value


def _restore_numpy_arrays(value: Any, *, state_root: Path) -> Any:
    if isinstance(value, dict) and set(value) == {_ARRAY_MARKER}:
        shard = state_root / value[_ARRAY_MARKER]
        return np.load(shard, mmap_mode="r", allow_pickle=False)
    if isinstance(value, dict):
        return {
            key: _restore_numpy_arrays(item, state_root=state_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_numpy_arrays(item, state_root=state_root) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _restore_numpy_arrays(item, state_root=state_root) for item in value
        )
    return value


def save_sharded_training_state(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically save a continuation state without a large pickle buffer.

    Replay arrays are written one at a time as ``.npy`` files. The remaining
    small PyTorch/Python metadata is saved separately, and ``manifest.json`` is
    written last so incomplete directories are never considered resumable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    )
    try:
        shards: list[dict[str, Any]] = []
        metadata = _externalise_numpy_arrays(
            dict(payload), state_root=temporary, shards=shards
        )
        metadata_path = temporary / SHARDED_METADATA
        metadata_temporary = metadata_path.with_suffix(".pt.tmp")
        torch.save(metadata, metadata_temporary)
        metadata_temporary.replace(metadata_path)
        manifest = {
            "schema_version": SHARDED_SCHEMA_VERSION,
            "type": SHARDED_STATE_TYPE,
            "payload_type": payload.get("type"),
            "metadata_path": SHARDED_METADATA,
            "metadata_size_bytes": int(metadata_path.stat().st_size),
            "metadata_sha256": _sha256_file(metadata_path),
            "array_shards": shards,
            "complete": True,
        }
        manifest["total_size_bytes"] = int(
            manifest["metadata_size_bytes"]
            + sum(int(record["size_bytes"]) for record in shards)
        )
        manifest_path = temporary / SHARDED_MANIFEST
        manifest_temporary = manifest_path.with_suffix(".json.tmp")
        with open(manifest_temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_temporary.replace(manifest_path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        temporary.replace(path)
        return training_state_storage_info(path, verify=True)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_sharded_manifest(path: Path, *, verify: bool) -> dict[str, Any]:
    path = Path(path)
    manifest_path = path / SHARDED_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("type") != SHARDED_STATE_TYPE
        or int(manifest.get("schema_version", -1)) != SHARDED_SCHEMA_VERSION
        or manifest.get("complete") is not True
    ):
        raise ValueError(f"Incomplete or unsupported sharded state: {path}")
    metadata_path = path / manifest["metadata_path"]
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    records = list(manifest.get("array_shards", ()))
    for record in records:
        shard = path / record["path"]
        if not shard.is_file():
            raise FileNotFoundError(shard)
    if verify:
        if (
            int(metadata_path.stat().st_size) != int(manifest["metadata_size_bytes"])
            or _sha256_file(metadata_path) != manifest["metadata_sha256"]
        ):
            raise ValueError(f"Sharded-state metadata hash mismatch: {metadata_path}")
        for record in records:
            shard = path / record["path"]
            if (
                int(shard.stat().st_size) != int(record["size_bytes"])
                or _sha256_file(shard) != record["sha256"]
            ):
                raise ValueError(f"Sharded-state array hash mismatch: {shard}")
            array = np.load(shard, mmap_mode="r", allow_pickle=False)
            if str(array.dtype) != record["dtype"] or list(array.shape) != list(
                record["shape"]
            ):
                raise ValueError(f"Sharded-state array metadata mismatch: {shard}")
    return manifest


def training_state_storage_info(path: Path, *, verify: bool = False) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        manifest = _read_sharded_manifest(path, verify=verify)
        manifest_path = path / SHARDED_MANIFEST
        return {
            "format": "sharded_numpy_v1",
            "sha256": _sha256_file(manifest_path),
            "size_bytes": int(
                manifest["total_size_bytes"] + manifest_path.stat().st_size
            ),
        }
    return {
        "format": "legacy_torch_v1",
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def read_sharded_training_state(path: Path, *, verify: bool = True) -> dict[str, Any]:
    path = Path(path)
    manifest = _read_sharded_manifest(path, verify=verify)
    metadata = torch.load(
        path / manifest["metadata_path"], map_location="cpu", weights_only=False
    )
    return _restore_numpy_arrays(metadata, state_root=path)


def restore_training_state(
    solver,
    payload: Mapping[str, Any],
    *,
    seed: int,
    repository_commit: str,
    config: Mapping[str, Any],
) -> list[dict]:
    if (
        payload.get("type") != STATE_TYPE
        or int(payload.get("schema_version", -1)) != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported full-training-state schema")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError("Training state belongs to a different seed")
    if payload.get("repository_commit") != repository_commit:
        raise ValueError("Training state was produced by a different commit")
    if payload.get("config") != dict(config):
        raise ValueError("Training-state configuration differs from the worker contract")

    core = payload["solver"]
    solver.num_iteration = int(core["num_iteration"])
    solver.episode = int(core["episode"])
    solver.nodes_touched = int(core["nodes_touched"])
    solver.checkpoint_rows = list(core["checkpoint_rows"])
    solver._resume_training_elapsed_seconds = float(core["training_elapsed_seconds"])
    solver._resume_completed_checkpoint_count = len(payload["captured_checkpoints"])
    solver._architecture_stats = dict(core["architecture_stats"])
    solver._minimum_sample_probability = float(core["minimum_sample_probability"])
    solver._last_average_policy_loss = float(core["last_average_policy_loss"])

    for trainer, state in zip(solver.regret_trainers, payload["regret_trainers"]):
        _load_trainer_state(trainer, state)
    _load_trainer_state(solver.ave_policy_trainer, payload["average_policy_trainer"])

    ensemble_state = payload["q_ensemble"]
    ensemble = solver.q_value_trainer
    if len(ensemble.members) != len(ensemble_state["members"]):
        raise ValueError("Critic ensemble size differs from saved state")
    ensemble.active_fold = int(ensemble_state["active_fold"])
    ensemble._active_add_count = int(ensemble_state["active_add_count"])
    for member, state in zip(ensemble.members, ensemble_state["members"]):
        member.model.load_state_dict(state["model"])
        member.target_model.load_state_dict(state["target_model"])
        member.optimizer.load_state_dict(state["optimizer"])
        _load_circular_state(member.buffer, state["buffer"])
        member.target_version = int(state["target_version"])
        member.target_history.clear()
        member.target_history.extend(state["target_history"])
        member.last_target_update_l2 = float(state["last_target_update_l2"])

    calibration_state = payload["calibration"]
    calibration = solver.calibration_trainer
    if (calibration_state is None) != (calibration is None):
        raise ValueError("Calibration presence differs from saved state")
    if calibration is not None:
        calibration.model.load_state_dict(calibration_state["model"])
        calibration.target_model.load_state_dict(calibration_state["target_model"])
        calibration.optimizer.load_state_dict(calibration_state["optimizer"])
        _load_calibration_state(calibration.buffer, calibration_state["buffer"])
        calibration.target_version = int(calibration_state["target_version"])

    gate = payload["gate_controller"]
    solver.gate_controller.gates = np.asarray(gate["gates"]).copy()
    solver.gate_controller.prediction_mse = np.asarray(gate["prediction_mse"]).copy()
    solver.gate_controller.zero_mse = np.asarray(gate["zero_mse"]).copy()
    solver.gate_controller.relative_skill = np.asarray(gate["relative_skill"]).copy()
    solver.logger._pending = dict(payload["logger"]["pending"])
    solver.logger.history = list(payload["logger"]["history"])
    solver._restore_rng_state(payload["rng"])
    return [dict(row) for row in payload["captured_checkpoints"]]


__all__ = [
    "SCHEMA_VERSION",
    "STATE_TYPE",
    "build_training_state",
    "read_sharded_training_state",
    "read_training_state",
    "restore_training_state",
    "save_sharded_training_state",
    "save_training_state",
    "training_state_storage_info",
]
