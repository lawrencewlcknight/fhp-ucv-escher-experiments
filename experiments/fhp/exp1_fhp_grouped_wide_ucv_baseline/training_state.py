"""Exact, memory-conscious continuation states for FHP Experiment 1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1
STATE_TYPE = "exp1_fhp_grouped_wide_full_training_state"


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def read_training_state(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


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
    "read_training_state",
    "restore_training_state",
    "save_training_state",
]
