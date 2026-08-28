"""Synchronous Ray-parallel traversal collection for FHP UCV-ESCHER.

Ported from the Leduc ESCHER architecture repository at commit
c09bbe5a9adc6b495c9a0f74c2993dcb682ea754. The
learner, optimisers and persistent replay remain in one authoritative driver.
Workers receive one frozen inference snapshot per traverser and return only new
trajectory samples, so there are no stale asynchronous gradient updates.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from fhp_escher.game import FHP_GAME_NAME, load_fhp_game
from vr_deep_cfr.logger import Logger

from .parallel_utils import partition_total, worker_seed
from .solver import UnbiasedControlVariateEscher


DEFAULT_RAY_OBJECT_STORE_MEMORY = 4 * 1024 * 1024 * 1024
DEFAULT_COLLECTION_CHUNK_SIZE = 1_200


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _reservoir_payload(buffer) -> dict[str, np.ndarray]:
    size = min(int(buffer.cur_id), int(buffer.buffer_size))
    if int(buffer.cur_id) > int(buffer.buffer_size):
        raise RuntimeError(
            "Traversal worker capacity was exceeded; increase the derived "
            "worker collection capacity"
        )
    return {
        "infostates": np.asarray(buffer.infostate_buf[:size]).copy(),
        "values": np.asarray(buffer.q_value_buf[:size]).copy(),
        "legal_masks": np.asarray(buffer.q_value_mask_buf[:size]).copy(),
        "iterations": np.asarray(buffer.iteration_buf[:size]).copy(),
    }


def _circular_payload(buffer) -> dict[str, np.ndarray]:
    size = int(buffer.size)
    return {
        "histories": np.asarray(buffer.history_buf[:size]).copy(),
        "actions": np.asarray(buffer.action_buf[:size]).copy(),
        "next_histories": np.asarray(buffer.next_history_buf[:size]).copy(),
        "next_states": np.asarray(buffer.next_state_buf[:size]).copy(),
        "next_legal_masks": np.asarray(
            buffer.next_legal_actions_mask_buf[:size]
        ).copy(),
        "next_players": np.asarray(buffer.next_player_buf[:size]).copy(),
        "dones": np.asarray(buffer.done_buf[:size]).copy(),
        "rewards": np.asarray(buffer.reward_buf[:size]).copy(),
    }


def _append_reservoir(buffer, payload: Mapping[str, np.ndarray]) -> None:
    for sample in zip(
        payload["infostates"],
        payload["values"],
        payload["legal_masks"],
        payload["iterations"],
    ):
        buffer.add(sample[0], sample[1], sample[2], sample[3])


def _append_circular(buffer, payload: Mapping[str, np.ndarray]) -> None:
    """Append a transition batch with one vectorised circular-buffer write."""
    original_count = int(len(payload["actions"]))
    if original_count == 0:
        return
    capacity = int(buffer.buffer_size)
    count = min(original_count, capacity)
    start = original_count - count
    write_start = (int(buffer.cur_id) + start) % capacity
    indices = (write_start + np.arange(count)) % capacity
    source = slice(start, start + count)
    buffer.history_buf[indices] = payload["histories"][source]
    buffer.action_buf[indices] = payload["actions"][source]
    buffer.next_history_buf[indices] = payload["next_histories"][source]
    buffer.next_state_buf[indices] = payload["next_states"][source]
    buffer.next_legal_actions_mask_buf[indices] = payload["next_legal_masks"][source]
    buffer.next_player_buf[indices] = payload["next_players"][source]
    buffer.done_buf[indices] = payload["dones"][source]
    buffer.reward_buf[indices] = payload["rewards"][source]
    buffer.cur_id = (int(buffer.cur_id) + original_count) % capacity
    buffer.size = min(int(buffer.size) + original_count, capacity)


def _append_calibration(buffer, payload: Mapping[str, np.ndarray]) -> None:
    original_count = int(len(payload["targets"]))
    if original_count == 0:
        return
    capacity = int(buffer.capacity)
    count = min(original_count, capacity)
    start = original_count - count
    write_start = (int(buffer.cursor) + start) % capacity
    indices = (write_start + np.arange(count)) % capacity
    source = slice(start, start + count)
    buffer.features[indices] = payload["features"][source]
    buffer.targets[indices] = payload["targets"][source]
    buffer.cursor = (int(buffer.cursor) + original_count) % capacity
    buffer.size = min(int(buffer.size) + original_count, capacity)


class UCVEscherTraversalWorker:
    """CPU-only Ray actor with inference networks and bounded scratch replay."""

    def __init__(
        self,
        game_name: str,
        solver_kwargs: dict[str, Any],
        worker_seed_value: int,
        max_traversals_per_request: int,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        if str(game_name) != FHP_GAME_NAME:
            raise ValueError(f"Parallel FHP workers do not support {game_name!r}")
        game = load_fhp_game()
        max_traversals = max(1, int(max_traversals_per_request))
        max_records = max_traversals * (int(game.max_game_length()) + 1)
        ensemble_size = int(solver_kwargs.get("q_ensemble_size", 3))
        max_fold_trajectories = math.ceil(max_traversals / ensemble_size)
        q_capacity = (
            ensemble_size
            * max_fold_trajectories
            * (int(game.max_game_length()) + 1)
        )

        kwargs = dict(solver_kwargs)
        kwargs.update(
            {
                "device": "cpu",
                "seed": int(worker_seed_value),
                "logger": Logger(verbose=False),
                "num_episodes": 2 * max_traversals,
                "advantage_buffer_size": max_records,
                "ave_policy_buffer_size": max_records,
                "baseline_buffer_size": q_capacity,
                "calibration_buffer_size": max_records,
                "advantage_network_train_steps": 0,
                "ave_policy_network_train_steps": 0,
                "baseline_network_train_steps": 0,
                "calibration_train_steps": 0,
            }
        )
        self._max_traversals = max_traversals
        self._solver = UnbiasedControlVariateEscher(**kwargs)
        self._solver.regret_trainers[1].buffer = (
            self._solver.regret_trainers[0].buffer
        )
        self._compact_collection_storage()
        self._set_inference_mode()

    def ping(self) -> bool:
        return True

    def _compact_collection_storage(self) -> None:
        """Use transfer-efficient dtypes for worker-only scratch replay."""
        reservoir_buffers = {
            id(self._solver.regret_trainers[0].buffer): (
                self._solver.regret_trainers[0].buffer
            ),
            id(self._solver.ave_policy_trainer.buffer): (
                self._solver.ave_policy_trainer.buffer
            ),
        }
        for buffer in reservoir_buffers.values():
            buffer.infostate_buf = buffer.infostate_buf.astype(np.float32)
            buffer.q_value_buf = buffer.q_value_buf.astype(np.float32)
            buffer.q_value_mask_buf = buffer.q_value_mask_buf.astype(np.float32)
            buffer.iteration_buf = buffer.iteration_buf.astype(np.float32)
        for member in self._solver.q_value_trainer.members:
            buffer = member.buffer
            buffer.history_buf = buffer.history_buf.astype(np.float32)
            buffer.next_history_buf = buffer.next_history_buf.astype(np.float32)
            buffer.next_state_buf = buffer.next_state_buf.astype(np.float32)
            buffer.reward_buf = buffer.reward_buf.astype(np.float32)
            buffer.action_buf = buffer.action_buf.astype(np.int16)
            buffer.next_legal_actions_mask_buf = (
                buffer.next_legal_actions_mask_buf.astype(np.int8)
            )
            buffer.next_player_buf = buffer.next_player_buf.astype(np.int8)
            buffer.done_buf = buffer.done_buf.astype(np.int8)

    def _set_inference_mode(self) -> None:
        for trainer in self._solver.regret_trainers:
            trainer.model.eval()
            trainer.target_model.eval()
            if hasattr(trainer, "imm_model"):
                trainer.imm_model.eval()
        for member in self._solver.q_value_trainer.members:
            member.model.eval()
            member.target_model.eval()
        calibration = self._solver.calibration_trainer
        if calibration is not None:
            calibration.model.eval()
            calibration.target_model.eval()

    def _clear_collection_buffers(self) -> None:
        for trainer in self._solver.regret_trainers:
            trainer.reset_buffer()
        self._solver.ave_policy_trainer.reset_buffer()
        for member in self._solver.q_value_trainer.members:
            member.buffer.cur_id = 0
            member.buffer.size = 0
        calibration = self._solver.calibration_trainer
        if calibration is not None:
            calibration.buffer.cursor = 0
            calibration.buffer.size = 0
        self._solver._reset_architecture_diagnostics()

    def _load_snapshot(self, snapshot: Mapping[str, Any], iteration: int) -> None:
        for trainer, state in zip(
            self._solver.regret_trainers,
            snapshot["regret_trainers"],
        ):
            trainer.model.load_state_dict(state["model"])
            if "imm_model" in state:
                trainer.imm_model.load_state_dict(state["imm_model"])
            if hasattr(trainer, "set_prediction_gate"):
                trainer.set_prediction_gate(float(state.get("prediction_gate", 0.0)))
        for member, state in zip(
            self._solver.q_value_trainer.members,
            snapshot["q_target_models"],
        ):
            member.target_model.load_state_dict(state)
        calibration = self._solver.calibration_trainer
        if calibration is not None and snapshot.get("calibration_target_model") is not None:
            calibration.target_model.load_state_dict(snapshot["calibration_target_model"])
        self._solver.num_iteration = int(iteration)
        self._set_inference_mode()

    def collect(
        self,
        n: int,
        traverser: int,
        trajectory_start: int,
        snapshot: Mapping[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        n = int(n)
        if n < 0 or n > self._max_traversals:
            raise ValueError(
                f"Requested {n} traversals; worker capacity is {self._max_traversals}"
            )
        self._clear_collection_buffers()
        self._load_snapshot(snapshot, int(iteration))
        before_nodes = int(self._solver.nodes_touched)
        started_at = time.perf_counter()
        with torch.inference_mode():
            for offset in range(n):
                trajectory_id = int(trajectory_start) + offset
                self._solver.episode = trajectory_id
                self._solver.q_value_trainer.begin_trajectory(trajectory_id)
                root = self._solver.skip_chance_state(
                    self._solver.game.new_initial_state()
                )
                self._solver.dfs(root, int(traverser))
        elapsed = time.perf_counter() - started_at

        calibration = self._solver.calibration_trainer
        if calibration is None:
            calibration_payload = None
        else:
            size = int(calibration.buffer.size)
            calibration_payload = {
                "features": np.asarray(calibration.buffer.features[:size]).copy(),
                "targets": np.asarray(calibration.buffer.targets[:size]).copy(),
            }
        return {
            "nodes_touched": int(self._solver.nodes_touched - before_nodes),
            "num_trajectories": n,
            "worker_collection_seconds": float(elapsed),
            "regret": _reservoir_payload(
                self._solver.regret_trainers[int(traverser)].buffer
            ),
            "average_policy": _reservoir_payload(
                self._solver.ave_policy_trainer.buffer
            ),
            "q_folds": [
                _circular_payload(member.buffer)
                for member in self._solver.q_value_trainer.members
            ],
            "calibration": calibration_payload,
            "architecture_stats": dict(self._solver._architecture_stats),
            "minimum_sample_probability": float(
                self._solver._minimum_sample_probability
            ),
        }


class ParallelUnbiasedControlVariateEscher(UnbiasedControlVariateEscher):
    """UCV-ESCHER with synchronous Ray-parallel traversal collection."""

    def __init__(
        self,
        *args,
        parallel_num_workers: int = 3,
        parallel_run_seed: int = 0,
        parallel_ray_address: str | None = None,
        parallel_log_to_driver: bool = False,
        parallel_ray_object_store_memory: int | None = None,
        parallelize_independent_learners: bool = True,
        parallel_learner_threads: int | None = None,
        parallel_collection_chunk_size: int = DEFAULT_COLLECTION_CHUNK_SIZE,
        **solver_kwargs,
    ):
        self._parallel_num_workers = int(parallel_num_workers)
        if self._parallel_num_workers < 2:
            raise ValueError("parallel_num_workers must be at least 2")
        self._parallel_run_seed = int(parallel_run_seed)
        self._parallel_ray_address = parallel_ray_address
        self._parallel_log_to_driver = bool(parallel_log_to_driver)
        self._parallel_ray_object_store_memory = (
            DEFAULT_RAY_OBJECT_STORE_MEMORY
            if parallel_ray_object_store_memory is None
            else int(parallel_ray_object_store_memory)
        )
        if self._parallel_ray_object_store_memory <= 0:
            raise ValueError("parallel_ray_object_store_memory must be positive")
        self._parallelize_independent_learners = bool(
            parallelize_independent_learners
        )
        self._parallel_learner_threads = (
            None
            if parallel_learner_threads is None
            else int(parallel_learner_threads)
        )
        if self._parallel_learner_threads is not None and self._parallel_learner_threads < 1:
            raise ValueError("parallel_learner_threads must be positive")
        self._parallel_collection_chunk_size = int(parallel_collection_chunk_size)
        if self._parallel_collection_chunk_size <= 0:
            raise ValueError("parallel_collection_chunk_size must be positive")

        worker_solver_kwargs = dict(solver_kwargs)
        if args:
            raise TypeError(
                "Parallel UCV-ESCHER requires keyword arguments so workers can "
                "reconstruct the frozen traversal configuration"
            )
        game_name = str(worker_solver_kwargs.get("game_name", ""))
        if game_name != FHP_GAME_NAME:
            raise ValueError("Parallel UCV-ESCHER requires the canonical FHP game")
        super().__init__(**solver_kwargs)
        self._workers: list[Any] = []
        self._ray = None
        self._owns_ray_runtime = False
        self._cumulative_parallel_collection_seconds = 0.0
        self._cumulative_worker_collection_seconds = 0.0
        self._cumulative_parallel_sync_seconds = 0.0
        self._cumulative_parallel_merge_seconds = 0.0
        self._cumulative_parallel_learner_seconds = 0.0
        self._parallel_peak_result_bytes = 0
        self._parallel_dispatch_count = 0
        self._effective_parallel_learner_threads = min(
            len(self.q_value_trainer.members)
            + int(self.calibration_trainer is not None),
            self._parallel_learner_threads
            or len(self.q_value_trainer.members)
            + int(self.calibration_trainer is not None),
        )

        worker_kwargs = dict(worker_solver_kwargs)
        worker_kwargs.pop("logger", None)
        maximum_dispatch = min(
            int(self.num_traversals),
            self._parallel_collection_chunk_size,
        )
        max_worker_traversals = max(
            partition_total(maximum_dispatch, self._parallel_num_workers)
        )
        try:
            import ray

            self._ray = ray
            self._owns_ray_runtime = not ray.is_initialized()
            if self._owns_ray_runtime:
                init_kwargs = {
                    "include_dashboard": False,
                    "log_to_driver": self._parallel_log_to_driver,
                    "ignore_reinit_error": True,
                }
                if self._parallel_ray_address:
                    init_kwargs["address"] = str(self._parallel_ray_address)
                else:
                    init_kwargs["num_cpus"] = self._parallel_num_workers
                    init_kwargs["object_store_memory"] = (
                        self._parallel_ray_object_store_memory
                    )
                ray.init(**init_kwargs)

            worker_class = ray.remote(num_cpus=1)(UCVEscherTraversalWorker)
            self._workers = [
                worker_class.remote(
                    game_name,
                    worker_kwargs,
                    worker_seed(self._parallel_run_seed, worker_index),
                    max_worker_traversals,
                )
                for worker_index in range(self._parallel_num_workers)
            ]
            ray.get([worker.ping.remote() for worker in self._workers])
        except Exception:
            self.close()
            raise

    @property
    def execution_backend(self) -> str:
        return "ray_parallel"

    @property
    def parallel_num_workers(self) -> int:
        return int(self._parallel_num_workers)

    def _inference_snapshot(self) -> dict[str, Any]:
        regret_states = []
        for trainer in self.regret_trainers:
            state: dict[str, Any] = {
                "model": _cpu_state_dict(trainer.model),
                "prediction_gate": float(getattr(trainer, "prediction_gate", 0.0)),
            }
            if hasattr(trainer, "imm_model"):
                state["imm_model"] = _cpu_state_dict(trainer.imm_model)
            regret_states.append(state)
        calibration = self.calibration_trainer
        return {
            "regret_trainers": regret_states,
            "q_target_models": [
                _cpu_state_dict(member.target_model)
                for member in self.q_value_trainer.members
            ],
            "calibration_target_model": (
                None
                if calibration is None
                else _cpu_state_dict(calibration.target_model)
            ),
        }

    @staticmethod
    def _payload_nbytes(value: Any) -> int:
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, Mapping):
            return sum(
                ParallelUnbiasedControlVariateEscher._payload_nbytes(item)
                for item in value.values()
            )
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return sum(
                ParallelUnbiasedControlVariateEscher._payload_nbytes(item)
                for item in value
            )
        return 0

    def _merge_worker_result(self, traverser: int, result: Mapping[str, Any]) -> None:
        _append_reservoir(
            self.regret_trainers[int(traverser)].buffer,
            result["regret"],
        )
        _append_reservoir(self.ave_policy_trainer.buffer, result["average_policy"])
        for member, payload in zip(
            self.q_value_trainer.members,
            result["q_folds"],
        ):
            _append_circular(member.buffer, payload)
        if self.calibration_trainer is not None and result["calibration"] is not None:
            _append_calibration(
                self.calibration_trainer.buffer,
                result["calibration"],
            )

        worker_stats = result["architecture_stats"]
        central_stats = self._architecture_stats
        for key in central_stats:
            if key == "beta_min":
                central_stats[key] = min(central_stats[key], worker_stats[key])
            elif key == "beta_max":
                central_stats[key] = max(central_stats[key], worker_stats[key])
            else:
                central_stats[key] += worker_stats[key]
        self._minimum_sample_probability = min(
            self._minimum_sample_probability,
            float(result["minimum_sample_probability"]),
        )

    def _collect_chunk(
        self,
        *,
        player: int,
        trajectory_start: int,
        count: int,
        snapshot_ref,
    ) -> int:
        counts = partition_total(count, self._parallel_num_workers)
        starts = []
        next_start = int(trajectory_start)
        for worker_count in counts:
            starts.append(next_start)
            next_start += int(worker_count)
        refs = [
            worker.collect.remote(
                int(worker_count),
                int(player),
                int(start),
                snapshot_ref,
                int(self.num_iteration),
            )
            for worker, worker_count, start in zip(self._workers, counts, starts)
            if int(worker_count) > 0
        ]
        results = self._ray.get(refs) if refs else []
        self._parallel_dispatch_count += 1
        self._cumulative_worker_collection_seconds += sum(
            float(result["worker_collection_seconds"])
            for result in results
        )

        merge_start = time.perf_counter()
        for result in results:
            self._parallel_peak_result_bytes = max(
                self._parallel_peak_result_bytes,
                self._payload_nbytes(result),
            )
            self._merge_worker_result(int(player), result)
        self._cumulative_parallel_merge_seconds += time.perf_counter() - merge_start
        self.nodes_touched += sum(int(result["nodes_touched"]) for result in results)
        collected = sum(int(result["num_trajectories"]) for result in results)
        if collected != int(count):
            raise RuntimeError(
                f"Parallel workers returned {collected} of {count} traversals"
            )
        self.episode += collected
        return collected

    def collect_training_data(self, player):
        self.regret_trainers[int(player)].reset_buffer()
        collection_start = time.perf_counter()
        sync_start = time.perf_counter()
        snapshot_ref = self._ray.put(self._inference_snapshot())
        self._cumulative_parallel_sync_seconds += time.perf_counter() - sync_start
        remaining = int(self.num_traversals)
        try:
            while remaining > 0:
                chunk_count = min(remaining, self._parallel_collection_chunk_size)
                dispatch_start = time.perf_counter()
                collected = self._collect_chunk(
                    player=int(player),
                    trajectory_start=int(self.episode) + 1,
                    count=chunk_count,
                    snapshot_ref=snapshot_ref,
                )
                self._cumulative_parallel_collection_seconds += (
                    time.perf_counter() - dispatch_start
                )
                remaining -= collected
                self._maybe_run_early_node_checkpoint()
                self._maybe_run_training_time_checkpoint()
                if self._stop_requested:
                    break
        finally:
            del snapshot_ref
            self._cumulative_experience_collection_seconds = getattr(
                self, "_cumulative_experience_collection_seconds", 0.0
            ) + (time.perf_counter() - collection_start)

    def _train_independent_control_learners(self):
        """Train disjoint Q folds and calibration concurrently on the driver."""
        calibration = self.calibration_trainer
        task_count = len(self.q_value_trainer.members) + (calibration is not None)
        if not self._parallelize_independent_learners or task_count <= 1:
            calibration_loss = (
                calibration.train_model() if calibration is not None else None
            )
            q_loss = self.q_value_trainer.train_model(self.num_iteration)
            return calibration_loss, q_loss

        max_workers = min(
            task_count,
            self._parallel_learner_threads or task_count,
        )
        self._effective_parallel_learner_threads = max_workers
        original_torch_threads = int(torch.get_num_threads())
        per_task_threads = max(1, original_torch_threads // max_workers)
        torch.set_num_threads(per_task_threads)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                calibration_future = (
                    executor.submit(calibration.train_model)
                    if calibration is not None
                    else None
                )
                q_futures = [
                    executor.submit(member.train_model, self.num_iteration)
                    for member in self.q_value_trainer.members
                ]
                calibration_loss = (
                    calibration_future.result()
                    if calibration_future is not None
                    else None
                )
                q_losses = [future.result() for future in q_futures]
        finally:
            torch.set_num_threads(original_torch_threads)
        finite = [float(loss) for loss in q_losses if loss is not None]
        q_loss = float(np.mean(finite)) if finite else None
        return calibration_loss, q_loss

    def iteration(self):
        """Preserve UCV update ordering while parallelising safe subphases."""
        self._reset_architecture_diagnostics()
        self.num_iteration += 1
        for player in range(self.num_players):
            trainer = self.regret_trainers[player]
            if getattr(trainer, "predictor_enabled", False):
                trainer.set_prediction_gate(
                    0.0
                    if self.force_prediction_gate_zero
                    else self.gate_controller.value(player)
                )
        holdout_errors = []
        for player in range(self.num_players):
            self.collect_training_data(player)
            if self._stop_requested:
                return
            holdout_errors.append(self._predictor_holdout_error(player))
            self.train_regret(player)
        for player, (prediction_mse, zero_mse) in enumerate(holdout_errors):
            self.gate_controller.observe(player, prediction_mse, zero_mse)
            if self.force_prediction_gate_zero or not self.use_instantaneous_predictor:
                self.gate_controller.gates[player] = 0.0
            self.logger.record(f"predictor_holdout_mse_player_{player}", prediction_mse)
            self.logger.record(f"predictor_zero_mse_player_{player}", zero_mse)

        learner_start = time.perf_counter()
        calibration_loss, q_loss = self._train_independent_control_learners()
        self._cumulative_parallel_learner_seconds += (
            time.perf_counter() - learner_start
        )
        self.logger.record("calibration_loss", calibration_loss)
        if q_loss is not None:
            self.logger.record("baseline_loss_0", q_loss)
            self.logger.record("baseline_loss_1", q_loss)
        if (
            self.evaluation_frequency > 0
            and self.num_iteration % self.evaluation_frequency == 0
        ):
            self._run_checkpoint(checkpoint_kind="outer_iteration")

    def evaluate(self, **kwargs):
        self.logger.record("execution_backend", self.execution_backend)
        self.logger.record("parallel_num_workers", self.parallel_num_workers)
        self.logger.record(
            "parallel_collection_chunk_size",
            self._parallel_collection_chunk_size,
        )
        self.logger.record("parallel_dispatch_count", self._parallel_dispatch_count)
        self.logger.record(
            "cumulative_parallel_collection_seconds",
            self._cumulative_parallel_collection_seconds,
        )
        self.logger.record(
            "cumulative_worker_collection_seconds",
            self._cumulative_worker_collection_seconds,
        )
        self.logger.record(
            "cumulative_parallel_sync_seconds",
            self._cumulative_parallel_sync_seconds,
        )
        self.logger.record(
            "cumulative_parallel_merge_seconds",
            self._cumulative_parallel_merge_seconds,
        )
        self.logger.record(
            "cumulative_parallel_learner_seconds",
            self._cumulative_parallel_learner_seconds,
        )
        self.logger.record(
            "parallel_independent_learner_threads",
            self._effective_parallel_learner_threads,
        )
        self.logger.record(
            "parallel_peak_worker_result_mib",
            self._parallel_peak_result_bytes / (1024.0 * 1024.0),
        )
        return super().evaluate(**kwargs)

    def close(self) -> None:
        ray = getattr(self, "_ray", None)
        if ray is None:
            return
        for worker in getattr(self, "_workers", []):
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self._workers = []
        if self._owns_ray_runtime and ray.is_initialized():
            ray.shutdown()
        self._ray = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
