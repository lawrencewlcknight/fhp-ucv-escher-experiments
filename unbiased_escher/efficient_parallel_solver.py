"""CPU-optimised synchronous parallel UCV-ESCHER for large poker games."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from vr_deep_cfr.solver import AvePolicyTrainer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .efficient_replay import (
    CompactCalibrationBuffer,
    CompactCircularBuffer,
    CompactReservoirBuffer,
)
from .parallel_solver import (
    ParallelUnbiasedControlVariateEscher,
    UCVEscherTraversalWorker,
)
from .solver import (
    CrossFittedQEnsemble,
    CrossFittedQMember,
    GatedPredictiveRegretTrainer,
    ResidualCalibrationTrainer,
    UnbiasedControlVariateEscher,
)


class _CompactReservoirMixin:
    def __init__(self, *args, replay_seed: int, **kwargs):
        self._replay_seed = int(replay_seed)
        super().__init__(*args, **kwargs)

    def init_buffer(self):
        return CompactReservoirBuffer(
            self.buffer_size,
            self.input_size,
            self.output_size,
            device=self.device,
            seed=self._replay_seed,
        )


class EfficientAvePolicyTrainer(_CompactReservoirMixin, AvePolicyTrainer):
    pass


class EfficientGatedPredictiveRegretTrainer(
    _CompactReservoirMixin,
    GatedPredictiveRegretTrainer,
):
    def get_policy(self, state, T) -> np.ndarray:
        infostate = self.get_infostate_tensor(state)
        legal_mask = state.legal_actions_mask()
        scores = self.predictive_scores(
            self.forward(self.model, infostate, legal_mask),
            self.forward(self.imm_model, infostate, legal_mask),
            int(T),
        )
        return self.regret_matching(scores, state.legal_actions())


class EfficientDCFRPlusRegretTrainer(
    _CompactReservoirMixin,
    VRDCFRPlusRegretTrainer,
):
    pass


class EfficientCrossFittedQMember(CrossFittedQMember):
    def __init__(self, *args, replay_seed: int, **kwargs):
        self._replay_seed = int(replay_seed)
        super().__init__(*args, **kwargs)

    def init_buffer(self):
        return CompactCircularBuffer(
            self.buffer_size,
            self.input_size,
            self.state_size,
            self.output_size,
            device=self.device,
            seed=self._replay_seed,
        )


class EfficientCrossFittedQEnsemble(CrossFittedQEnsemble):
    def __init__(
        self,
        *,
        ensemble_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers,
        learning_rate: float,
        total_buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger,
        regret_trainers,
        device: str,
        gradient_clip_norm: float,
        replay_seeds,
    ):
        if ensemble_size < 1:
            raise ValueError("At least one critic is required")
        seeds = list(replay_seeds)
        if len(seeds) != int(ensemble_size):
            raise ValueError("One replay seed is required per Q fold")
        self.ensemble_size = int(ensemble_size)
        member_buffer_size = max(1, int(total_buffer_size) // self.ensemble_size)
        self.members = [
            EfficientCrossFittedQMember(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                member_buffer_size,
                batch_size,
                train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
                replay_seed=seeds[index],
            )
            for index in range(self.ensemble_size)
        ]
        self.active_fold = 0
        self._active_add_count = 0

    def predictions(self, state, player: int) -> np.ndarray:
        history_tensor = self.members[0].get_history_tensor(state)
        legal_mask = np.asarray(state.legal_actions_mask(), dtype=float)
        coefficient = 1.0 if int(player) == 0 else -1.0
        predictions = []
        for index in self.heldout_member_indices():
            member = self.members[index]
            x = torch.as_tensor(
                history_tensor,
                dtype=torch.float32,
                device=member.device,
            )
            with torch.no_grad():
                prediction = member.target_model(x).cpu().numpy() * coefficient
            predictions.append(prediction * legal_mask)
        return np.stack(predictions, axis=0)


class EfficientResidualCalibrationTrainer(ResidualCalibrationTrainer):
    def __init__(self, *args, replay_seed: int, **kwargs):
        self._replay_seed = int(replay_seed)
        super().__init__(*args, **kwargs)

    def init_buffer(self):
        return CompactCalibrationBuffer(
            self.buffer_size,
            self.feature_size,
            seed=self._replay_seed,
        )


class _EfficientReplaySolverMixin:
    """Construct compact replay and cache repeated OpenSpiel tensors.

    The estimator, network architecture, optimiser steps, traversal budget,
    player-update ordering, and frozen-target semantics are inherited unchanged.
    """

    def __init__(self, *args, seed: int = 0, **kwargs):
        self._efficient_seed = int(seed)
        self._efficient_seed_index = 0
        super().__init__(*args, seed=seed, **kwargs)

    def _next_replay_seed(self) -> int:
        self._efficient_seed_index += 1
        return self._efficient_seed + 10_007 * self._efficient_seed_index

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = EfficientAvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.device,
            self.gamma,
            replay_seed=self._next_replay_seed(),
        )

    def init_regret_trainers(self):
        trainer_class = (
            EfficientGatedPredictiveRegretTrainer
            if self.use_instantaneous_predictor
            else EfficientDCFRPlusRegretTrainer
        )
        self.regret_trainers = []
        for _ in range(self.num_players):
            common = (
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
            )
            if self.use_instantaneous_predictor:
                trainer = trainer_class(
                    *common,
                    self.reinitialize_imm_regret_networks,
                    self.use_regret_matching_argmax,
                    self.device,
                    self.alpha,
                    replay_seed=self._next_replay_seed(),
                )
            else:
                trainer = trainer_class(
                    *common,
                    self.use_regret_matching_argmax,
                    self.device,
                    self.alpha,
                    replay_seed=self._next_replay_seed(),
                )
                trainer.predictor_enabled = False
            self.regret_trainers.append(trainer)

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        seeds = [self._next_replay_seed() for _ in range(self.q_ensemble_size)]
        self.q_value_trainer = EfficientCrossFittedQEnsemble(
            ensemble_size=self.q_ensemble_size,
            history_size=history_size,
            state_size=self.infostate_size,
            action_size=self.action_size,
            network_layers=self.network_layers,
            learning_rate=self.learning_rate,
            total_buffer_size=self.baseline_buffer_size,
            batch_size=self.baseline_batch_size,
            train_steps=self.baseline_network_train_steps,
            logger=self.logger,
            regret_trainers=self.regret_trainers,
            device=self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
            replay_seeds=seeds,
        )

    def init_calibration_trainer(self):
        self.calibration_trainer = EfficientResidualCalibrationTrainer(
            infostate_size=self.infostate_size,
            action_size=self.action_size,
            hidden_layers=self.network_layers,
            learning_rate=self.calibration_learning_rate,
            buffer_size=self.calibration_buffer_size,
            batch_size=self.calibration_batch_size,
            train_steps=self.calibration_train_steps,
            device=self.device,
            minimum_variance=self.calibration_minimum_variance,
            replay_seed=self._next_replay_seed(),
        )

    def _central_replay_nbytes(self) -> int:
        return sum(int(buffer.nbytes()) for buffer in self._efficient_replay_buffers())

    def _efficient_replay_buffers(self):
        buffers = [
            self.ave_policy_trainer.buffer,
            *(trainer.buffer for trainer in self.regret_trainers),
            *(member.buffer for member in self.q_value_trainer.members),
        ]
        if self.calibration_trainer is not None:
            buffers.append(self.calibration_trainer.buffer)
        unique = {id(buffer): buffer for buffer in buffers}
        return list(unique.values())

    def _capture_rng_state(self):
        state = super()._capture_rng_state()
        state["efficient_replay"] = [
            deepcopy(buffer.rng.bit_generator.state)
            for buffer in self._efficient_replay_buffers()
        ]
        return state

    def _restore_rng_state(self, state):
        super()._restore_rng_state(state)
        replay_states = state.get("efficient_replay", ())
        buffers = self._efficient_replay_buffers()
        if len(replay_states) != len(buffers):
            raise RuntimeError("Efficient replay RNG state does not match buffer layout")
        for buffer, replay_state in zip(buffers, replay_states):
            buffer.rng.bit_generator.state = deepcopy(replay_state)


class EfficientWorkerSolver(
    _EfficientReplaySolverMixin,
    UnbiasedControlVariateEscher,
):
    """Non-Ray efficient solver used inside one traversal actor."""


class EfficientUCVEscherTraversalWorker(UCVEscherTraversalWorker):
    SOLVER_CLASS = EfficientWorkerSolver


class EfficientParallelUnbiasedControlVariateEscher(
    _EfficientReplaySolverMixin,
    ParallelUnbiasedControlVariateEscher,
):
    """Synchronous UCV-ESCHER with compact replay and high CPU occupancy."""

    @property
    def execution_backend(self) -> str:
        return "ray_parallel_cpu_optimized"

    def _traversal_worker_class(self):
        return EfficientUCVEscherTraversalWorker

    def evaluate(self, **kwargs):
        self.logger.record("efficient_replay_enabled", 1)
        self.logger.record("efficient_replay_sampling", "without_replacement")
        self.logger.record(
            "central_replay_allocated_gib",
            self._central_replay_nbytes() / float(1024**3),
        )
        return super().evaluate(**kwargs)
