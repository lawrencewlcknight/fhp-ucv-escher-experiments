"""Cross-fitted, uncertainty-adaptive, always-unbiased ESCHER solver."""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_escher.solver import PersistentFrozenTargetQValueTrainer
from vr_deep_cfr.variants import (
    VRDCFRPlusRegretTrainer,
    VRDeepPDCFRPlus,
    VRPDCFRPlusRegretTrainer,
)

from .estimator import (
    control_variate_advantage,
    residual_adaptive_sampling_policy,
    variance_optimal_beta,
)


class GatedPredictiveRegretTrainer(VRPDCFRPlusRegretTrainer):
    """Interpolate conservative DCFR+ and predictive PDCFR+ mechanisms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_enabled = True
        self.prediction_gate = 0.0

    def set_prediction_gate(self, value: float) -> None:
        self.prediction_gate = float(np.clip(value, 0.0, 1.0))

    def compute_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, _ = samples
        with torch.no_grad():
            previous = torch.clamp(
                self.predict(self.target_model, infostates, legal_actions_mask),
                min=0.0,
            )
        age = math.pow(T - 1, self.alpha)
        conservative = previous * age / (age + 1.5) + cf_regrets
        predictive = previous * age / (age + 1.0) + cf_regrets
        target = (
            (1.0 - self.prediction_gate) * conservative
            + self.prediction_gate * predictive
        )
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        return self.loss_fn(outputs, target)

    def predictive_scores(self, cumulative, immediate, iteration: int):
        positive = np.maximum(cumulative, 0.0)
        age = np.power(max(iteration - 1, 0), self.alpha)
        optimistic = np.maximum(positive * age / (age + 1.0) + immediate, 0.0)
        return (1.0 - self.prediction_gate) * positive + (
            self.prediction_gate * optimistic
        )

    def get_policy(self, state, T) -> np.ndarray:
        scores = self.predictive_scores(
            self.get_regrets(state),
            self.get_imm_regrets(state),
            int(T),
        )
        return self.regret_matching(scores, state.legal_actions())


class PredictorGateController:
    """One-iteration-lagged gate based on held-out predictor skill."""

    def __init__(self, num_players: int, *, ema_decay: float, initial_gate: float):
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        self.ema_decay = float(ema_decay)
        self.gates = np.full(num_players, float(initial_gate), dtype=np.float64)
        self.prediction_mse = np.full(num_players, np.nan, dtype=np.float64)
        self.zero_mse = np.full(num_players, np.nan, dtype=np.float64)
        self.relative_skill = np.zeros(num_players, dtype=np.float64)

    def value(self, player: int) -> float:
        return float(self.gates[player])

    def observe(self, player: int, prediction_mse: float, zero_mse: float) -> None:
        if not np.isfinite(prediction_mse) or not np.isfinite(zero_mse):
            return
        if np.isnan(self.prediction_mse[player]):
            self.prediction_mse[player] = prediction_mse
            self.zero_mse[player] = zero_mse
        else:
            decay = self.ema_decay
            self.prediction_mse[player] = (
                decay * self.prediction_mse[player]
                + (1.0 - decay) * prediction_mse
            )
            self.zero_mse[player] = (
                decay * self.zero_mse[player] + (1.0 - decay) * zero_mse
            )
        if float(self.zero_mse[player]) <= 1e-12:
            self.relative_skill[player] = 0.0
            self.gates[player] = 0.0
            return
        denominator = max(float(self.zero_mse[player]), 1e-12)
        skill = 1.0 - float(self.prediction_mse[player]) / denominator
        self.relative_skill[player] = skill
        self.gates[player] = float(np.clip(skill, 0.0, 1.0))


class CrossFittedQMember(PersistentFrozenTargetQValueTrainer):
    """One persistent critic trained on one disjoint trajectory fold."""

    def _batched_predictive_strategies(
        self,
        next_states: torch.Tensor,
        next_legal_actions_mask: torch.Tensor,
        next_players: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        player_strategies = []
        for player in range(2):
            trainer = self.regret_trainers[player]
            cumulative = torch.clamp(
                trainer.predict(
                    trainer.model,
                    next_states,
                    next_legal_actions_mask,
                ),
                min=0.0,
            )
            if getattr(trainer, "predictor_enabled", False):
                immediate = trainer.predict(
                    trainer.imm_model,
                    next_states,
                    next_legal_actions_mask,
                )
                age = math.pow(max(iteration - 1, 0), trainer.alpha)
                optimistic = torch.clamp(
                    cumulative * age / (age + 1.0) + immediate,
                    min=0.0,
                )
                scores = (
                    (1.0 - trainer.prediction_gate) * cumulative
                    + trainer.prediction_gate * optimistic
                )
            else:
                scores = cumulative
            scores *= next_legal_actions_mask
            positive_sum = scores.sum(dim=1, keepdim=True)
            normalised = scores / torch.clamp(positive_sum, min=1e-12)
            masked = torch.where(
                next_legal_actions_mask == 1,
                scores,
                torch.full_like(scores, float("-inf")),
            )
            fallback = F.one_hot(
                torch.argmax(masked, dim=1),
                self.output_size,
            ).to(scores.dtype)
            player_strategies.append(
                torch.where(positive_sum > 0.0, normalised, fallback)
            )
        return torch.where(
            next_players.unsqueeze(1) == 0,
            player_strategies[0],
            player_strategies[1],
        )


class CrossFittedQEnsemble:
    """Disjoint-fold critics with leave-one-fold-out trajectory inference."""

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
    ):
        if ensemble_size < 1:
            raise ValueError("At least one critic is required")
        self.ensemble_size = int(ensemble_size)
        member_buffer_size = max(1, int(total_buffer_size) // self.ensemble_size)
        self.members = [
            CrossFittedQMember(
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
            )
            for _ in range(self.ensemble_size)
        ]
        self.active_fold = 0
        self._active_add_count = 0

    @property
    def target_version(self) -> int:
        return min(member.target_version for member in self.members)

    def begin_trajectory(self, trajectory_id: int) -> int:
        self.active_fold = int(trajectory_id) % self.ensemble_size
        self._active_add_count = 0
        return self.active_fold

    def heldout_member_indices(self) -> List[int]:
        if self.ensemble_size == 1:
            # This arm deliberately removes cross-fitting. The persistent target
            # remains frozen during each outer iteration, so inference is still
            # predictable even though it is no longer fold-independent.
            return [0]
        return [
            index
            for index in range(self.ensemble_size)
            if index != self.active_fold
        ]

    def predictions(self, state, player: int) -> np.ndarray:
        return np.stack(
            [
                self.members[index].get_baseline(state, player)
                for index in self.heldout_member_indices()
            ],
            axis=0,
        )

    def get_baseline_and_disagreement(self, state, player: int):
        predictions = self.predictions(state, player)
        mean = np.mean(predictions, axis=0)
        disagreement = np.var(predictions, axis=0)
        return mean, disagreement

    def get_baseline(self, state, player: int) -> np.ndarray:
        mean, _ = self.get_baseline_and_disagreement(state, player)
        return mean

    def add_data(self, *args) -> None:
        self.members[self.active_fold].add_data(*args)
        self._active_add_count += 1

    def train_model(self, iteration: int):
        losses = [member.train_model(iteration) for member in self.members]
        finite = [float(loss) for loss in losses if loss is not None]
        return float(np.mean(finite)) if finite else None

    def fold_sizes(self) -> List[int]:
        return [len(member.buffer) for member in self.members]


class CalibrationBuffer:
    """Fixed-size circular replay for cross-fitted residual calibration."""

    def __init__(self, capacity: int, feature_size: int):
        if capacity <= 0 or feature_size <= 0:
            raise ValueError("capacity and feature_size must be positive")
        self.capacity = int(capacity)
        self.features = np.zeros((capacity, feature_size), dtype=np.float32)
        self.targets = np.zeros(capacity, dtype=np.float32)
        self.cursor = 0
        self.size = 0

    def add(self, features, target: float) -> None:
        self.features[self.cursor] = np.asarray(features, dtype=np.float32)
        self.targets[self.cursor] = float(target)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, device: str):
        count = min(int(count), self.size)
        indices = random.sample(range(self.size), count)
        return (
            torch.as_tensor(
                self.features[indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.targets[indices],
                dtype=torch.float32,
                device=device,
            ),
        )

    def __len__(self):
        return self.size


class ResidualCalibrationTrainer:
    """Frozen-target heteroscedastic residual calibration network."""

    def __init__(
        self,
        *,
        infostate_size: int,
        action_size: int,
        hidden_layers: Iterable[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        device: str,
        minimum_variance: float,
    ):
        self.infostate_size = int(infostate_size)
        self.action_size = int(action_size)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.minimum_variance = float(minimum_variance)
        self.feature_size = self.infostate_size + self.action_size + 3
        sizes = [self.feature_size, *list(hidden_layers), 2]
        layers = []
        for input_size, output_size in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(input_size, output_size), nn.ReLU()])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.model = nn.Sequential(*layers).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
        )
        self.buffer = CalibrationBuffer(buffer_size, self.feature_size)
        self.target_version = 0

    def feature(
        self,
        infostate,
        action: int,
        iteration: int,
        disagreement: float,
        player: int,
    ) -> np.ndarray:
        action_one_hot = np.zeros(self.action_size, dtype=np.float32)
        action_one_hot[int(action)] = 1.0
        scalars = np.asarray(
            [
                math.log1p(max(int(iteration), 0)) / math.log(101.0),
                math.log1p(max(float(disagreement), 0.0)),
                float(player),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [np.asarray(infostate, dtype=np.float32), action_one_hot, scalars]
        )

    def predict_all(
        self,
        infostate,
        iteration: int,
        disagreement,
        player: int,
    ):
        features = np.stack(
            [
                self.feature(
                    infostate,
                    action,
                    iteration,
                    float(disagreement[action]),
                    player,
                )
                for action in range(self.action_size)
            ]
        )
        with torch.no_grad():
            outputs = self.target_model(
                torch.as_tensor(features, dtype=torch.float32, device=self.device)
            )
            means = outputs[:, 0]
            log_variances = torch.clamp(outputs[:, 1], min=-8.0, max=6.0)
            variances = torch.exp(log_variances) + self.minimum_variance
        return (
            means.cpu().numpy().astype(np.float64),
            variances.cpu().numpy().astype(np.float64),
            features,
        )

    def add(self, features, target_residual: float) -> None:
        self.buffer.add(features, target_residual)

    def train_model(self):
        if len(self.buffer) < max(1, self.batch_size) or self.train_steps <= 0:
            return None
        self.model.train()
        final_loss = None
        for _ in range(self.train_steps):
            features, targets = self.buffer.sample(self.batch_size, self.device)
            outputs = self.model(features)
            mean = outputs[:, 0]
            log_variance = torch.clamp(outputs[:, 1], min=-8.0, max=6.0)
            loss = 0.5 * torch.mean(
                log_variance + torch.square(targets - mean) * torch.exp(-log_variance)
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            final_loss = float(loss.item())
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_version += 1
        return final_loss


class UnbiasedControlVariateEscher(VRDeepPDCFRPlus):
    """Always-unbiased adaptive control-variate predictive neural CFR."""

    def __init__(
        self,
        *args,
        q_ensemble_size: int = 3,
        beta_min: float = 0.0,
        beta_max: float = 2.0,
        beta_ridge: float = 1e-4,
        sampling_uniform_floor_mass: float = 0.2,
        calibration_buffer_size: int = 1_000_000,
        calibration_batch_size: int = 2_048,
        calibration_train_steps: int = 2_000,
        calibration_learning_rate: float = 1e-3,
        calibration_minimum_variance: float = 1e-5,
        prediction_gate_ema_decay: float = 0.9,
        prediction_gate_initial: float = 0.0,
        q_gradient_clip_norm: float = 10.0,
        fixed_control_variate_beta: float | None = None,
        force_prediction_gate_zero: bool = False,
        use_instantaneous_predictor: bool = True,
        use_residual_calibration: bool = True,
        **kwargs,
    ):
        self.q_ensemble_size = int(q_ensemble_size)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_ridge = float(beta_ridge)
        self.sampling_uniform_floor_mass = float(sampling_uniform_floor_mass)
        self.calibration_buffer_size = int(calibration_buffer_size)
        self.calibration_batch_size = int(calibration_batch_size)
        self.calibration_train_steps = int(calibration_train_steps)
        self.calibration_learning_rate = float(calibration_learning_rate)
        self.calibration_minimum_variance = float(calibration_minimum_variance)
        self.prediction_gate_ema_decay = float(prediction_gate_ema_decay)
        self.prediction_gate_initial = float(prediction_gate_initial)
        self.q_gradient_clip_norm = float(q_gradient_clip_norm)
        self.fixed_control_variate_beta = (
            None
            if fixed_control_variate_beta is None
            else float(fixed_control_variate_beta)
        )
        self.force_prediction_gate_zero = bool(force_prediction_gate_zero)
        self.use_instantaneous_predictor = bool(use_instantaneous_predictor)
        self.use_residual_calibration = bool(use_residual_calibration)
        super().__init__(*args, **kwargs)
        if not self.use_baseline or not self.fit_advantage:
            raise ValueError("The unbiased architecture requires Q and advantages")
        if not self.use_residual_calibration:
            if self.fixed_control_variate_beta is None:
                raise ValueError(
                    "Disabling residual calibration requires a fixed control-variate beta"
                )
            if self.sampling_uniform_floor_mass != 1.0:
                raise ValueError(
                    "Disabling residual calibration requires uniform full-support sampling"
                )
            self.calibration_trainer = None
        else:
            self.calibration_trainer = ResidualCalibrationTrainer(
                infostate_size=self.infostate_size,
                action_size=self.action_size,
                hidden_layers=self.network_layers,
                learning_rate=self.calibration_learning_rate,
                buffer_size=self.calibration_buffer_size,
                batch_size=self.calibration_batch_size,
                train_steps=self.calibration_train_steps,
                device=self.device,
                minimum_variance=self.calibration_minimum_variance,
            )
        self.gate_controller = PredictorGateController(
            self.num_players,
            ema_decay=self.prediction_gate_ema_decay,
            initial_gate=self.prediction_gate_initial,
        )
        self._reset_architecture_diagnostics()

    def init_regret_trainers(self):
        if self.use_instantaneous_predictor:
            self.regret_trainers = [
                GatedPredictiveRegretTrainer(
                    self.infostate_size,
                    self.action_size,
                    self.network_layers,
                    self.learning_rate,
                    self.advantage_buffer_size,
                    self.advantage_batch_size,
                    self.advantage_network_train_steps,
                    self.logger,
                    self.reinitialize_imm_regret_networks,
                    self.use_regret_matching_argmax,
                    self.device,
                    self.alpha,
                )
                for _ in range(self.num_players)
            ]
        else:
            self.regret_trainers = [
                VRDCFRPlusRegretTrainer(
                    self.infostate_size,
                    self.action_size,
                    self.network_layers,
                    self.learning_rate,
                    self.advantage_buffer_size,
                    self.advantage_batch_size,
                    self.advantage_network_train_steps,
                    self.logger,
                    self.use_regret_matching_argmax,
                    self.device,
                    self.alpha,
                )
                for _ in range(self.num_players)
            ]
            for trainer in self.regret_trainers:
                trainer.predictor_enabled = False

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = CrossFittedQEnsemble(
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
        )

    def _reset_architecture_diagnostics(self) -> None:
        self._architecture_stats: Dict[str, float] = {
            "count": 0.0,
            "beta_sum": 0.0,
            "beta_min": float("inf"),
            "beta_max": float("-inf"),
            "variance_sum": 0.0,
            "disagreement_sum": 0.0,
            "q_residual_abs_sum": 0.0,
            "control_residual_abs_sum": 0.0,
            "correction_abs_sum": 0.0,
            "centering_abs_sum": 0.0,
        }
        self._minimum_sample_probability = 1.0

    def _predictor_holdout_error(self, player: int):
        trainer = self.regret_trainers[player]
        if not getattr(trainer, "predictor_enabled", False):
            return np.nan, np.nan
        if len(trainer.buffer) == 0:
            return np.nan, np.nan
        infostates, targets, legal_mask, _ = trainer.buffer.sample(-1)
        with torch.no_grad():
            predictions = trainer.predict(trainer.imm_model, infostates, legal_mask)
            denominator = torch.clamp(torch.sum(legal_mask), min=1.0)
            prediction_mse = torch.sum(
                torch.square(predictions - targets) * legal_mask
            ) / denominator
            zero_mse = torch.sum(torch.square(targets) * legal_mask) / denominator
        return float(prediction_mse.item()), float(zero_mse.item())

    def collect_training_data(self, player):
        self.regret_trainers[player].reset_buffer()
        for _ in range(self.num_traversals):
            self.episode += 1
            self.q_value_trainer.begin_trajectory(self.episode)
            root_state = self.skip_chance_state(self.game.new_initial_state())
            self.dfs(root_state, player)
            self._maybe_run_early_node_checkpoint()
            self._maybe_run_training_time_checkpoint()
            if self._stop_requested:
                break

    def iteration(self):
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
        calibration_loss = (
            self.calibration_trainer.train_model()
            if self.calibration_trainer is not None
            else None
        )
        self.logger.record("calibration_loss", calibration_loss)
        q_loss = self.q_value_trainer.train_model(self.num_iteration)
        if q_loss is not None:
            self.logger.record("baseline_loss_0", q_loss)
            self.logger.record("baseline_loss_1", q_loss)
        if (
            self.evaluation_frequency > 0
            and self.num_iteration % self.evaluation_frequency == 0
        ):
            self._run_checkpoint(checkpoint_kind="outer_iteration")

    def _record_estimate_diagnostics(
        self,
        *,
        beta: float,
        predicted_variance: float,
        disagreement: float,
        estimate,
    ) -> None:
        stats = self._architecture_stats
        stats["count"] += 1.0
        stats["beta_sum"] += beta
        stats["beta_min"] = min(stats["beta_min"], beta)
        stats["beta_max"] = max(stats["beta_max"], beta)
        stats["variance_sum"] += predicted_variance
        stats["disagreement_sum"] += disagreement
        stats["q_residual_abs_sum"] += abs(estimate.q_residual)
        stats["control_residual_abs_sum"] += abs(estimate.control_residual)
        stats["correction_abs_sum"] += abs(estimate.importance_correction)
        stats["centering_abs_sum"] += abs(estimate.policy_weighted_advantage)

    def evaluate(self, **kwargs):
        stats = self._architecture_stats
        count = stats["count"]
        denominator = max(count, 1.0)
        self.logger.record("unbiased_estimator_sample_count", count)
        self.logger.record("control_variate_beta_mean", stats["beta_sum"] / denominator)
        self.logger.record(
            "control_variate_beta_min",
            stats["beta_min"] if count else np.nan,
        )
        self.logger.record(
            "control_variate_beta_max",
            stats["beta_max"] if count else np.nan,
        )
        self.logger.record(
            "predicted_residual_variance_mean",
            stats["variance_sum"] / denominator,
        )
        self.logger.record(
            "q_ensemble_disagreement_mean",
            stats["disagreement_sum"] / denominator,
        )
        self.logger.record(
            "q_residual_abs_mean",
            stats["q_residual_abs_sum"] / denominator,
        )
        self.logger.record(
            "control_residual_abs_mean",
            stats["control_residual_abs_sum"] / denominator,
        )
        self.logger.record(
            "importance_correction_abs_mean",
            stats["correction_abs_sum"] / denominator,
        )
        self.logger.record(
            "policy_weighted_advantage_abs_mean",
            stats["centering_abs_sum"] / denominator,
        )
        self.logger.record(
            "full_support_sampling_min_probability",
            self._minimum_sample_probability,
        )
        self.logger.record(
            "calibration_target_version",
            (
                self.calibration_trainer.target_version
                if self.calibration_trainer is not None
                else 0
            ),
        )
        versions = [member.target_version for member in self.q_value_trainer.members]
        self.logger.record("q_ensemble_target_version_min", min(versions))
        self.logger.record("q_ensemble_target_version_max", max(versions))
        for fold, size in enumerate(self.q_value_trainer.fold_sizes()):
            self.logger.record(f"q_fold_{fold}_replay_size", size)
        for player in range(self.num_players):
            trainer = self.regret_trainers[player]
            self.logger.record(
                f"prediction_gate_player_{player}",
                float(getattr(trainer, "prediction_gate", 0.0)),
            )
            self.logger.record(
                f"prediction_gate_next_player_{player}",
                self.gate_controller.value(player),
            )
            self.logger.record(
                f"predictor_relative_skill_player_{player}",
                self.gate_controller.relative_skill[player],
            )
        return super().evaluate(**kwargs)

    def _traverser_sampling_policy(
        self,
        *,
        q_values,
        beta,
        residual_means,
        predicted_variances,
        policy,
        legal_mask,
    ) -> np.ndarray:
        """Residual-standard-deviation sampling rule for UCV-ESCHER.

        Subclasses may replace the predictable proposal while retaining the
        common full-support importance correction in ``dfs``.
        """

        del q_values, beta, residual_means, policy
        return residual_adaptive_sampling_policy(
            predicted_variances,
            legal_mask,
            uniform_floor_mass=self.sampling_uniform_floor_mass,
            minimum_variance=self.calibration_minimum_variance,
        )

    def dfs(
        self,
        state,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        opp_sample_reach=1.0,
        sample_reach=1.0,
    ):
        del my_reach, opp_reach, opp_sample_reach, sample_reach
        self.nodes_touched += 1
        player = state.current_player()
        if player == -4:
            return state.returns()[traverser] / self.max_utility

        legal_actions = state.legal_actions()
        legal_mask = np.asarray(state.legal_actions_mask(), dtype=float)
        policy = self.regret_trainers[player].get_policy(state, self.num_iteration)
        q_values, disagreement = self.q_value_trainer.get_baseline_and_disagreement(
            state,
            traverser,
        )
        infostate = state.information_state_tensor(traverser)
        if self.calibration_trainer is not None:
            residual_means, predicted_variances, calibration_features = (
                self.calibration_trainer.predict_all(
                    infostate,
                    self.num_iteration,
                    disagreement,
                    traverser,
                )
            )
        else:
            residual_means = np.zeros(self.action_size, dtype=np.float64)
            predicted_variances = np.ones(self.action_size, dtype=np.float64)
            calibration_features = None
        if self.fixed_control_variate_beta is None:
            beta = variance_optimal_beta(
                q_values,
                residual_means,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                ridge=self.beta_ridge,
            )
        else:
            beta = np.full(
                self.action_size,
                self.fixed_control_variate_beta,
                dtype=np.float64,
            )
        if player == traverser:
            sample_policy = self._traverser_sampling_policy(
                q_values=q_values,
                beta=beta,
                residual_means=residual_means,
                predicted_variances=predicted_variances,
                policy=policy,
                legal_mask=legal_mask,
            )
        else:
            sample_policy = np.where(legal_mask > 0.0, policy, 0.0)
            sample_policy /= float(np.sum(sample_policy))
        action = int(
            np.random.choice(
                range(state.num_distinct_actions()),
                p=sample_policy,
            )
        )
        sample_probability = float(sample_policy[action])
        if player == traverser:
            self._minimum_sample_probability = min(
                self._minimum_sample_probability,
                sample_probability,
            )

        next_state = self.skip_chance_state(state.child(action))
        sampled_return = self.dfs(next_state, traverser)
        estimate = control_variate_advantage(
            q_values,
            beta=beta,
            sampled_action=action,
            sample_probability=sample_probability,
            sampled_return=sampled_return,
            policy=policy,
            legal_actions_mask=legal_mask,
        )
        observe_control_return = getattr(
            self.q_value_trainer,
            "observe_control_return",
            None,
        )
        if observe_control_return is not None:
            # Optional control-critic controllers receive only the sampled
            # return after all mixture and sampling decisions were made.
            observe_control_return(
                state=state,
                player=traverser,
                action=action,
                sampled_return=sampled_return,
                iteration=self.num_iteration,
            )
        if self.calibration_trainer is not None:
            self.calibration_trainer.add(
                calibration_features[action],
                estimate.q_residual,
            )
        self._record_estimate_diagnostics(
            beta=float(beta[action]),
            predicted_variance=float(predicted_variances[action]),
            disagreement=float(disagreement[action]),
            estimate=estimate,
        )

        if player == traverser:
            self.regret_trainers[player].add_data(
                self.get_infostate_tensor(state),
                estimate.advantages,
                legal_mask,
                self.num_iteration,
            )
        else:
            self.ave_policy_trainer.add_data(
                self.get_infostate_tensor(state),
                policy,
                legal_mask,
                self.num_iteration,
            )

        self.q_value_trainer.add_data(
            self.get_history_tensor(state),
            action,
            self.get_history_tensor(next_state),
            self.get_infostate_tensor(next_state) if not next_state.is_terminal() else None,
            next_state.legal_actions_mask() if not next_state.is_terminal() else None,
            next_state.current_player(),
            int(next_state.is_terminal()),
            next_state.returns()[0] / self.max_utility,
        )
        return estimate.policy_value
