"""Leduc Experiment 35 grouped-wide UCV mechanisms adapted for FHP."""

from __future__ import annotations

from collections import deque
import math
import random

import numpy as np
import torch

from vr_deep_cfr.solver import AvePolicyTrainer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .solver import (
    CrossFittedQEnsemble,
    CrossFittedQMember,
    UnbiasedControlVariateEscher,
)


GROUPED_SOFT_TARGET_CROSS_ENTROPY = "grouped_soft_target_cross_entropy"


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


class GroupedSoftTargetCrossEntropyAvePolicyTrainer(AvePolicyTrainer):
    """Fit one iteration-weighted soft target per observed information set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grouped_target_lookup = {}
        self.grouped_num_rows = 0
        self.grouped_num_information_sets = 0
        self.grouped_reduction_ratio = 0.0

    def _grouped_training_data(self, iteration):
        size = min(int(self.buffer.cur_id), int(self.buffer.buffer_size))
        if size <= 0:
            raise ValueError("Cannot fit an average policy from an empty reservoir")
        features = np.asarray(self.buffer.infostate_buf[:size], dtype=np.float32)
        policies = np.asarray(self.buffer.q_value_buf[:size], dtype=np.float64)
        masks = np.asarray(self.buffer.q_value_mask_buf[:size], dtype=np.float32)
        iterations = np.asarray(
            self.buffer.iteration_buf[:size], dtype=np.float64
        ).reshape(-1)
        raw_weights = np.power(
            np.maximum(iterations, 0.0) / float(iteration) * 2.0,
            self.gamma,
            dtype=np.float64,
        )
        unique, inverse, counts = np.unique(
            features, axis=0, return_inverse=True, return_counts=True
        )
        masses = np.zeros(len(unique), dtype=np.float64)
        numerators = np.zeros((len(unique), policies.shape[1]), dtype=np.float64)
        np.add.at(masses, inverse, raw_weights)
        np.add.at(numerators, inverse, policies * raw_weights[:, None])
        if np.any(masses <= 0.0):
            raise ValueError("Grouped average-policy target has non-positive mass")
        targets = numerators / masses[:, None]

        order = np.argsort(inverse, kind="stable")
        first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
        grouped_masks = masks[order[first]]
        for row_index, group_index in enumerate(inverse):
            if not np.array_equal(masks[row_index], grouped_masks[group_index]):
                raise ValueError(
                    "Legal-action masks differ within an information set"
                )

        objective_weights = masses * (float(len(unique)) / float(size))
        self.grouped_num_rows = int(size)
        self.grouped_num_information_sets = int(len(unique))
        self.grouped_reduction_ratio = float(len(unique)) / float(size)
        # Experiment 35 retained a lookup solely for exact Leduc empirical-
        # policy diagnostics. FHP cannot perform that tree enumeration; keeping
        # up to one million byte-string keys would add material memory without
        # affecting the grouped training objective.
        self.grouped_target_lookup = {}
        return tuple(
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in (unique, targets, grouped_masks, objective_weights)
        )

    def train_model(self, iteration):
        features, targets, masks, weights = self._grouped_training_data(iteration)
        size = len(features)
        loss = None
        for train_step in range(self.train_steps):
            if self.batch_size == -1 or self.batch_size >= size:
                indices = torch.arange(size, device=self.device)
            else:
                indices = torch.as_tensor(
                    random.sample(range(size), int(self.batch_size)),
                    dtype=torch.long,
                    device=self.device,
                )
            logits = self.model(features.index_select(0, indices))
            selected_masks = masks.index_select(0, indices)
            logits = logits.masked_fill(selected_masks != 1, -1e20)
            per_sample = -(
                targets.index_select(0, indices)
                * torch.log_softmax(logits, dim=-1)
            ).sum(dim=1)
            loss = torch.mean(per_sample * weights.index_select(0, indices))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    f"[{train_step}/{self.train_steps}] grouped policy loss: "
                    f"{loss.item()}"
                )
        if loss is None:
            raise ValueError("Average-policy training requires at least one update")
        return float(loss.item())


class TemporallyAveragedCrossFittedQMember(CrossFittedQMember):
    """Critic whose frozen target is the mean of recent completed fits."""

    def __init__(self, *args, target_average_window: int, **kwargs):
        self.target_average_window = int(target_average_window)
        if self.target_average_window <= 0:
            raise ValueError("target_average_window must be positive")
        self.target_history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.target_average_window
        )
        self.last_target_update_l2 = 0.0
        super().__init__(*args, **kwargs)

    def _install_temporal_average(self, previous):
        self.target_history.append(_cpu_state_dict(self.model))
        averaged = {}
        for name in self.target_history[0]:
            tensors = [snapshot[name] for snapshot in self.target_history]
            averaged[name] = (
                torch.stack(tensors, dim=0).mean(dim=0)
                if torch.is_floating_point(tensors[0])
                else tensors[-1]
            )
        self.target_model.load_state_dict(averaged)
        self.target_model.eval()
        squared = 0.0
        for name, value in averaged.items():
            if torch.is_floating_point(value):
                difference = value - previous[name]
                squared += float(torch.sum(difference * difference).item())
        self.last_target_update_l2 = math.sqrt(squared)

    def train_model(self, iteration: int):
        previous = _cpu_state_dict(self.target_model)
        loss = super().train_model(iteration)
        if loss is not None:
            self._install_temporal_average(previous)
        return loss


class TemporallyAveragedCrossFittedQEnsemble(CrossFittedQEnsemble):
    """Cross-fitted ensemble with temporal averaging inside every fold."""

    member_class = TemporallyAveragedCrossFittedQMember

    def __init__(
        self,
        *,
        target_average_window: int,
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
        self.ensemble_size = int(ensemble_size)
        self.target_average_window = int(target_average_window)
        if self.ensemble_size < 1 or self.target_average_window < 1:
            raise ValueError("Ensemble size and target-average window must be positive")
        member_buffer_size = max(1, int(total_buffer_size) // self.ensemble_size)
        self.members = [
            self.member_class(
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
                target_average_window=self.target_average_window,
            )
            for _ in range(self.ensemble_size)
        ]
        self.active_fold = 0
        self._active_add_count = 0


class GroupedWideUnbiasedControlVariateEscher(UnbiasedControlVariateEscher):
    """FHP UCV-ESCHER using the selected Leduc Experiment 35 configuration."""

    nonpredictive_regret_trainer_class = VRDCFRPlusRegretTrainer
    average_policy_trainer_class = GroupedSoftTargetCrossEntropyAvePolicyTrainer
    q_ensemble_class = TemporallyAveragedCrossFittedQEnsemble

    def __init__(
        self,
        *args,
        critic_target_average_window: int = 4,
        average_policy_loss: str = GROUPED_SOFT_TARGET_CROSS_ENTROPY,
        average_policy_reset_each_fit: bool = True,
        average_policy_network_layers=(136, 136, 136),
        average_policy_learning_rate: float = 3e-3,
        average_policy_train_steps: int = 20_000,
        paired_legacy_network_layers=(64, 64, 64),
        paired_legacy_learning_rate: float = 1e-3,
        paired_legacy_train_steps: int = 5_000,
        regret_network_type: str = "mlp",
        regret_residual_width: int = 64,
        regret_residual_blocks: int = 4,
        regret_policy_gradient_clip_norm=None,
        anneal_start_nodes=None,
        anneal_end_nodes=None,
        anneal_final_learning_rate=None,
        **kwargs,
    ):
        self.critic_target_average_window = int(critic_target_average_window)
        self.average_policy_loss = str(average_policy_loss)
        self.average_policy_reset_each_fit = bool(average_policy_reset_each_fit)
        self.average_policy_network_layers = tuple(
            int(value) for value in average_policy_network_layers
        )
        self.average_policy_learning_rate = float(average_policy_learning_rate)
        self.average_policy_train_steps = int(average_policy_train_steps)
        self.paired_legacy_network_layers = tuple(
            int(value) for value in paired_legacy_network_layers
        )
        self.paired_legacy_learning_rate = float(paired_legacy_learning_rate)
        self.paired_legacy_train_steps = int(paired_legacy_train_steps)
        self.regret_network_type = str(regret_network_type)
        self.regret_residual_width = int(regret_residual_width)
        self.regret_residual_blocks = int(regret_residual_blocks)
        self.regret_policy_gradient_clip_norm = regret_policy_gradient_clip_norm
        self.anneal_start_nodes = anneal_start_nodes
        self.anneal_end_nodes = anneal_end_nodes
        self.anneal_final_learning_rate = anneal_final_learning_rate
        if self.critic_target_average_window <= 0:
            raise ValueError("critic_target_average_window must be positive")
        if self.average_policy_loss != GROUPED_SOFT_TARGET_CROSS_ENTROPY:
            raise ValueError("Grouped soft-target cross-entropy is required")
        if not self.average_policy_reset_each_fit:
            raise ValueError("The Experiment 35 policy must reset for every fit")
        if self.regret_network_type != "mlp":
            raise ValueError("Experiment 35 requires the MLP regret network")
        if self.regret_policy_gradient_clip_norm is not None:
            raise ValueError("Experiment 35 does not clip regret gradients")
        if any(
            value is not None
            for value in (
                self.anneal_start_nodes,
                self.anneal_end_nodes,
                self.anneal_final_learning_rate,
            )
        ):
            raise ValueError("Experiment 35 disables regret learning-rate annealing")
        super().__init__(*args, **kwargs)
        if self.use_instantaneous_predictor:
            raise ValueError("Experiment 35 requires the non-predictive UCV core")

    def init_regret_trainers(self):
        if self.use_instantaneous_predictor:
            return super().init_regret_trainers()
        self.regret_trainers = [
            self.nonpredictive_regret_trainer_class(
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

    def init_ave_policy_trainer(self):
        # Experiment 35 constructed its legacy 3x64 head first, then restored
        # that post-construction RNG state after creating the wider candidate.
        # Retain the seed stream without retaining or fitting the Leduc-only
        # paired diagnostic. Buffer construction is deterministic, so a
        # one-row temporary buffer avoids a needless FHP-scale allocation.
        legacy_rng_anchor = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            list(self.paired_legacy_network_layers),
            self.paired_legacy_learning_rate,
            1,
            self.ave_policy_batch_size,
            self.paired_legacy_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )
        rng_after_legacy = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_after_legacy["cuda"] = torch.cuda.get_rng_state_all()
        self.ave_policy_trainer = self.average_policy_trainer_class(
            self.infostate_size,
            self.action_size,
            list(self.average_policy_network_layers),
            self.average_policy_learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.average_policy_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )
        random.setstate(rng_after_legacy["python"])
        np.random.set_state(rng_after_legacy["numpy"])
        torch.random.set_rng_state(rng_after_legacy["torch"])
        if "cuda" in rng_after_legacy:
            torch.cuda.set_rng_state_all(rng_after_legacy["cuda"])
        del legacy_rng_anchor

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = self.q_ensemble_class(
            target_average_window=self.critic_target_average_window,
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

    def collect_training_data(self, player):
        """Collect a complete player batch before any resumable checkpoint."""
        self.regret_trainers[player].reset_buffer()
        for _ in range(self.num_traversals):
            self.episode += 1
            self.q_value_trainer.begin_trajectory(self.episode)
            root_state = self.skip_chance_state(self.game.new_initial_state())
            self.dfs(root_state, player)
            self._maybe_run_early_node_checkpoint()

    def iteration(self):
        """Checkpoint only after the entire outer iteration is resumable."""
        super().iteration()
        self._maybe_run_training_time_checkpoint()

    def evaluate(self, **kwargs):
        self.logger.record(
            "grouped_num_rows", self.ave_policy_trainer.grouped_num_rows
        )
        self.logger.record(
            "grouped_num_information_sets",
            self.ave_policy_trainer.grouped_num_information_sets,
        )
        self.logger.record(
            "grouped_reduction_ratio",
            self.ave_policy_trainer.grouped_reduction_ratio,
        )
        updates = [
            member.last_target_update_l2 for member in self.q_value_trainer.members
        ]
        self.logger.record("critic_target_average_window", self.critic_target_average_window)
        self.logger.record("critic_target_update_l2_mean", float(np.mean(updates)))
        return super().evaluate(**kwargs)


__all__ = [
    "GROUPED_SOFT_TARGET_CROSS_ENTROPY",
    "GroupedSoftTargetCrossEntropyAvePolicyTrainer",
    "GroupedWideUnbiasedControlVariateEscher",
    "TemporallyAveragedCrossFittedQEnsemble",
    "TemporallyAveragedCrossFittedQMember",
]
