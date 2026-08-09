"""Adaptive residual-corrected predictive ESCHER solver.

This module reuses the audited PDCFR+ cumulative and instantaneous advantage
trainers, but replaces its traversal estimator and reinitialised Q learner.
The baseline VR solvers are not modified.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from vr_deep_cfr.solver import SpielState
from vr_deep_cfr.variants import VRDeepPDCFRPlus, VRPDCFRPlusQValueTrainer

from .estimator import AdaptiveLambdaController, adaptive_residual_corrected_advantage


class PersistentFrozenTargetQValueTrainer(VRPDCFRPlusQValueTrainer):
    """Persistent all-action Q learner with a frozen collection/TD snapshot.

    The upstream VR-PDCFR+ implementation reinitialises its online and target Q
    networks every time it trains. Here the online network and optimiser persist
    across iterations. ``target_model`` remains frozen for the complete training
    call and is hard-synchronised only after all gradient steps finish. Traversal
    inference also uses that target snapshot, so every trajectory segment sees
    stable all-action Q estimates.
    """

    def __init__(self, *args, gradient_clip_norm: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.target_version = 0

    def get_baseline(self, s: SpielState, player: int) -> np.ndarray:
        history_tensor = self.get_history_tensor(s)
        coefficient = 1.0 if player == 0 else -1.0
        x = torch.as_tensor(history_tensor, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            baseline = self.target_model(x).cpu().numpy() * coefficient
        return baseline * np.asarray(s.legal_actions_mask(), dtype=float)

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
            cumulative = trainer.predict(
                trainer.model, next_states, next_legal_actions_mask
            )
            immediate = trainer.predict(
                trainer.imm_model, next_states, next_legal_actions_mask
            )
            factor = math.pow(iteration, trainer.alpha) / (
                math.pow(iteration, trainer.alpha) + 1.0
            )
            predictive = torch.clamp(
                torch.clamp(cumulative, min=0.0) * factor + immediate,
                min=0.0,
            )
            predictive = predictive * next_legal_actions_mask
            positive_sum = predictive.sum(dim=1, keepdim=True)
            normalised = predictive / torch.clamp(positive_sum, min=1e-12)

            masked_for_argmax = torch.where(
                next_legal_actions_mask == 1,
                predictive,
                torch.full_like(predictive, float("-inf")),
            )
            argmax_ids = torch.argmax(masked_for_argmax, dim=1)
            fallback = F.one_hot(argmax_ids, self.output_size).to(predictive.dtype)
            strategy = torch.where(positive_sum > 0.0, normalised, fallback)
            player_strategies.append(strategy)

        return torch.where(
            next_players.unsqueeze(1) == 0,
            player_strategies[0],
            player_strategies[1],
        )

    def train_model(self, T):
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return None
        if len(self.buffer) == 0 or self.train_steps <= 0:
            return None

        self.model.train()
        final_loss = None
        for train_step in range(self.train_steps):
            (
                histories,
                next_histories,
                next_states,
                rewards,
                next_legal_actions_mask,
                next_players,
                actions,
                dones,
            ) = self.buffer.sample(self.batch_size)

            q_value = self.model(histories).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = self.target_model(next_histories)
                next_strategies = self._batched_predictive_strategies(
                    next_states,
                    next_legal_actions_mask,
                    next_players,
                    int(T),
                )
                continuation = torch.sum(next_q_values * next_strategies, dim=1)
                target = rewards + (1 - dones) * continuation

            loss = self.loss_fn(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.optimizer.step()
            final_loss = float(loss.item())
            if train_step % 100 == 0:
                self.logger.info(
                    f"train_step[{train_step}/{self.train_steps}]: "
                    f"persistent Q loss {final_loss}"
                )

        # No traversal or TD target can observe a partially updated Q model.
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_version += 1
        return final_loss


class AdaptiveResidualPredictiveEscher(VRDeepPDCFRPlus):
    """Predictive discounted ESCHER with adaptive residual correction."""

    def __init__(
        self,
        *args,
        lambda_start: float = 0.2,
        lambda_schedule_half_life: float = 2.0,
        lambda_schedule_power: float = 1.0,
        lambda_residual_ema_decay: float = 0.99,
        lambda_residual_scale: float = 0.25,
        lambda_initial_residual: float = 1.0,
        q_gradient_clip_norm: float = 10.0,
        sampling_mode: str = "fixed_uniform",
        **kwargs,
    ):
        if sampling_mode != "fixed_uniform":
            raise ValueError(
                "AdaptiveResidualPredictiveEscher currently requires fixed_uniform "
                "sampling for the traverser so every updated legal action has "
                "time-independent positive support"
            )
        self.lambda_start = float(lambda_start)
        self.lambda_schedule_half_life = float(lambda_schedule_half_life)
        self.lambda_schedule_power = float(lambda_schedule_power)
        self.lambda_residual_ema_decay = float(lambda_residual_ema_decay)
        self.lambda_residual_scale = float(lambda_residual_scale)
        self.lambda_initial_residual = float(lambda_initial_residual)
        self.q_gradient_clip_norm = float(q_gradient_clip_norm)
        self.sampling_mode = sampling_mode
        super().__init__(*args, **kwargs)
        if not self.use_baseline:
            raise ValueError("Adaptive residual correction requires use_baseline=True")
        if not self.fit_advantage:
            raise ValueError("The predictive accumulator must fit advantages")
        self.lambda_controller = AdaptiveLambdaController(
            self.num_players,
            self.action_size,
            lambda_start=self.lambda_start,
            schedule_half_life=self.lambda_schedule_half_life,
            schedule_power=self.lambda_schedule_power,
            residual_ema_decay=self.lambda_residual_ema_decay,
            residual_scale=self.lambda_residual_scale,
            initial_residual=self.lambda_initial_residual,
        )
        self._minimum_sample_probability = 1.0

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = PersistentFrozenTargetQValueTrainer(
            history_size,
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.baseline_buffer_size,
            self.baseline_batch_size,
            self.baseline_network_train_steps,
            self.logger,
            self.regret_trainers,
            self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
        )

    def iteration(self):
        self.lambda_controller.reset_diagnostics()
        self._minimum_sample_probability = 1.0
        self.num_iteration += 1
        # Preserve the paper implementation's alternating advantage updates,
        # but delay Q optimisation until both traversers have collected data.
        # Consequently both players' samples in one outer iteration use the
        # identical frozen Q snapshot.
        for player in range(self.num_players):
            self.collect_training_data(player)
            self.train_regret(player)
        for player in range(self.num_players):
            self.train_baseline(player)
        if self.num_iteration % self.evaluation_frequency == 0:
            self._run_checkpoint(checkpoint_kind="outer_iteration")

    def evaluate(self, **kwargs):
        for key, value in self.lambda_controller.diagnostics(
            max(1, self.num_iteration)
        ).items():
            self.logger.record(key, value)
        self.logger.record(
            "full_support_traverser_sampling_min_probability",
            self._minimum_sample_probability,
        )
        self.logger.record("q_target_version", self.q_value_trainer.target_version)
        return super().evaluate(**kwargs)

    def dfs(
        self,
        s,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        opp_sample_reach=1.0,
        sample_reach=1.0,
    ):
        del my_reach, opp_reach, opp_sample_reach, sample_reach
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility

        legal_actions = s.legal_actions()
        legal_mask = np.asarray(s.legal_actions_mask(), dtype=float)
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)

        # Match ESCHER's sampling contract: the updating player's policy is
        # fixed and full support; the opponent is sampled from the current
        # strategy so average-strategy observations retain own-reach weighting.
        if player == traverser:
            sample_policy = legal_mask / float(np.sum(legal_mask))
        else:
            sample_policy = np.where(legal_mask > 0.0, policy, 0.0)
            sample_policy /= float(np.sum(sample_policy))
        action = int(np.random.choice(range(s.num_distinct_actions()), p=sample_policy))
        sample_probability = float(sample_policy[action])
        if player == traverser:
            self._minimum_sample_probability = min(
                self._minimum_sample_probability, sample_probability
            )

        # Both Q and lambda are chosen before the current return is observed.
        q_values = self.q_value_trainer.get_baseline(s, traverser)
        lambda_value = self.lambda_controller.value(
            traverser, action, self.num_iteration
        )
        next_state = self.skip_chance_state(s.child(action))
        sampled_return = self.dfs(next_state, traverser)

        estimate = adaptive_residual_corrected_advantage(
            q_values,
            sampled_action=action,
            sample_probability=sample_probability,
            sampled_return=sampled_return,
            lambda_value=lambda_value,
            policy=policy,
            legal_actions_mask=legal_mask,
        )
        self.lambda_controller.observe(
            traverser,
            action,
            sampled_residual=estimate.sampled_residual,
            lambda_value=lambda_value,
            residual_correction=estimate.residual_correction,
            policy_weighted_advantage=estimate.policy_weighted_advantage,
        )

        if player == traverser:
            self.regret_trainers[player].add_data(
                self.get_infostate_tensor(s),
                estimate.advantages,
                legal_mask,
                self.num_iteration,
            )
        else:
            self.ave_policy_trainer.add_data(
                self.get_infostate_tensor(s),
                policy,
                legal_mask,
                self.num_iteration,
            )

        self.q_value_trainer.add_data(
            self.get_history_tensor(s),
            action,
            self.get_history_tensor(next_state),
            self.get_infostate_tensor(next_state) if not next_state.is_terminal() else None,
            next_state.legal_actions_mask() if not next_state.is_terminal() else None,
            next_state.current_player(),
            int(next_state.is_terminal()),
            next_state.returns()[0] / self.max_utility,
        )
        return estimate.policy_value
