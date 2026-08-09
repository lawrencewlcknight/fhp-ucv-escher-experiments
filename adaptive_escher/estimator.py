"""Bias--variance controlled action-value and advantage estimates.

The estimator deliberately lives outside the solver so its limiting cases and
centering invariant can be tested without running a poker environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class AdaptiveValueEstimate:
    """One residual-corrected all-action estimate and its centred advantage."""

    q_values: np.ndarray
    advantages: np.ndarray
    policy_value: float
    sampled_residual: float
    residual_correction: float
    policy_weighted_advantage: float


def adaptive_residual_corrected_advantage(
    q_values,
    *,
    sampled_action: int,
    sample_probability: float,
    sampled_return: float,
    lambda_value: float,
    policy,
    legal_actions_mask,
) -> AdaptiveValueEstimate:
    """Apply a shrunken sampled residual and centre over the current policy.

    For legal action ``a`` the action-value estimate is

    ``Q_hat[a] + lambda * 1{A=a} / xi[A] * (G - Q_hat[A])``.

    Illegal actions are zeroed and the policy is renormalised over legal
    actions before centering. This makes the policy-weighted instantaneous
    advantage zero up to floating-point precision for every sample.
    """

    values = np.asarray(q_values, dtype=np.float64)
    policy_array = np.asarray(policy, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    if values.ndim != 1 or policy_array.shape != values.shape or legal_mask.shape != values.shape:
        raise ValueError("q_values, policy and legal_actions_mask must be matching 1-D arrays")
    legal = legal_mask > 0.0
    if not np.any(legal):
        raise ValueError("legal_actions_mask must contain at least one legal action")
    if sampled_action < 0 or sampled_action >= values.size or not legal[sampled_action]:
        raise ValueError("sampled_action must be legal")
    if not np.isfinite(sample_probability) or sample_probability <= 0.0:
        raise ValueError("sample_probability must be positive and finite")
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must lie in [0, 1]")
    if not np.isfinite(sampled_return) or not np.all(np.isfinite(values[legal])):
        raise ValueError("sampled_return and legal Q values must be finite")

    legal_policy = np.where(legal, policy_array, 0.0)
    policy_mass = float(np.sum(legal_policy))
    if not np.isfinite(policy_mass) or policy_mass <= 0.0:
        raise ValueError("policy must assign positive finite mass to legal actions")
    legal_policy /= policy_mass

    corrected = np.where(legal, values, 0.0).copy()
    sampled_residual = float(sampled_return - corrected[sampled_action])
    residual_correction = float(lambda_value * sampled_residual / sample_probability)
    corrected[sampled_action] += residual_correction
    policy_value = float(np.dot(legal_policy, corrected))
    advantages = np.where(legal, corrected - policy_value, 0.0)
    policy_weighted_advantage = float(np.dot(legal_policy, advantages))
    return AdaptiveValueEstimate(
        q_values=corrected,
        advantages=advantages,
        policy_value=policy_value,
        sampled_residual=sampled_residual,
        residual_correction=residual_correction,
        policy_weighted_advantage=policy_weighted_advantage,
    )


class AdaptiveLambdaController:
    """Predictable residual calibration with an asymptotically-one floor.

    Lambda is selected before observing the current rollout return. Its
    uncertainty component is based only on an EMA of *past* absolute Q
    residuals, preserving the usual conditional-expectation calculation. A
    deterministic floor tends to one:

    ``1 - (1 - lambda_start) / (1 + (t - 1) / half_life) ** power``.

    With bounded Q error and ``power > 0``, the average shrinkage bias vanishes;
    ``power >= 1`` gives at most logarithmic cumulative shrinkage for bounded
    error. The residual term may raise lambda above this floor when the frozen
    Q snapshot is poorly calibrated.
    """

    def __init__(
        self,
        num_players: int,
        num_actions: int,
        *,
        lambda_start: float = 0.2,
        schedule_half_life: float = 2.0,
        schedule_power: float = 1.0,
        residual_ema_decay: float = 0.99,
        residual_scale: float = 0.25,
        initial_residual: float = 1.0,
    ):
        if num_players <= 0 or num_actions <= 0:
            raise ValueError("num_players and num_actions must be positive")
        if not 0.0 <= lambda_start <= 1.0:
            raise ValueError("lambda_start must lie in [0, 1]")
        if schedule_half_life <= 0.0 or schedule_power <= 0.0:
            raise ValueError("schedule_half_life and schedule_power must be positive")
        if not 0.0 <= residual_ema_decay < 1.0:
            raise ValueError("residual_ema_decay must lie in [0, 1)")
        if residual_scale <= 0.0 or initial_residual < 0.0:
            raise ValueError("residual_scale must be positive and initial_residual non-negative")
        self.lambda_start = float(lambda_start)
        self.schedule_half_life = float(schedule_half_life)
        self.schedule_power = float(schedule_power)
        self.residual_ema_decay = float(residual_ema_decay)
        self.residual_scale = float(residual_scale)
        self.residual_ema = np.full(
            (int(num_players), int(num_actions)),
            float(initial_residual),
            dtype=np.float64,
        )
        self._stats: Dict[str, float] = {}
        self.reset_diagnostics()

    def schedule_floor(self, iteration: int) -> float:
        if iteration < 1:
            raise ValueError("iteration must be at least one")
        age = float(iteration - 1) / self.schedule_half_life
        return float(
            1.0
            - (1.0 - self.lambda_start)
            / np.power(1.0 + age, self.schedule_power)
        )

    def value(self, player: int, action: int, iteration: int) -> float:
        residual = float(self.residual_ema[player, action])
        uncertainty_lambda = residual / (residual + self.residual_scale)
        return float(np.clip(max(self.schedule_floor(iteration), uncertainty_lambda), 0.0, 1.0))

    def observe(
        self,
        player: int,
        action: int,
        *,
        sampled_residual: float,
        lambda_value: float,
        residual_correction: float,
        policy_weighted_advantage: float,
    ) -> None:
        absolute_residual = abs(float(sampled_residual))
        old = float(self.residual_ema[player, action])
        self.residual_ema[player, action] = (
            self.residual_ema_decay * old
            + (1.0 - self.residual_ema_decay) * absolute_residual
        )
        self._stats["count"] += 1.0
        self._stats["lambda_sum"] += float(lambda_value)
        self._stats["lambda_min"] = min(self._stats["lambda_min"], float(lambda_value))
        self._stats["lambda_max"] = max(self._stats["lambda_max"], float(lambda_value))
        self._stats["residual_abs_sum"] += absolute_residual
        self._stats["correction_abs_sum"] += abs(float(residual_correction))
        self._stats["centering_abs_sum"] += abs(float(policy_weighted_advantage))

    def diagnostics(self, iteration: int) -> Dict[str, float]:
        count = self._stats["count"]
        denominator = max(count, 1.0)
        return {
            "adaptive_lambda_schedule_floor": self.schedule_floor(max(1, iteration)),
            "adaptive_lambda_mean": self._stats["lambda_sum"] / denominator,
            "adaptive_lambda_min": (
                self._stats["lambda_min"] if count else float("nan")
            ),
            "adaptive_lambda_max": (
                self._stats["lambda_max"] if count else float("nan")
            ),
            "q_residual_abs_mean": self._stats["residual_abs_sum"] / denominator,
            "residual_correction_abs_mean": self._stats["correction_abs_sum"] / denominator,
            "policy_weighted_advantage_abs_mean": self._stats["centering_abs_sum"] / denominator,
            "adaptive_estimator_sample_count": count,
        }

    def reset_diagnostics(self) -> None:
        self._stats = {
            "count": 0.0,
            "lambda_sum": 0.0,
            "lambda_min": float("inf"),
            "lambda_max": float("-inf"),
            "residual_abs_sum": 0.0,
            "correction_abs_sum": 0.0,
            "centering_abs_sum": 0.0,
        }
