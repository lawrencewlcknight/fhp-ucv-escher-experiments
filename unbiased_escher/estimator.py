"""Unbiased control-variate estimator and predictable adaptive controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlVariateEstimate:
    """One all-action estimate and its policy-centred advantage."""

    q_values: np.ndarray
    advantages: np.ndarray
    control_values: np.ndarray
    policy_value: float
    q_residual: float
    control_residual: float
    importance_correction: float
    policy_weighted_advantage: float


def _legal_policy(policy, legal_mask) -> np.ndarray:
    values = np.asarray(policy, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=np.float64) > 0.0
    result = np.where(legal, values, 0.0)
    mass = float(np.sum(result))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("policy must assign positive finite mass to legal actions")
    return result / mass


def control_variate_advantage(
    q_values,
    *,
    beta,
    sampled_action: int,
    sample_probability: float,
    sampled_return: float,
    policy,
    legal_actions_mask,
) -> ControlVariateEstimate:
    """Construct an always-unbiased all-action estimate and centre it.

    For each legal action ``a`` this computes

    ``beta[a] * Q_hat[a] + 1{A=a}/xi[A] * (G-beta[A]*Q_hat[A])``.

    ``beta`` and ``xi`` must be selected before ``G`` is observed. Under that
    predictability condition and full support, the conditional expectation of
    every legal action value is exactly its true continuation value for any
    finite beta.
    """

    values = np.asarray(q_values, dtype=np.float64)
    coefficients = np.asarray(beta, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    if (
        values.ndim != 1
        or coefficients.shape != values.shape
        or legal_mask.shape != values.shape
    ):
        raise ValueError("q_values, beta and legal_actions_mask must match")
    legal = legal_mask > 0.0
    if not np.any(legal):
        raise ValueError("legal_actions_mask must contain a legal action")
    if sampled_action < 0 or sampled_action >= values.size or not legal[sampled_action]:
        raise ValueError("sampled_action must be legal")
    if not np.isfinite(sample_probability) or sample_probability <= 0.0:
        raise ValueError("sample_probability must be positive and finite")
    if (
        not np.isfinite(sampled_return)
        or not np.all(np.isfinite(values[legal]))
        or not np.all(np.isfinite(coefficients[legal]))
    ):
        raise ValueError("return, legal Q values and legal beta values must be finite")

    legal_policy = _legal_policy(policy, legal_mask)
    controls = np.where(legal, coefficients * values, 0.0)
    corrected = controls.copy()
    control_residual = float(sampled_return - controls[sampled_action])
    importance_correction = control_residual / float(sample_probability)
    corrected[sampled_action] += importance_correction
    policy_value = float(np.dot(legal_policy, corrected))
    advantages = np.where(legal, corrected - policy_value, 0.0)
    return ControlVariateEstimate(
        q_values=corrected,
        advantages=advantages,
        control_values=controls,
        policy_value=policy_value,
        q_residual=float(sampled_return - values[sampled_action]),
        control_residual=control_residual,
        importance_correction=float(importance_correction),
        policy_weighted_advantage=float(np.dot(legal_policy, advantages)),
    )


def variance_optimal_beta(
    q_values,
    predicted_residual_means,
    *,
    beta_min: float,
    beta_max: float,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Estimate the variance-minimising coefficient from held-out residuals.

    For the Horvitz--Thompson control-variate estimator the conditional
    variance is minimised when ``beta * Q_hat = E[G | I,a]``. A frozen
    calibration model supplies the held-out estimate
    ``E[G-Q_hat | I,a]``. The ridge-stabilised ratio below approaches
    ``E[G]/Q_hat`` away from zero and remains bounded near zero.
    """

    if beta_min > beta_max or ridge <= 0.0:
        raise ValueError("beta bounds and ridge must define a valid interval")
    q_values = np.asarray(q_values, dtype=np.float64)
    residual_means = np.asarray(predicted_residual_means, dtype=np.float64)
    if q_values.shape != residual_means.shape:
        raise ValueError("q_values and predicted_residual_means must match")
    coefficients = 1.0 + residual_means * q_values / (
        np.square(q_values) + float(ridge)
    )
    return np.clip(coefficients, beta_min, beta_max)


def residual_adaptive_sampling_policy(
    predicted_variances,
    legal_actions_mask,
    *,
    uniform_floor_mass: float,
    minimum_variance: float = 1e-6,
) -> np.ndarray:
    """Allocate samples by residual standard deviation with uniform support."""

    if not 0.0 < uniform_floor_mass <= 1.0:
        raise ValueError("uniform_floor_mass must lie in (0, 1]")
    if minimum_variance <= 0.0:
        raise ValueError("minimum_variance must be positive")
    variances = np.asarray(predicted_variances, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    if variances.shape != legal_mask.shape or variances.ndim != 1:
        raise ValueError("predicted_variances and legal_actions_mask must match")
    legal = legal_mask > 0.0
    count = int(np.sum(legal))
    if count == 0:
        raise ValueError("legal_actions_mask must contain a legal action")
    safe = np.where(
        legal,
        np.maximum(np.nan_to_num(variances, nan=minimum_variance), minimum_variance),
        0.0,
    )
    scores = np.sqrt(safe)
    adaptive = scores / float(np.sum(scores))
    uniform = np.where(legal, 1.0 / float(count), 0.0)
    result = (1.0 - uniform_floor_mass) * adaptive + uniform_floor_mass * uniform
    result = np.where(legal, result, 0.0)
    return result / float(np.sum(result))
