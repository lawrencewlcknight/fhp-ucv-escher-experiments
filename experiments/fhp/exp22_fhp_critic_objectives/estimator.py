"""Differentiable version of the production recursive control-variate algebra.

For fixed history h, policies and predictable proposal, let R_b(h,Z) be the
native centred estimate. Two INDEPENDENT continuations conditional on h give
E[||R_b(h,Z1)-R_b(h,Z2)||^2/2] = tr Cov(R_b | h). Thus its mean is not penalised.
We divide by the number of legal root actions. Cross-action covariance enters
through centring, not by shrinking genuine action differences. Chance sampling
is included in Z. Models never control the frozen proposal during this audit.
"""
import torch
from torch.nn import functional as F


def corrected(q, policy, mask, action, probability, continuation):
    """Vectorised beta=1 equivalent of control_variate_advantage."""
    if torch.any(probability <= 0) or not torch.isfinite(probability).all():
        raise ValueError("Sampled action probabilities must be finite and positive")
    policy = policy * mask
    policy = policy / policy.sum(-1, keepdim=True)
    controls = q * mask
    residual = continuation - controls.gather(-1, action[..., None]).squeeze(-1)
    values = controls + F.one_hot(action, q.shape[-1]) * (residual / probability)[..., None]
    value = (values * policy).sum(-1)
    advantage = (values - value[..., None]) * mask
    return value, advantage


def recursive_estimates(model, batch):
    """All network calls are batched; only the short game-depth recurrence loops."""
    features, active = batch["features"], batch["active"]
    n, depth, _ = features.shape
    # Do not run the network on padding. Preserve gradients through scatter.
    flat = features[active]
    predictions = model(flat)
    q = predictions.new_zeros((n, depth, predictions.shape[-1]))
    q[active] = predictions
    # Float64 correction matches the native estimator; network stays float32.
    q = q.double() * batch["sign"][:, None, None]
    returns = batch["terminal"].double()
    advantage = None
    for step in range(depth - 1, -1, -1):
        value, current = corrected(q[:, step], batch["policy"][:, step], batch["mask"][:, step],
                                   batch["action"][:, step], batch["probability"][:, step], returns)
        returns = torch.where(active[:, step], value, returns)
        if step == 0:
            advantage = current
    return returns, advantage


def paired_variance_loss(advantages, legal_mask):
    if len(advantages) % 2 or not torch.equal(legal_mask[::2], legal_mask[1::2]):
        raise ValueError("Expected independent same-history continuation pairs")
    difference = advantages[::2] - advantages[1::2]
    return (0.5 * (difference.square() * legal_mask[::2]).sum(-1)
            / legal_mask[::2].sum(-1)).mean()


def root_td_loss(model, batch, weighted=False):
    prediction = model(batch["features"][:, 0]).gather(1, batch["action"][:, 0, None]).squeeze(1)
    errors = (prediction - batch["td_target"]).square()
    if weighted:
        # A uniform-action TD risk under the root sampling distribution. This is
        # a simple sampling-aware comparator, NOT a claim of variance optimality.
        weights = 1. / (batch["mask"][:, 0].sum(-1) * batch["probability"][:, 0])
        errors = errors * weights
    return errors.mean()
