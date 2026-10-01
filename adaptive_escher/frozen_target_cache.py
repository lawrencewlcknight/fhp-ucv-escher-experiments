"""Opt-in, per-fit frozen TD target cache for persistent UCV critics.

No cache survives a train_model call. Regret predictors and the target critic
must remain fixed while the online critic takes its optimiser steps.
"""
from __future__ import annotations

import random
import time
import numpy as np
import torch


def replay_indices(buffer, count):
    """Preserve the two existing replay samplers, including full-batch RNG use."""
    size = len(buffer)
    if hasattr(buffer, "rng"):
        if not isinstance(buffer.rng, np.random.Generator):
            raise TypeError("Unsupported replay RNG")
        return slice(0, size) if count < 0 or count >= size else buffer.rng.choice(size, size=count, replace=False)
    return list(range(size)) if count == -1 else random.sample(range(size), count)


def transition_batch(buffer, indices, device):
    floats = ("history", "next_history", "next_state", "reward")
    integers = ("next_legal_actions_mask", "next_player", "action", "done")
    return tuple(torch.as_tensor(getattr(buffer, name + "_buf")[indices],
                                 dtype=dtype, device=device)
                 for names, dtype in ((floats, torch.float32), (integers, torch.int64))
                 for name in names)


def frozen_targets(member, batch, iteration):
    _, next_histories, next_states, rewards, masks, players, _, dones = batch
    with torch.no_grad():
        values = member.target_model(next_histories)
        strategies = member._batched_predictive_strategies(next_states, masks, players, int(iteration))
        return rewards + (1 - dones) * torch.sum(values * strategies, dim=1)


def build_target_cache(member, iteration):
    size = len(member.buffer)
    batch_size = size if member.batch_size == -1 else min(size, member.batch_size)
    if not size or batch_size <= 0:
        raise ValueError("Cannot cache an empty or invalid replay")
    for model in [member.target_model] + [
        model for trainer in member.regret_trainers
        for model in (trainer.model, getattr(trainer, "imm_model", None)) if model is not None
    ]:
        # Batch-dependent or stochastic layers invalidate row-wise reuse.
        for layer in model.modules():
            if isinstance(layer, (torch.nn.modules.batchnorm._BatchNorm,
                                  torch.nn.modules.dropout._DropoutNd)):
                raise ValueError("Target caching requires deterministic, row-independent networks")
    targets = torch.empty(size, dtype=torch.float32, device=member.device)
    for start in range(0, size, batch_size):
        stop = min(start + batch_size, size)
        indices = np.arange(start, stop)
        # Same forward batch shape as the baseline, including the final tail.
        # These networks are row-independent; padding consumes no RNG.
        if len(indices) < batch_size:
            indices = np.resize(indices, batch_size)
        batch = transition_batch(member.buffer, indices, member.device)
        targets[start:stop] = frozen_targets(member, batch, iteration)[:stop-start]
    if not torch.isfinite(targets).all():
        raise FloatingPointError("Nonfinite frozen critic targets")
    return targets


def train_with_cached_targets(member, iteration):
    if member.batch_size > 0 and len(member.buffer) < member.batch_size:
        return None
    if len(member.buffer) == 0 or member.train_steps <= 0:
        return None
    member.model.train()
    started = time.perf_counter()
    targets = build_target_cache(member, iteration)
    cache_seconds = time.perf_counter() - started
    loss = None
    for step in range(member.train_steps):
        indices = replay_indices(member.buffer, member.batch_size)
        histories = torch.as_tensor(member.buffer.history_buf[indices],
                                    dtype=torch.float32, device=member.device)
        actions = torch.as_tensor(member.buffer.action_buf[indices],
                                  dtype=torch.int64, device=member.device)
        target = targets[indices]
        q_value = member.model(histories).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = member.loss_fn(q_value, target)
        member.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.model.parameters(), member.gradient_clip_norm)
        member.optimizer.step()
        if step % 100 == 0:
            member.logger.info(f"train_step[{step}/{member.train_steps}]: persistent Q loss {loss.item()}")
    member.target_model.load_state_dict(member.model.state_dict())
    member.target_model.eval()
    member.target_version += 1
    member.last_target_cache_stats = {
        "cache_build_seconds": cache_seconds,
        "cache_bytes": targets.numel() * targets.element_size(),
        "cached_rows": len(member.buffer),
        "cache_lifetime": "this_fit_only",
    }
    # The caller's four-fit temporal averaging still runs after this returns.
    return float(loss.item())
