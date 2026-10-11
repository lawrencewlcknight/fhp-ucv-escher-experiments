"""Matched native initialisations; candidates are prefixes, never sequential target refreshes."""
from copy import deepcopy
from collections import deque
import hashlib
import time

import numpy as np
import torch

from adaptive_escher.frozen_target_cache import build_target_cache
from experiments.critic_target_cache_benchmark import weight_hash
from experiments.fhp.exp17_fhp_critic_budget.fitting import cpu_state, averaged_target
from .estimator import recursive_estimates, paired_variance_loss, root_td_loss


def optimizer_hash(state):
    digest = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(repr(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple, deque)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(state)
    return digest.hexdigest()


def fit_objective(member, iteration, panel, objective, config, *, seed, fold, callback):
    started, overhead = time.perf_counter(), 0.
    model = deepcopy(member.model).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=member.learning_rate)
    optimizer.load_state_dict(deepcopy(member.optimizer.state_dict()))
    initial_hash = weight_hash(cpu_state(model))
    initial_optimizer_hash = optimizer_hash(optimizer.state_dict())
    rng = np.random.default_rng(np.random.SeedSequence([config["random_seed"], seed, fold, 91]))
    if objective == "replay_td" and config.get("native_replay_rng", False):
        rng.bit_generator.state = deepcopy(member.buffer.rng.bit_generator.state)
    history = deepcopy(member.target_history)
    initialization_seconds = time.perf_counter()-started
    minibatch_digest = hashlib.sha256()
    network_rows = root_traces = 0
    records = []
    # A zero budget is a genuine no-op: no target cache and, crucially, no
    # history append / deployment refresh. Diagnostic cloning is charged apart.
    if 0 in config["updates"]:
        snapshot_started = time.perf_counter()
        online, deployed = cpu_state(member.model), cpu_state(member.target_model)
        candidate = dict(objective=objective, updates=0, member=fold, online=online,
                         deployed=deployed, optimizer=deepcopy(optimizer.state_dict()),
                         sampler_rng=deepcopy(rng.bit_generator.state),
                         type="diagnostic_critic_not_resumable_solver")
        metrics = dict(objective=objective, updates=0, member=fold, loss=None, gradient_norm=None,
                       initial_online_sha256=initial_hash, initial_optimizer_sha256=initial_optimizer_hash,
                       online_sha256=weight_hash(online), deployed_sha256=weight_hash(deployed),
                       fit_prefix_seconds=0., target_cache_seconds=0.,
                       initialization_seconds=initialization_seconds,
                       minibatch_stream_sha256=minibatch_digest.hexdigest(),
                       cumulative_root_traces=0, cumulative_online_input_rows=0,
                       batch_root_traces=0, target_average_window=member.target_average_window)
        callback(candidate, metrics)
        duration = time.perf_counter()-snapshot_started
        metrics["snapshot_and_selection_seconds"] = duration
        overhead += duration
        records.append(metrics)
    cache_started = time.perf_counter()
    targets = build_target_cache(member, iteration) if objective == "replay_td" else None
    cache_seconds = time.perf_counter()-cache_started
    for step in range(1, max(config["updates"])+1):
        if objective == "replay_td":
            size = len(member.buffer)
            count = size if member.batch_size < 0 else member.batch_size
            if count > size:
                raise ValueError("Insufficient native replay for source minibatch")
            indices = np.arange(size) if count == size else rng.choice(size, count, replace=False)
            minibatch_digest.update(indices.astype("<i8").tobytes())
            root_traces += len(indices)
            network_rows += len(indices)
            features = torch.as_tensor(member.buffer.history_buf[indices], dtype=torch.float32)
            actions = torch.as_tensor(member.buffer.action_buf[indices], dtype=torch.int64)
            prediction = model(features).gather(1, actions[:, None]).squeeze(1)
            loss = member.loss_fn(prediction, targets[indices])
        else:
            # Exactly the same pairs/minibatch stream in all three matched arms.
            pairs = rng.integers(panel.trace_count//2, size=config["pair_batch_size"])
            ids = np.stack([2*pairs, 2*pairs+1], axis=1).reshape(-1)
            minibatch_digest.update(ids.astype("<i8").tobytes())
            root_traces += len(ids)
            batch = panel.batch(ids, root_only=objective != "regret_variance")
            network_rows += int(batch["active"].sum())
            if objective == "regret_variance":
                _, advantage = recursive_estimates(model, batch)
                loss = paired_variance_loss(advantage, batch["mask"][:, 0])
            elif objective in ("matched_td", "weighted_td"):
                loss = root_td_loss(model, batch, weighted=objective == "weighted_td")
            else:
                raise ValueError("Unknown objective")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {objective} loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), member.gradient_clip_norm,
                                              error_if_nonfinite=True)
        optimizer.step()
        if step % 500 == 0 or step in config["updates"]:
            print(f"fit seed={seed} member={fold} objective={objective} step={step} loss={loss.item():.6g}", flush=True)
        if step in config["updates"]:
            elapsed = time.perf_counter()-started-overhead
            snapshot_started = time.perf_counter()
            online = cpu_state(model)
            deployed = averaged_target(history, online, member.target_average_window)
            candidate = dict(objective=objective, updates=step, member=fold,
                             online=online, deployed=deployed, optimizer=deepcopy(optimizer.state_dict()),
                             sampler_rng=deepcopy(rng.bit_generator.state),
                             type="exp22_diagnostic_critic_not_resumable_solver")
            metrics = dict(objective=objective, updates=step, member=fold, loss=float(loss.item()),
                           gradient_norm=float(norm), initial_online_sha256=initial_hash,
                           initial_optimizer_sha256=initial_optimizer_hash,
                           online_sha256=weight_hash(online), deployed_sha256=weight_hash(deployed),
                           fit_prefix_seconds=elapsed, target_cache_seconds=cache_seconds,
                           initialization_seconds=initialization_seconds,
                           minibatch_stream_sha256=minibatch_digest.hexdigest(),
                           cumulative_root_traces=root_traces, cumulative_online_input_rows=network_rows,
                           batch_root_traces=(member.batch_size if objective == "replay_td"
                                              else 2*config["pair_batch_size"]),
                           target_average_window=member.target_average_window)
            callback(candidate, metrics)
            duration = time.perf_counter()-snapshot_started
            metrics["snapshot_and_selection_seconds"] = duration
            overhead += duration
            records.append(metrics)
    return records
