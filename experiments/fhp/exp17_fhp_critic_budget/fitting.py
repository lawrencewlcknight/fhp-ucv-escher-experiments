"""Observe the native fit without splitting it or refreshing its frozen TD target."""
from copy import deepcopy
import hashlib
import time

import numpy as np
import torch

from adaptive_escher.frozen_target_cache import frozen_targets, transition_batch
from experiments.critic_target_cache_benchmark import weight_hash


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def averaged_target(history, online, window):
    """Exactly the native parameter-average rule, with one new completed fit."""
    entries = (list(history) + [online])[-window:]
    return {k: (torch.stack([s[k] for s in entries]).mean(0)
                if torch.is_floating_point(v) else entries[-1][k]) for k, v in online.items()}


def replay_digest(buffer):
    digest = hashlib.sha256()
    for name in ("history", "next_history", "next_state", "reward", "action",
                 "next_legal_actions_mask", "next_player", "done"):
        array = getattr(buffer, name + "_buf")[:len(buffer)]
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    digest.update(str((buffer.cur_id, len(buffer))).encode())
    return digest.hexdigest()


def audit_member(member, iteration, *, updates, probe_rows, diagnostic_seed):
    short, full = updates
    if not 0 < short < full or member.train_steps != full or not member.cache_frozen_targets:
        raise ValueError("Expected native cached long fit and an interior short-fit checkpoint")
    if len(member.buffer) < max(1, member.batch_size):
        raise ValueError("Insufficient critic replay for the configured minibatch")
    initial = cpu_state(member.model)
    old_target = cpu_state(member.target_model)
    history = deepcopy(member.target_history)
    version = member.target_version
    replay_hash = replay_digest(member.buffer)
    ids = np.random.default_rng(diagnostic_seed).choice(len(member.buffer), min(probe_rows, len(member.buffer)), replace=False)
    probe = transition_batch(member.buffer, ids, member.device)
    labels = frozen_targets(member, probe, iteration)
    snapshots, prefix_times = {}, {}
    overhead = 0.0
    step = 0
    original = member.optimizer.step

    def observed_step(*args, **kwargs):
        nonlocal step, overhead
        result = original(*args, **kwargs)
        step += 1
        if step in updates:
            prefix_times[step] = time.perf_counter() - started - overhead
        if step == short:
            before = time.perf_counter()
            online = cpu_state(member.model)
            snapshots[short] = dict(online=online, deployed=averaged_target(history, online, member.target_average_window))
            overhead += time.perf_counter() - before
        return result

    member.optimizer.step = observed_step
    started = time.perf_counter()
    try:
        loss = member.train_model(iteration)
        elapsed = time.perf_counter() - started - overhead
    finally:
        member.optimizer.step = original
    if step != full or loss is None or not np.isfinite(loss) or member.target_version != version + 1:
        raise RuntimeError("Incomplete critic fit or more than one target refresh")
    online = cpu_state(member.model)
    snapshots[full] = dict(online=online, deployed=cpu_state(member.target_model))
    expected = averaged_target(history, online, member.target_average_window)
    if any(not torch.equal(expected[k], snapshots[full]["deployed"][k]) for k in expected):
        raise RuntimeError("Observed full target does not implement the native temporal average")
    if replay_digest(member.buffer) != replay_hash:
        raise RuntimeError("Replay changed during a frozen-data fit")
    # Evaluation uses detached copies: no mode, optimiser, target or RNG mutation.
    evaluator = deepcopy(member.model).eval()
    metrics = {}
    with torch.no_grad():
        for budget, state in snapshots.items():
            scores = {}
            for representation, weights in state.items():
                evaluator.load_state_dict(weights)
                q = evaluator(probe[0]).gather(1, probe[6].unsqueeze(1)).squeeze(1)
                scores[representation + "_training_probe_td_mse"] = float(torch.mean((q - labels)**2))
            metrics[str(budget)] = dict(**scores, prefix_seconds_including_cache=prefix_times[budget],
                estimated_native_fit_seconds=prefix_times[budget] + max(0., elapsed - prefix_times[full]),
                online_sha256=weight_hash(state["online"]), deployed_sha256=weight_hash(state["deployed"]))
    return dict(initial_online_sha256=weight_hash(initial), frozen_target_sha256=weight_hash(old_target),
                replay_sha256=replay_hash, replay_rows=len(member.buffer), training_probe_rows=len(ids),
                target_version_before=version, target_version_after=member.target_version,
                target_history_before=len(history), target_average_window=member.target_average_window,
                full_native_fit_seconds=elapsed, full_native_final_loss=float(loss), observation_overhead_seconds=overhead,
                timing_note="short fit estimated from native prefix plus full-fit finalisation; not a separately timed arm",
                cache=dict(member.last_target_cache_stats), arms=metrics), snapshots, old_target
