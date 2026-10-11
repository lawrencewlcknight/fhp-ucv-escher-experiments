"""Independent split panels of recorded, frozen-policy FHP continuations."""
import hashlib
import time

import numpy as np
import torch

from experiments.fhp.exp17_fhp_critic_budget.diagnostics import FrozenRollouts
from experiments.fhp.exp17_fhp_critic_budget.fitting import cpu_state


def frozen_sampler(solver):
    targets = [cpu_state(m.target_model) for m in solver.q_value_trainer.members]
    snapshots = [{u: dict(online=t, deployed=t) for u in (0, 1)} for t in targets]
    return FrozenRollouts(solver, snapshots, targets, [0, 1])


@torch.no_grad()
def record_trace(sampler, root, traverser, fold, rng, max_decisions):
    state, nodes, target = root.clone(), [], None
    while not state.is_terminal():
        if len(nodes) >= max_decisions:
            raise RuntimeError("FHP trace exceeded prespecified depth bound")
        policy, mask, proposal, _ = sampler.info(state, traverser, fold)
        action = int(rng.choice(len(proposal), p=proposal))
        child = sampler.chance(state.child(action), rng)
        nodes.append((sampler.solver.get_history_tensor(state), policy.copy(), mask.copy(),
                      action, float(proposal[action])))
        if target is None:
            target = child.returns()[0] / sampler.solver.max_utility
            if not child.is_terminal():
                next_policy = sampler.solver.regret_trainers[child.current_player()].get_policy(
                    child, sampler.solver.num_iteration)
                target += float(sampler.predictions(child, [sampler.old[1-fold]])[0] @ next_policy)
        state = child
    return nodes, state.returns()[traverser] / sampler.solver.max_utility, target


class Panel:
    def __init__(self, arrays, roots):
        self.arrays, self.roots = arrays, roots

    @property
    def trace_count(self):
        return len(self.arrays["terminal"])

    def batch(self, ids, root_only=False):
        ids = np.asarray(ids, dtype=np.int64)
        a = self.arrays
        starts, ends = a["offset"][ids], a["offset"][ids+1]
        lengths = ends-starts
        positions = np.arange(1 if root_only else int(lengths.max()))[None, :]
        active = positions < lengths[:, None]
        indices = np.minimum(starts[:, None] + positions, ends[:, None]-1)
        batch = {name: torch.from_numpy(a[name][indices].copy())
                 for name in ("features", "policy", "mask", "action", "probability")}
        batch["active"] = torch.from_numpy(active)
        # Harmless valid padding avoids division by zero, then recurrence skips it.
        for name in ("terminal", "sign", "td_target"):
            batch[name] = torch.from_numpy(a[name][ids].copy())
        return batch

    def save(self, path):
        np.savez_compressed(path, **self.arrays)


def generate_panel(sampler, config, *, seed, member, split, seen):
    split_index = ("fit", "selection", "assessment").index(split)
    stream = [config["random_seed"], seed, member, split_index]
    # Root selection and each continuation have distinct deterministic streams.
    root_rng = np.random.default_rng(np.random.SeedSequence(stream + [0]))
    count, repeats = config["split_roots"][split], config["split_repeats"][split]
    if count % 2 or repeats < 2 or repeats % 2:
        raise ValueError("Balanced roots and independent pairs require even counts")
    all_nodes, offsets, terminals, signs, targets, root_ids, roots = [], [0], [], [], [], [], []
    started = time.perf_counter()
    for index in range(count):
        traverser, fold = index % 2, 1-member
        for _ in range(1000):
            state = sampler.select_state(traverser, fold, root_rng)
            digest = hashlib.sha256(state.serialize().encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                break
        else:
            raise RuntimeError("Could not generate disjoint full-history roots")
        policy, mask, proposal, _ = sampler.info(state, traverser, fold)
        if np.any(proposal[mask > 0] <= 0):
            raise ValueError("Root traverser sampling must retain full support")
        roots.append(dict(history=list(state.history()), history_sha256=digest, traverser=traverser,
                          trajectory_fold=fold, member=member, policy=policy.tolist(),
                          proposal=proposal.tolist(), legal_actions=np.flatnonzero(mask).tolist()))
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence(stream + [1, index, repeat]))
            nodes, terminal, target = record_trace(sampler, state, traverser, fold, rng,
                                                    config["max_trace_decisions"])
            all_nodes.extend(nodes)
            offsets.append(len(all_nodes))
            terminals.append(terminal)
            targets.append(target)
            signs.append(1. if traverser == 0 else -1.)
            root_ids.append(index)
        if (index+1) % 16 == 0:
            print(f"panel {split} member={member} roots={index+1}/{count}", flush=True)
    features, policies, masks, actions, probabilities = zip(*all_nodes)
    arrays = dict(features=np.asarray(features, dtype=np.float32), policy=np.asarray(policies, dtype=np.float64),
                  mask=np.asarray(masks, dtype=np.float64), action=np.asarray(actions, dtype=np.int64),
                  probability=np.asarray(probabilities, dtype=np.float64), offset=np.asarray(offsets, dtype=np.int64),
                  terminal=np.asarray(terminals, dtype=np.float64), sign=np.asarray(signs, dtype=np.float64),
                  td_target=np.asarray(targets, dtype=np.float32), root_id=np.asarray(root_ids, dtype=np.int64))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite generated panel")
    metadata = dict(split=split, stream=stream, roots=roots, repeats=repeats,
                    trace_count=len(terminals), decision_rows=len(all_nodes),
                    generation_seconds=time.perf_counter()-started,
                    array_bytes=sum(a.nbytes for a in arrays.values()))
    return Panel(arrays, roots), metadata


@torch.no_grad()
def assess(model, panel, batch_size):
    from .estimator import recursive_estimates
    advantages, errors = [], []
    model.eval()
    for start in range(0, panel.trace_count, batch_size):
        batch = panel.batch(np.arange(start, min(start+batch_size, panel.trace_count)))
        _, values = recursive_estimates(model, batch)
        advantages.append(values.numpy())
        prediction = model(batch["features"][:, 0]).gather(1, batch["action"][:, 0, None]).squeeze(1)
        errors.append((prediction-batch["td_target"]).square().numpy())
    values, errors = np.concatenate(advantages), np.concatenate(errors)
    rows = []
    for index, root in enumerate(panel.roots):
        ids = panel.arrays["root_id"] == index
        sample, legal = values[ids], root["legal_actions"]
        covariance = np.atleast_2d(np.cov(sample[:, legal], rowvar=False, ddof=1))
        rows.append(dict(root_index=index, history_sha256=root["history_sha256"],
                         variance=float(np.trace(covariance)/len(legal)),
                         covariance=covariance.tolist(), mean_advantage=sample.mean(0).tolist(),
                         td_mse=float(errors[ids].mean()),
                         max_abs_advantage=float(np.max(np.abs(sample[:, legal])))))
    variances = np.asarray([r["variance"] for r in rows])
    summary = dict(mean_variance=float(variances.mean()), median_variance=float(np.median(variances)),
                   p90_variance=float(np.quantile(variances, .9)), p95_variance=float(np.quantile(variances, .95)),
                   p99_variance=float(np.quantile(variances, .99)), max_variance=float(variances.max()),
                   mean_td_mse=float(errors.mean()), max_abs_advantage=float(np.abs(values).max()),
                   max_centring_error=max(float(np.max(np.abs(values[panel.arrays["root_id"] == i]
                                                             @ np.asarray(root["policy"]))))
                                            for i, root in enumerate(panel.roots)))
    if not all(np.isfinite(v) for v in summary.values()):
        raise FloatingPointError("Nonfinite assessment")
    return dict(summary=summary, roots=rows)
