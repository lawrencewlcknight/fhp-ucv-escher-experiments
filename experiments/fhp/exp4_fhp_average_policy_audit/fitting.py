"""Matched-reset policy fits; both samplers preserve the original loss scale."""

import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch

from fhp_escher.features import POLICY_LAYOUT, StructuredFHPMLP
from fhp_escher.checkpointing import sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def new_model(config, seed):
    # SonnetLinear uses SciPy's truncated normal, hence NumPy's global RNG.
    # Seeding Torch alone would NOT match the initial networks across arms.
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = StructuredFHPMLP(POLICY_LAYOUT, config["network_layers"], 3,
                                     branch_width=config["branch_width"])
    finally:
        np.random.set_state(numpy_state)
    return model


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class GroupSampler:
    def __init__(self, groups, kind, seed, batch_size):
        self.groups, self.kind = groups, kind
        self.batch_size = min(batch_size, len(groups.features))
        self.uniform_rng = random.Random(seed)
        self.mass_rng = np.random.default_rng(seed)
        self.cdf = np.cumsum(groups.masses / groups.masses.sum())
        self.cdf[-1] = 1.0
        if kind not in ("uniform", "mass"):
            raise ValueError(kind)

    def sample(self):
        groups, size = self.groups, len(self.groups.features)
        if self.kind == "uniform":
            indices = (np.arange(size) if self.batch_size == size else
                       np.asarray(self.uniform_rng.sample(range(size), self.batch_size)))
            weights = groups.masses[indices] * (float(size) / groups.rows)
        else:
            # IID sampling WITH replacement is essential for these q(g) weights.
            indices = np.searchsorted(self.cdf, self.mass_rng.random(self.batch_size), side="right")
            weights = np.full(self.batch_size, groups.masses.sum() / groups.rows)
        return indices, weights.astype(np.float32)


def diagnostics(model, groups):
    """Bounded batches; aggregate fidelity by round, frequency and legal action."""
    count = len(groups.features)
    ce, kl, l1 = (np.empty(count) for _ in range(3))
    action_error = np.empty((count, 3))
    model.eval()
    with torch.no_grad():
        for start in range(0, count, 4096):
            sl = slice(start, start + 4096)
            targets = torch.from_numpy(groups.targets[sl])
            logits = model(torch.from_numpy(groups.features[sl]))
            logs = torch.log_softmax(logits.masked_fill(torch.from_numpy(groups.masks[sl]) != 1, -1e20), -1)
            ce[sl] = -(targets * logs).sum(1).numpy()
            entropy = -(targets * torch.log(targets.clamp_min(1e-30))).sum(1).numpy()
            kl[sl] = np.maximum(ce[sl] - entropy, 0)
            action_error[sl] = np.abs(logs.exp().numpy() - groups.targets[sl])
            l1[sl] = action_error[sl].sum(1)
    strata = {"all": np.ones(count, dtype=bool),
              "preflop": groups.features[:, 106] == 1,
              "flop": groups.features[:, 107] == 1,
              "singletons": groups.counts == 1,
              "frequency_2_4": (groups.counts >= 2) & (groups.counts <= 4),
              "frequency_5_plus": groups.counts >= 5}
    result = {}
    for name, selected in strata.items():
        if not selected.any():
            continue
        weights = groups.masses[selected]
        result[name] = {"groups": int(selected.sum()), "rows": int(groups.counts[selected].sum()),
                        "mass": float(weights.sum()),
                        "weighted_ce": float(np.average(ce[selected], weights=weights)),
                        "weighted_kl": float(np.average(kl[selected], weights=weights)),
                        "weighted_l1": float(np.average(l1[selected], weights=weights)),
                        "uniform_group_ce": float(ce[selected].mean())}
    for action in range(3):
        selected = groups.masks[:, action] == 1
        if selected.any():
            result[f"action_{action}"] = {
                "groups": int(selected.sum()),
                "weighted_absolute_probability_error": float(np.average(
                    action_error[selected, action], weights=groups.masses[selected]))}
    model.train()
    return result


def save_playable(template, model, path, arm, fitting_seconds, updates, init_seed):
    payload = dict(template)
    payload.update(experiment_id=4, experiment_name="exp4_fhp_average_policy_audit",
                   algorithm_id=f"frozen_policy_{arm}",
                   policy_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    payload["offline_fitting"] = {"arm": arm, "seconds": fitting_seconds, "updates": updates,
                                  "initialisation_seed": init_seed,
                                  "source_algorithm_id": template["algorithm_id"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def fit_path(groups, heldout, *, config, seed, sampler, directory, template=None, on_checkpoint=None):
    """Save metrics/policies only; an interrupted fitting path repeats from reset."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    init_seed = config["initialisation_seed_base"] + seed
    model = new_model(config, init_seed)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    draw = GroupSampler(groups, sampler, init_seed + 1000, config["batch_size"])
    completed, fitting_seconds, diagnostic_seconds = 0, 0.0, 0.0
    results_path = directory / "metrics.json"
    contract_path = directory / "fit_contract.json"
    fit_contract = {"config": config, "seed": seed, "sampler": sampler,
                    "initial_hash": initial_hash, "training_state_retention": "none"}
    if (directory / "resume.pt").exists():
        raise ValueError("Legacy fitting resume state found; use a new RUN_ID")
    if contract_path.exists() and json.loads(contract_path.read_text()) != fit_contract:
        raise ValueError("Refusing incompatible fitting outputs")
    write_json(contract_path, fit_contract)
    results = []
    if results_path.exists():
        saved = json.loads(results_path.read_text())
        if [r["updates"] for r in saved] == list(config["updates"]):
            valid = all(r["initial_state_sha256"] == initial_hash and r["sampler"] == sampler
                        for r in saved)
            if template is not None:
                valid = valid and all((directory / f"{r['updates']}.pkl").is_file()
                    and sha256_file(directory / f"{r['updates']}.pkl") == r.get("policy_sha256")
                    for r in saved)
            if valid:
                return saved
        # No optimizer state is retained: replay the same seeded nested fit,
        # rather than warm-starting from a policy with a different optimizer.
    features, targets, masks = (torch.from_numpy(a) for a in
                                (groups.features, groups.targets, groups.masks))
    for endpoint in config["updates"]:
        if endpoint <= completed:
            continue
        started = time.monotonic()
        for step in range(completed + 1, endpoint + 1):
            ids, weights = draw.sample()
            index = torch.from_numpy(ids)
            logits = model(features[index]).masked_fill(masks[index] != 1, -1e20)
            loss = (-(targets[index] * torch.log_softmax(logits, -1)).sum(1)
                    * torch.from_numpy(weights)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {sampler} step {step}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                print(f"{directory.name} {sampler}: {step}/{config['updates'][-1]}, loss={loss.item():.6g}", flush=True)
        fitting_seconds += time.monotonic() - started
        diag_start = time.monotonic()
        row = {"arm": f"{sampler}_{endpoint}", "seed": seed, "updates": endpoint,
               "sampler": sampler, "initial_state_sha256": initial_hash,
               "initialisation_seed": init_seed, "fitting_seconds": fitting_seconds,
               "processed_examples": endpoint * draw.batch_size,
               "training": diagnostics(model, groups),
               "validation": None if heldout is None else diagnostics(model, heldout)}
        diagnostic_seconds += time.monotonic() - diag_start
        row["diagnostic_seconds"] = diagnostic_seconds
        if template is not None:
            save_playable(template, model, directory / f"{endpoint}.pkl", row["arm"],
                          fitting_seconds, endpoint, init_seed)
            row["policy_sha256"] = sha256_file(directory / f"{endpoint}.pkl")
        results.append(row)
        write_json(results_path, results)
        if on_checkpoint is not None:
            on_checkpoint()
        completed = endpoint
    return results
