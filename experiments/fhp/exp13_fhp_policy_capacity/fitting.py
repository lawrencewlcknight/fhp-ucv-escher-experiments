"""Reset fits with independent sampling RNGs and reload-tested playable endpoints."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from experiments.fhp.exp4_fhp_average_policy_audit.fitting import (
    GroupSampler, diagnostics, new_model as new_structured_model, state_hash, write_json,
)
from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file
from fhp_escher.game import load_fhp_game
from .data import group_digest


def new_model(settings, seed):
    if "model_type" not in settings:
        return new_structured_model(settings, seed)
    from fhp_escher.card_policy import new_model as new_card_model
    return new_card_model(settings, seed)


def optimizer_for(model, recipe):
    if recipe["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"])
    # Decoupled decay of weight matrices only, not biases.
    weights, biases = [], []
    for parameter in model.parameters():
        (weights if parameter.ndim >= 2 else biases).append(parameter)
    return torch.optim.AdamW([
        {"params": weights, "weight_decay": recipe["weight_decay"]},
        {"params": biases, "weight_decay": 0.0},
    ], lr=recipe["learning_rate"])


def save_playable(template, model, path, metadata, probe_features, probe_masks):
    payload = deepcopy(template)
    payload["source_training_config"] = payload.pop("training_config", {})
    payload.update(
        experiment_id=metadata.get("experiment_id", 13),
        experiment_name=metadata.get("experiment_name", "exp13_fhp_policy_capacity"),
        algorithm_id=f"{metadata.get('policy_id_prefix', 'policy_capacity')}_{metadata['architecture']}",
        policy_model=model.checkpoint_metadata(), policy_network_layers=list(model.hidden_layers),
        policy_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        offline_fitting=metadata,
    )
    path = Path(path)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    restored = LoadedFHPPolicy(load_fhp_game(), path)
    if (restored.model.checkpoint_metadata() != model.checkpoint_metadata()
            or restored.feature_encoder.metadata() != template["feature_encoder"]
            or state_hash(restored.model) != state_hash(model)):
        raise ValueError("Saved policy fails reload equivalence")
    with torch.no_grad():
        inputs, masks = torch.from_numpy(probe_features), torch.from_numpy(probe_masks)
        expected = torch.softmax(model(inputs).masked_fill(masks != 1, -1e20), -1)
        observed = torch.softmax(restored.model(inputs).masked_fill(masks != 1, -1e20), -1)
    if not torch.equal(expected, observed):
        raise ValueError("Saved policy fails prediction equivalence")


def fit_path(groups, validation, *, config, seed, replicate, architecture, recipe,
             updates, directory, template, data_sha256, phase, on_checkpoint=None):
    """No replay/optimizer snapshots. An interrupted path repeats deterministically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    updates = sorted(set(updates))
    if not updates or updates[0] < 1:
        raise ValueError("Positive, increasing fitting endpoints required")
    if group_digest(groups) != data_sha256:
        raise ValueError("Training data checksum does not match the supplied identity")
    validation_hash = None if validation is None else group_digest(validation)
    init_seed = config["initialisation_seed_base"] + seed * 100 + replicate
    sample_seed = config["sampler_seed_base"] + seed * 100 + replicate
    model = new_model(config["architectures"][architecture], init_seed)
    if sum(p.numel() for p in model.parameters()) != config["architectures"][architecture]["parameters"]:
        raise ValueError("Unexpected architecture parameter count")
    initial_hash = state_hash(model)
    fit_contract = {"config": config, "seed": seed, "replicate": replicate,
                    "architecture": architecture, "recipe": recipe, "updates": updates,
                    "initial_state_sha256": initial_hash, "data_sha256": data_sha256,
                    "validation_data_sha256": validation_hash, "phase": phase}
    contract_path, metrics_path = directory / "fit_contract.json", directory / "metrics.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != fit_contract:
        raise ValueError("Incompatible fit identity; use a new RUN_ID")
    write_json(contract_path, fit_contract)
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        if ([r["updates"] for r in saved] == updates and all(
            r["initial_state_sha256"] == initial_hash
            and all(r[k] == fit_contract[k] for k in ("seed", "replicate", "architecture", "recipe",
                                                     "phase", "data_sha256", "validation_data_sha256"))
            and (directory / f"{r['updates']}.pkl").exists()
            and sha256_file(directory / f"{r['updates']}.pkl") == r["policy_sha256"]
            for r in saved
        )):
            return saved
    optimizer = optimizer_for(model, config["recipes"][recipe])
    draw = GroupSampler(groups, "uniform", sample_seed, config["batch_size"])
    features, targets, masks = map(torch.from_numpy, (groups.features, groups.targets, groups.masks))
    sampling_digest = hashlib.sha256()
    completed, fitting_seconds, diagnostic_seconds, results = 0, 0.0, 0.0, []
    for endpoint in updates:
        started = time.monotonic()
        for step in range(completed + 1, endpoint + 1):
            ids, weights = draw.sample()
            sampling_digest.update(ids.astype("<i8", copy=False).tobytes())
            index = torch.from_numpy(ids)
            logits = model(features[index]).masked_fill(masks[index] != 1, -1e20)
            loss = (-(targets[index] * torch.log_softmax(logits, -1)).sum(1)
                    * torch.from_numpy(weights)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {architecture}/{recipe}/{step}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                print(f"{phase}/{architecture}/{recipe}/rep{replicate}: {step}/{updates[-1]} loss={loss.item():.6g}", flush=True)
        fitting_seconds += time.monotonic() - started
        started = time.monotonic()
        row = {"seed": seed, "replicate": replicate, "architecture": architecture,
               "recipe": recipe, "phase": phase, "updates": endpoint,
               "initialisation_seed": init_seed, "sampler_seed": sample_seed,
               "initial_state_sha256": initial_hash, "sample_sequence_sha256": sampling_digest.hexdigest(),
               "data_sha256": data_sha256, "fitting_seconds": fitting_seconds,
               "validation_data_sha256": validation_hash,
               "processed_examples": endpoint * draw.batch_size,
               "training": diagnostics(model, groups),
               "validation": None if validation is None else diagnostics(model, validation)}
        # Test data is not an argument of this function and cannot enter selection.
        probe_ids = np.linspace(0, len(groups.features) - 1, min(64, len(groups.features)), dtype=int)
        save_playable(template, model, directory / f"{endpoint}.pkl", {
            **{k: row[k] for k in ("seed", "replicate", "architecture", "recipe", "phase", "updates",
                                    "initialisation_seed", "sampler_seed", "data_sha256")},
            "recipe_config": config["recipes"][recipe], "fitting_seconds": fitting_seconds,
            "source_algorithm_id": template["algorithm_id"],
            **{k: config[k] for k in ("experiment_id", "experiment_name", "policy_id_prefix") if k in config},
        }, groups.features[probe_ids], groups.masks[probe_ids])
        row["policy_sha256"] = sha256_file(directory / f"{endpoint}.pkl")
        diagnostic_seconds += time.monotonic() - started
        row["diagnostic_seconds"] = diagnostic_seconds
        results.append(row)
        write_json(metrics_path, results)
        if on_checkpoint:
            on_checkpoint()
        completed = endpoint
    return results


def fitting_benchmark(groups, config, *, warmup_steps=10, timed_steps=60):
    """Disposable production-batch fits, not playable candidates or an end-to-end ETA."""
    torch.set_num_threads(1)
    features, targets, masks = map(torch.from_numpy, (groups.features, groups.targets, groups.masks))
    results = {}
    for architecture, settings in config["architectures"].items():
        model = new_model(settings, config["initialisation_seed_base"])
        optimizer = optimizer_for(model, config["recipes"][config["control_recipe"]])
        sampler = GroupSampler(groups, "uniform", config["sampler_seed_base"], config["batch_size"])
        started = None
        for step in range(warmup_steps + timed_steps):
            if step == warmup_steps:
                started = time.perf_counter()
            ids, weights = sampler.sample()
            index = torch.from_numpy(ids)
            logits = model(features[index]).masked_fill(masks[index] != 1, -1e20)
            loss = (-(targets[index] * torch.log_softmax(logits, -1)).sum(1)
                    * torch.from_numpy(weights)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite production-batch benchmark loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - started
        results[architecture] = {"batch_size": sampler.batch_size, "timed_steps": timed_steps,
                                 "seconds_per_update": elapsed / timed_steps,
                                 "projected_60000_update_seconds": elapsed * 60000 / timed_steps}
    return {"architectures": results, "warmup_steps": warmup_steps, "torch_threads": torch.get_num_threads(),
            "note": "Short fitting-only projection; excludes grouping, diagnostics, downloads, gameplay and queueing. Not a job ETA."}


def inference_benchmark(policy_path, features):
    """CPU-only network timing; encoding is unchanged and excluded for both arms."""
    model = LoadedFHPPolicy(load_fhp_game(), policy_path).model
    inputs = torch.from_numpy(np.ascontiguousarray(features[:2048]))
    result = {}
    with torch.no_grad():
        for size in (1, len(inputs)):
            value = inputs[:size]
            for _ in range(10):
                model(value)
            seconds = []
            for _ in range(3):
                started = time.perf_counter()
                for _ in range(100):
                    model(value)
                seconds.append((time.perf_counter() - started) / (100 * size))
            result[f"batch_{size}_seconds_per_state"] = float(np.median(seconds))
    return result
