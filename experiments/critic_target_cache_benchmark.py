"""Paired frozen-state critic-cache benchmark shared by the two game experiments."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from adaptive_escher.frozen_target_cache import build_target_cache, frozen_targets, transition_batch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_comparison(left, right, atol=1e-6, rtol=1e-5):
    if set(left) != set(right):
        raise ValueError("State keys differ")
    maximum, passed, exact = 0., True, True
    for name in left:
        a, b = left[name].cpu(), right[name].cpu()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            passed = False
        maximum = max(maximum, float((a-b).abs().max()))
        passed = passed and torch.allclose(a, b, atol=atol, rtol=rtol)
        exact = exact and torch.equal(a, b)
    return {"max_absolute_difference": maximum, "allclose": bool(passed), "bitwise_equal": bool(exact)}


def snapshot(member):
    return {
        "model": deepcopy(member.model.state_dict()),
        "target": deepcopy(member.target_model.state_dict()),
        "optimizer": deepcopy(member.optimizer.state_dict()),
        "history": deepcopy(member.target_history),
        "version": member.target_version,
    }


def restore(member, saved, seed):
    member.model.load_state_dict(saved["model"])
    member.target_model.load_state_dict(saved["target"])
    member.optimizer.load_state_dict(deepcopy(saved["optimizer"]))
    member.target_history = deepcopy(saved["history"])
    member.target_version = saved["version"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(member.buffer, "rng"):
        member.buffer.rng = np.random.default_rng(seed)


def rng_hash(member):
    payload = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().numpy(),
               None if not hasattr(member.buffer, "rng") else member.buffer.rng.bit_generator.state)
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()


def target_audit(member, iteration, seed):
    # This validation work is outside both timed arms. Timed caching still pays
    # the full construction cost again, on every independent critic fit.
    before = rng_hash(member)
    cached = build_target_cache(member, iteration)
    count = min(len(member.buffer), member.batch_size if member.batch_size > 0 else len(member.buffer))
    indices = np.random.default_rng(seed).choice(len(member.buffer), count, replace=False)
    expected = frozen_targets(member, transition_batch(member.buffer, indices, member.device), iteration)
    result = tensor_comparison({"target": expected}, {"target": cached[indices]})
    result.update(rows_checked=count, cache_rows=len(cached), construction_preserved_rng=before == rng_hash(member))
    del cached
    return result


def load_components(payload, make_solver):
    """Restore only critics/replay and frozen regret models, not a second full learner."""
    config = deepcopy(payload["config"])
    small = deepcopy(config)
    for key in ("advantage_buffer_size", "ave_policy_buffer_size", "calibration_buffer_size"):
        small[key] = 1
    small["baseline_buffer_size"] = int(config["q_ensemble_size"])
    small["evaluate_initial_policy"] = False
    small["evaluation_frequency"] = 10**9
    small["early_evaluation_node_thresholds"] = ()
    solver = make_solver(int(payload["seed"]), small)
    if "feature_encoder" in payload and payload["feature_encoder"] != solver.feature_encoder.metadata():
        raise ValueError("Source feature encoding differs from the reconstructed solver")
    members = solver.q_value_trainer.members
    if len(members) != 2 or len(payload["q_ensemble"]["members"]) != 2:
        raise ValueError("Expected the selected two-critic configuration")
    for trainer, state in zip(solver.regret_trainers, payload["regret_trainers"]):
        trainer.model.load_state_dict(state["model"])
        if getattr(trainer, "predictor_enabled", False):
            raise ValueError("This experiment requires the selected non-predictive core")
    for member, state in zip(members, payload["q_ensemble"]["members"]):
        member.model.load_state_dict(state["model"])
        member.target_model.load_state_dict(state["target_model"])
        member.optimizer.load_state_dict(state["optimizer"])
        member.target_version = int(state["target_version"])
        member.target_history.clear()
        member.target_history.extend(deepcopy(state["target_history"]))
        replay = state["buffer"]
        for name in ("history", "next_history", "next_state", "reward", "action",
                     "next_legal_actions_mask", "next_player", "done"):
            setattr(member.buffer, name + "_buf", replay[name])
        member.buffer.size = int(replay["size"])
        member.buffer.buffer_size = len(replay["history"])
        member.buffer.cur_id = int(replay["cur_id"])
    return members, int(payload["solver"]["num_iteration"]), config


def benchmark(members, iteration, *, seed, directory, manifest, smoke=False, upload=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output belongs to another source/configuration/code revision")
    write_json(manifest_path, manifest)
    repeats = 1 if smoke else 3
    for repeat in range(repeats):
        pair_path = directory / f"repeat_{repeat}.json"
        if pair_path.exists():
            continue
        records = []
        for fold, member in enumerate(members):
            saved = snapshot(member)
            sampling_seed = seed * 100 + repeat * 2 + fold
            restore(member, saved, sampling_seed)
            audit = target_audit(member, iteration, sampling_seed)
            probe_ids = np.arange(min(4096, len(member.buffer)))
            probe = transition_batch(member.buffer, probe_ids, member.device)[0]
            outputs = {}
            order = ("recompute", "cache") if (repeat + seed) % 2 == 0 else ("cache", "recompute")
            for arm in order:
                restore(member, saved, sampling_seed)
                member.cache_frozen_targets = arm == "cache"
                # Warm numerical kernels equally without consuming sample RNG.
                warm = transition_batch(member.buffer, probe_ids[:min(len(probe_ids), member.batch_size)], member.device)
                with torch.no_grad():
                    member.model(warm[0])
                    frozen_targets(member, warm, iteration)
                started = time.perf_counter()
                loss = member.train_model(iteration)
                elapsed = time.perf_counter() - started
                if loss is None or not np.isfinite(loss):
                    raise ValueError("Critic fit did not produce a finite loss")
                with torch.no_grad():
                    predictions = member.model(probe).detach().clone()
                # Keep only small network/Adam states, never duplicate replay.
                outputs[arm] = {"state": snapshot(member), "predictions": predictions,
                                "seconds": elapsed, "loss": float(loss), "rng": rng_hash(member),
                                "cache": dict(getattr(member, "last_target_cache_stats", {})) if arm == "cache" else {}}
            a, b = outputs["recompute"], outputs["cache"]
            parameters = tensor_comparison(a["state"]["model"], b["state"]["model"])
            targets = tensor_comparison(a["state"]["target"], b["state"]["target"])
            prediction = tensor_comparison({"q": a["predictions"]}, {"q": b["predictions"]})
            adam_a = {f"{index}/{key}": value for index, entry in a["state"]["optimizer"]["state"].items()
                      for key, value in entry.items() if torch.is_tensor(value)}
            adam_b = {f"{index}/{key}": value for index, entry in b["state"]["optimizer"]["state"].items()
                      for key, value in entry.items() if torch.is_tensor(value)}
            optimizer = tensor_comparison(adam_a, adam_b)
            passed = (audit["allclose"] and audit["construction_preserved_rng"] and parameters["allclose"]
                      and targets["allclose"] and prediction["allclose"] and optimizer["allclose"]
                      and a["rng"] == b["rng"]
                      and a["state"]["version"] == b["state"]["version"] == saved["version"] + 1)
            row = {
                "seed": seed, "repeat": repeat, "fold": fold, "execution_order": list(order),
                "replay_rows": len(member.buffer), "batch_size": member.batch_size,
                "updates": member.train_steps, "cache_audit": audit,
                "recomputed_target_rows": member.train_steps * (
                    len(member.buffer) if member.batch_size == -1 else member.batch_size),
                "parameter_comparison": parameters, "target_snapshot_comparison": targets,
                "prediction_comparison": prediction, "adam_comparison": optimizer,
                "sample_rng_equal": a["rng"] == b["rng"], "equivalence_passed": bool(passed),
                "arms": {arm: {"fit_seconds": result["seconds"], "final_loss": result["loss"],
                               "model_sha256": weight_hash(result["state"]["model"]),
                               "sampler_rng_sha256": result["rng"], "cache_stats": result["cache"]}
                         for arm, result in outputs.items()},
            }
            records.append(row)
            restore(member, saved, sampling_seed)
        write_json(pair_path, {"records": records})
        if upload:
            upload()
    all_rows = [row for repeat in range(repeats)
                for row in json.loads((directory / f"repeat_{repeat}.json").read_text())["records"]]
    passed = all(row["equivalence_passed"] for row in all_rows)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    write_json(directory / "summary.json", {"seed": seed, "equivalence_passed": passed,
                                          "peak_process_rss_mib": peak, "records": all_rows})
    if smoke and not passed:
        raise RuntimeError("Smoke equivalence failed; production must not run")
    if upload:
        upload()
    write_json(directory / "SUCCESS.json", {"seed": seed, "equivalence_passed": passed})
    if upload:
        upload()


def aggregate(root, seeds, *, smoke=False):
    import csv
    from scipy.stats import t
    root = Path(root)
    per_seed, detailed, manifests = [], [], []
    for seed in seeds:
        folder = root / "workers" / f"seed_{seed}"
        if not (folder / "SUCCESS.json").exists():
            raise ValueError(f"Seed {seed} is incomplete")
        payload = json.loads((folder / "summary.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        manifests.append(manifest)
        if payload["seed"] != seed or manifest["smoke"] != smoke:
            raise ValueError("Wrong source seed or smoke/production mode")
        if not smoke and manifest["source"].get("seed") != seed:
            raise ValueError("Source identity does not match the result")
        for key in ("experiment", "code_sha256", "commit", "torch_version", "threads", "repetitions", "atol", "rtol"):
            if manifest[key] != manifests[0][key]:
                raise ValueError(f"Inconsistent benchmark manifests: {key}")
        rows = payload["records"]
        expected = {(repeat, fold) for repeat in range(1 if smoke else 3) for fold in (0, 1)}
        if {(r["repeat"], r["fold"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("Incomplete/duplicate repeat-fold measurements")
        if any(r["seed"] != seed for r in rows):
            raise ValueError("Result rows belong to another source seed")
        if not smoke and any(r["updates"] != 10000 or r["batch_size"] != 2048 for r in rows):
            raise ValueError("Production budget differs from the registered protocol")
        if payload["equivalence_passed"] != all(r["equivalence_passed"] for r in rows):
            raise ValueError("Summary contradicts per-fold equivalence checks")
        sums = {arm: [sum(r["arms"][arm]["fit_seconds"] for r in rows if r["repeat"] == repeat)
                      for repeat in range(1 if smoke else 3)] for arm in ("recompute", "cache")}
        baseline, cached = (float(np.median(sums[arm])) for arm in ("recompute", "cache"))
        per_seed.append({"seed": seed, "recompute_seconds": baseline, "cached_seconds": cached,
                         "speedup": baseline / cached, "time_reduction_fraction": 1-cached/baseline,
                         "equivalence_passed": payload["equivalence_passed"],
                         "peak_process_rss_mib": payload["peak_process_rss_mib"]})
        detailed.extend(rows)
    speeds = np.array([r["speedup"] for r in per_seed])
    std = float(speeds.std(ddof=1)) if len(speeds)>1 else None
    se = None if std is None else std/np.sqrt(len(speeds))
    margin = None if se is None else float(t.ppf(.975, len(speeds)-1))*se
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_json(analysis / "summary.json", {
        "smoke": smoke, "num_source_seeds": len(seeds), "per_seed": per_seed,
        "mean_speedup": float(speeds.mean()), "speedup_std": std, "speedup_se": se,
        "speedup_ci95": None if margin is None else [float(speeds.mean())-margin, float(speeds.mean())+margin],
        "all_equivalence_checks_passed": all(r["equivalence_passed"] for r in per_seed),
        "scope": "critic-fitting block only; not end-to-end training or policy-quality evidence",
    })
    write_json(analysis / "detailed_results.json", detailed)
    write_json(analysis / "source_manifests.json", manifests)
    write_json(analysis / "experiment_metadata.json", {
        "experiment": manifests[0]["experiment"], "seeds": list(seeds), "smoke": smoke,
        "measurement_scope": "frozen critic-fitting component benchmark",
        "output_convention_deviation": "No new policies, game traversals or exploitability metrics; source-seed paired timings instead.",
        "source_manifests": manifests,
    })
    with (analysis / "per_seed_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(seeds))
    axes[0].bar(x-.18, [r["recompute_seconds"] for r in per_seed], .36, label="Recompute")
    axes[0].bar(x+.18, [r["cached_seconds"] for r in per_seed], .36, label="Cache (including construction)")
    axes[0].set_ylabel("Seconds per two-critic fitting block")
    axes[0].legend()
    axes[1].bar(x, speeds)
    axes[1].axhline(1, color="black", linestyle="--")
    axes[1].set_ylabel("Recompute time / cached time")
    for ax in axes:
        ax.set_xticks(x, [str(seed) for seed in seeds])
        ax.set_xlabel("Source training seed")
    fig.suptitle("SMOKE ONLY" if smoke else "Frozen critic-target cache efficiency")
    fig.savefig(analysis / "critic_cache_efficiency.png", dpi=220)
    plt.close(fig)
    (analysis / "README.md").write_text(
        "# Critic-cache benchmark\n\n"
        "Medians of three repeated two-critic blocks within each source seed; "
        "the independent unit is the source seed, not fold or timing repeat. "
        "Cached timing includes fresh target construction every fit. "
        "Loading, restoration, validation and warm-up are excluded from both arms. "
        "Exact equality and numerical tolerance checks are separate. "
        "Any failed equivalence check prohibits promotion. "
        "This measures a component speedup, not overall training speedup, convergence or stronger play. "
        "No new policy was trained. Source states and original experiments are unchanged.\n")
    return analysis


def upload(directory):
    remote = os.environ.get("CRITIC_CACHE_REMOTE_WORKER")
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(directory), remote], check=True)
