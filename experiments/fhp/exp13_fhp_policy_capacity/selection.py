"""Global selection using validation only, before exposing any held-out test scores."""

import hashlib
import json
from pathlib import Path
import numpy as np

from .fitting import write_json


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit_directory(worker, phase, architecture, recipe, replicate):
    return Path(worker) / phase / architecture / recipe / f"rep{replicate}"


def screen_key(seed, architecture, recipe, replicate):
    return f"{seed}/{architecture}/{recipe}/rep{replicate}"


def screen_identity(directory):
    directory = Path(directory)
    fit = json.loads((directory / "fit_contract.json").read_text())
    rows = json.loads((directory / "metrics.json").read_text())
    return fit, rows, object_hash({"contract": fit, "metrics": rows})


def select(root, config):
    root = Path(root)
    candidates = {(architecture, recipe, updates): []
                  for architecture in config["architectures"] for recipe in config["recipes"]
                  for updates in config["screen_updates"]}
    source_hashes, screen_hashes, sample_hashes, init_hashes = {}, {}, {}, {}
    implementation = None
    for seed in config["seeds"]:
        worker = root / "workers" / f"seed_{seed}"
        manifest = json.loads((worker / "manifest.json").read_text())
        marker = json.loads((worker / "SCREEN_SUCCESS.json").read_text())
        if manifest["config"] != config or marker != {"seed": seed, "smoke": config["smoke"], "status": "screen_complete"}:
            raise ValueError("Incomplete or incompatible diagnostic worker")
        code = {k: manifest[k] for k in ("audit_commit", "audit_source_sha256", "python", "torch", "numpy")}
        if implementation is not None and code != implementation:
            raise ValueError("Source workers used different code or runtime versions")
        implementation = code
        source_hashes[str(seed)] = object_hash(manifest)
        replay = json.loads((worker / "replay_diagnostics.json").read_text())
        seen = set()
        for architecture in config["architectures"]:
            for recipe in config["recipes"]:
                for replicate in config["replicates"]:
                    directory = fit_directory(worker, "screen", architecture, recipe, replicate)
                    fit, rows, fit_hash = screen_identity(directory)
                    expected = {"config": config, "seed": seed, "replicate": replicate,
                                "architecture": architecture, "recipe": recipe, "phase": "screen",
                                "updates": config["screen_updates"],
                                "data_sha256": replay["splits"]["training"]["data_sha256"],
                                "validation_data_sha256": replay["splits"]["validation"]["data_sha256"]}
                    if any(fit.get(k) != v for k, v in expected.items()):
                        raise ValueError("Screen fit contract/data differs from frozen split")
                    screen_hashes[screen_key(seed, architecture, recipe, replicate)] = fit_hash
                    if [row["updates"] for row in rows] != config["screen_updates"]:
                        raise ValueError("Incomplete diagnostic endpoints")
                    for row in rows:
                        if (row["seed"], row["replicate"], row["architecture"], row["recipe"], row["phase"]) != (
                            seed, replicate, architecture, recipe, "screen"
                        ) or "test" in row:
                            raise ValueError("Invalid screen identity or premature test access")
                        if any(row.get(k) != fit[k] for k in ("data_sha256", "validation_data_sha256", "initial_state_sha256")):
                            raise ValueError("Screen metric data or initial weights differ from fit contract")
                        key = (seed, replicate, row["updates"])
                        if key in sample_hashes and sample_hashes[key] != row["sample_sequence_sha256"]:
                            raise ValueError("Capacity/recipe arms used different minibatch sequences")
                        sample_hashes[key] = row["sample_sequence_sha256"]
                        key = (seed, replicate, architecture)
                        if key in init_hashes and init_hashes[key] != row["initial_state_sha256"]:
                            raise ValueError("Recipe arms used different initial weights")
                        init_hashes[key] = row["initial_state_sha256"]
                        identity = (architecture, recipe, replicate, row["updates"])
                        if identity in seen:
                            raise ValueError("Duplicate screen fit")
                        seen.add(identity)
                        value = row["validation"]["all"]["weighted_ce"]
                        if not np.isfinite(value):
                            raise ValueError("Nonfinite validation loss")
                        candidates[(architecture, recipe, row["updates"])].append((seed, replicate, value))
    ranking = []
    for (architecture, recipe, updates), values in candidates.items():
        per_seed = {str(seed): float(np.mean([v for s, _, v in values if s == seed]))
                    for seed in config["seeds"]}
        ranking.append({"architecture": architecture, "recipe": recipe, "updates": updates,
                        "mean_validation_ce": float(np.mean(list(per_seed.values()))), "per_seed": per_seed})
    selected = {}
    recipe_order = list(config["recipes"])
    for architecture in config["architectures"]:
        rows = [row for row in ranking if row["architecture"] == architecture]
        best = min(rows, key=lambda r: (r["mean_validation_ce"], r["updates"], recipe_order.index(r["recipe"])))
        penultimate, final = config["screen_updates"][-2:]
        boundary = {}
        for recipe in config["recipes"]:
            losses = {r["updates"]: r["mean_validation_ce"] for r in rows if r["recipe"] == recipe}
            boundary[recipe] = losses[final] - losses[penultimate]
        selected[architecture] = {**best,
            "budget_boundary_warning": best["updates"] == final or any(v < 0 for v in boundary.values()),
            "boundary_validation_ce_change_by_recipe": boundary,
            "boundary_warning_note": "A selected final endpoint or any still-improving recipe leaves optimisation potentially unresolved; this is a descriptive warning, not a significance test."}
    result = {"config": config, "source_manifest_hashes": source_hashes,
              "screen_fit_hashes": screen_hashes,
              "selected": selected, "validation_ranking": ranking,
              "selection_uses_test_or_gameplay": False}
    result["selection_sha256"] = object_hash(result)
    path = root / "selection.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Selection is locked; incompatible inputs require a new RUN_ID")
    write_json(path, result)
    return result


def read_selection(path, config, manifest=None, seed=None):
    result = json.loads(Path(path).read_text())
    identity = {k: v for k, v in result.items() if k != "selection_sha256"}
    if (result["config"] != config or result["selection_sha256"] != object_hash(identity)
            or result["selection_uses_test_or_gameplay"]):
        raise ValueError("Invalid locked selection")
    if manifest is not None and result["source_manifest_hashes"][str(seed)] != object_hash(manifest):
        raise ValueError("Selected source/code differs from deployment worker")
    return result
