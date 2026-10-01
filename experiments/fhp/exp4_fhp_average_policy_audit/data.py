"""Validate archived inputs and extract only policy replay before grouping."""

from dataclasses import dataclass
import gc
import json
from pathlib import Path

import numpy as np
import torch

from fhp_escher.checkpointing import load_checkpoint_payload, sha256_file
from fhp_escher.features import ENCODER_ID, POLICY_LAYOUT, encoder_from_metadata
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.training_state import (
    read_sharded_training_state, training_state_storage_info,
)
from .config import SOURCE_ALGORITHM, SOURCE_EXPERIMENT


def relative_path(root, declared):
    root = Path(root).resolve()
    path = (root / declared).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input manifest path escapes its worker directory")
    return path


def source_row(worker):
    worker = Path(worker)
    manifest = json.loads((worker / "run_manifest.json").read_text())
    if (manifest.get("experiment_name") != SOURCE_EXPERIMENT
            or manifest.get("algorithm_id") != SOURCE_ALGORITHM):
        raise ValueError("Audit requires the selected Experiment 2 source")
    if not (worker / "SUCCESS.json").is_file():
        raise ValueError("Source worker is not complete")
    rows = json.loads((worker / "checkpoint_manifest.json").read_text())
    selected = [r for r in rows if r["checkpoint_id"] == "time_24h"]
    if len(selected) != 1 or selected[0]["checkpoint_target_seconds"] != 86400:
        raise ValueError("Source must contain exactly one 24-hour endpoint")
    return manifest, selected[0]


def load_source(worker, seed):
    manifest, row = source_row(worker)
    if int(manifest["seed"]) != seed:
        raise ValueError("Source seed mismatch")
    policy_path = relative_path(worker, row["path"])
    state_path = relative_path(worker, row["training_state_path"])
    if sha256_file(policy_path) != row["sha256"]:
        raise ValueError("Policy checksum mismatch")
    storage = training_state_storage_info(state_path, verify=True)
    if storage["sha256"] != row["training_state_sha256"]:
        raise ValueError("Training-state checksum mismatch")
    payload = load_checkpoint_payload(policy_path)
    state = (read_sharded_training_state(state_path, verify=False) if state_path.is_dir()
             else torch.load(state_path, map_location="cpu", weights_only=False))
    if (state.get("type") != "exp2_fhp_lossless_structured_full_training_state"
            or state.get("schema_version") != 1 or state.get("seed") != seed
            or state.get("checkpoint_id") != "time_24h"):
        raise ValueError("Unsupported source state/seed/checkpoint")
    if (payload["seed"] != seed or payload["experiment_id"] != 2
            or payload["algorithm_id"] != SOURCE_ALGORITHM
            or payload["outer_iteration"] != row["outer_iteration"]
            or state["solver"]["num_iteration"] != row["outer_iteration"]):
        raise ValueError("Policy and replay provenance disagree")
    model = payload["policy_model"]
    encoder_from_metadata(payload["feature_encoder"])
    if (payload["feature_encoder"]["id"] != ENCODER_ID
            or model["type"] != "structured_fhp_mlp_v1"
            or model["hidden_layers"] != [192, 192] or model["branch_width"] != 64):
        raise ValueError("Source does not use the frozen Experiment 2 policy architecture")
    if not payload.get("audit_synthetic_fixture"):
        original_config = state["config"]
        for key, expected in {"gamma": 2.0, "average_policy_learning_rate": .003,
                              "average_policy_train_steps": 20_000,
                              "average_policy_reset_each_fit": True,
                              "ave_policy_batch_size": 2048,
                              "structured_branch_width": 64}.items():
            if original_config.get(key) != expected:
                raise ValueError(f"Source policy-fitting contract differs: {key}")
        if state["solver"]["training_elapsed_seconds"] < 86400:
            raise ValueError("Source did not complete 24 active hours")
    for name, value in payload["policy_state_dict"].items():
        if not torch.equal(value, state["average_policy_trainer"]["model"][name]):
            raise ValueError("Source policy and training-state model differ")
    buffer = state["average_policy_trainer"]["buffer"]
    iteration = int(state["solver"]["num_iteration"])
    del state
    gc.collect()  # Release critic/calibration replay before allocating grouped targets.
    return buffer, iteration, payload, {
        "seed": seed, "source_worker": str(Path(worker).resolve()),
        "source_policy_sha256": row["sha256"],
        "source_state_sha256": storage["sha256"],
        "source_commit": manifest.get("repository_commit"),
        "outer_iteration": iteration, "nodes_touched": row["nodes_touched"],
        "checkpoint_id": row["checkpoint_id"],
    }


@dataclass
class Groups:
    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    masses: np.ndarray
    counts: np.ndarray
    rows: int

    def subset(self, indices):
        return Groups(*(getattr(self, name)[indices] for name in
                        ("features", "targets", "masks", "masses", "counts")),
                      rows=int(self.counts[indices].sum()))


def group_replay(buffer, iteration, gamma=2.0):
    size = int(buffer["size"])
    if size <= 0 or iteration <= 0:
        raise ValueError("Empty replay or invalid iteration")
    features = np.asarray(buffer["infostate"][:size], dtype=np.float32)
    targets = np.asarray(buffer["q_value"][:size], dtype=np.float64)
    masks = np.asarray(buffer["q_value_mask"][:size], dtype=np.float32)
    times = np.asarray(buffer["iteration"][:size], dtype=np.float64).reshape(-1)
    if (features.shape != (size, POLICY_LAYOUT.total_size)
            or targets.shape != (size, 3) or masks.shape != (size, 3)
            or not all(np.isfinite(a).all() for a in (features, targets, masks, times))
            or np.any(times < 0) or np.any(times > iteration)
            or np.any(targets < -1e-7) or not np.allclose(targets.sum(1), 1, atol=1e-5)
            or not np.isin(masks, (0, 1)).all() or np.any(masks.sum(1) == 0)
            or np.any(np.abs(targets[masks == 0]) > 1e-7)):
        raise ValueError("Invalid replay arrays/probabilities/iterations")
    weights = np.power(np.maximum(times, 0) / float(iteration) * 2, gamma)
    unique, first, inverse, counts = np.unique(
        features, axis=0, return_index=True, return_inverse=True, return_counts=True)
    masses = np.zeros(len(unique), dtype=np.float64)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(masses, inverse, weights)
    np.add.at(sums, inverse, targets * weights[:, None])
    grouped_masks = masks[first]
    for start in range(0, size, 65536):
        stop = start + 65536
        if not np.array_equal(masks[start:stop], grouped_masks[inverse[start:stop]]):
            raise ValueError("Masks disagree within an information set")
    if np.any(masses <= 0):
        raise ValueError("Grouped mass must be positive")
    return Groups(unique, (sums / masses[:, None]).astype(np.float32),
                  grouped_masks, masses, counts, size)


def split_groups(groups, fraction, seed):
    if len(groups.features) < 3:
        raise ValueError("Need at least three distinct groups for validation")
    order = np.random.default_rng(seed).permutation(len(groups.features))
    count = max(1, min(len(order) - 1, int(round(len(order) * fraction))))
    return groups.subset(order[count:]), groups.subset(order[:count])


def group_diagnostics(groups):
    mass = groups.masses
    q = mass / mass.sum()
    uniform_weights = mass * len(mass) / groups.rows
    return {
        "rows": groups.rows, "groups": len(mass),
        "rows_per_group_mean": groups.rows / len(mass),
        "singleton_group_fraction": float(np.mean(groups.counts == 1)),
        "group_count_quantiles": np.quantile(groups.counts, [0, .5, .9, .99, 1]).tolist(),
        "mass_quantiles": np.quantile(mass, [0, .5, .9, .99, 1]).tolist(),
        "top_one_percent_mass_fraction": float(np.sort(q)[-max(1, len(q)//100):].sum()),
        "uniform_importance_ess_fraction": float(1 / (len(q) * np.sum(q*q))),
        "uniform_loss_weight_mean": float(uniform_weights.mean()),
        "mass_sampler_loss_weight": float(mass.sum() / groups.rows),
        "mass_importance_ess_fraction": 1.0,
        "note": "ESS describes loss-weight concentration, not strategic coverage or exploitability",
    }
