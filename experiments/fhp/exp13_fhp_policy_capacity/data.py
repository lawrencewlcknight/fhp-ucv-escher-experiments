"""Exact replay sufficient statistics and a deterministic group-disjoint split."""

import hashlib
import numpy as np

from experiments.fhp.exp4_fhp_average_policy_audit.data import (  # re-used, not reimplemented
    group_replay, group_diagnostics, load_source,
)


def digest_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def group_digest(groups):
    return digest_arrays(groups.features, groups.targets, groups.masks, groups.masses, groups.counts,
                         np.asarray([groups.rows], dtype=np.int64))


def split_groups(groups, config, seed):
    """Split canonical groups, never replay rows. Tiny strata remain mostly train."""
    if config["split_fractions"] != [.8, .1, .1]:
        raise ValueError("Frozen split requires 80/10/10 group proportions")
    if len(groups.features) < 10:
        raise ValueError("Need at least ten distinct groups")
    features = groups.features
    player = features[:, 104:106].argmax(1)
    round_id = features[:, 106:108].argmax(1)
    frequency = np.where(groups.counts == 1, 0, np.where(groups.counts <= 4, 1, 2))
    strata = player * 6 + round_id * 3 + frequency
    rng = np.random.default_rng(config["split_seed"] + seed)
    indices = {name: [] for name in ("training", "validation", "test")}
    for stratum in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == stratum))
        # At least one training example per stratum; singletons are not duplicated.
        count = max(1, int(round(len(ids) * .1))) if len(ids) >= 3 else 0
        indices["validation"].extend(ids[:count])
        indices["test"].extend(ids[count:2*count])
        indices["training"].extend(ids[2*count:])
    if not all(indices.values()):
        raise ValueError("Not enough groups for a nonempty stratified three-way split")
    indices = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in indices.items()}
    splits = {k: groups.subset(v) for k, v in indices.items()}
    metadata = {k: {**group_diagnostics(v), "indices_sha256": digest_arrays(indices[k]),
                    "data_sha256": group_digest(v), "group_fraction": len(indices[k])/len(features),
                    "replay_mass_fraction": float(v.masses.sum()/groups.masses.sum())}
                for k, v in splits.items()}
    return splits, metadata
