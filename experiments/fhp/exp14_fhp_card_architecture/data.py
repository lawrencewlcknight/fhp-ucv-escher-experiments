"""Card-disjoint diagnostic split, never conflated with replay target grouping."""

import numpy as np
from experiments.fhp.exp13_fhp_policy_capacity.data import digest_arrays, group_digest, group_diagnostics


def card_keys(features):
    cards = features[:, :104]
    if features.shape[1] != 183 or not np.isin(cards, (0, 1)).all():
        raise ValueError("Expected the original lossless binary hole/public card channels")
    # 104 exact binary indicators -> 13 bytes; no hashing collisions or buckets.
    return np.packbits(cards.astype(np.uint8), axis=1)


def split_groups(groups, config, seed):
    if config["split_fractions"] != [.8, .1, .1]:
        raise ValueError("Card split is prespecified as 80/10/10")
    keys, first, inverse = np.unique(card_keys(groups.features), axis=0,
                                    return_index=True, return_inverse=True)
    if len(keys) < 10:
        raise ValueError("Need at least ten distinct visible card configurations")
    counts = np.bincount(inverse, weights=groups.counts).astype(np.int64)
    flop = groups.features[first, 107] == 1
    category = np.where(flop, groups.features[first, 174:183].argmax(1) + 1, 0)
    frequency = np.where(counts == 1, 0, np.where(counts <= 4, 1, 2))
    strata = category * 3 + frequency
    rng = np.random.default_rng(config["split_seed"] + seed)
    labels = np.zeros(len(keys), dtype=np.int8)
    for stratum in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == stratum))
        count = max(1, int(round(len(ids) * .1))) if len(ids) >= 3 else 0
        labels[ids[:count]], labels[ids[count:2*count]] = 1, 2
    splits, metadata = {}, {}
    for label, name in enumerate(("training", "validation", "test")):
        ids = np.flatnonzero(labels[inverse] == label)
        if not len(ids):
            raise ValueError("Too few card configurations for three nonempty partitions")
        split = groups.subset(ids)
        key_ids = np.flatnonzero(labels == label)
        splits[name] = split
        metadata[name] = {**group_diagnostics(split), "indices_sha256": digest_arrays(ids),
            "data_sha256": group_digest(split), "group_fraction": len(ids)/len(groups.features),
            "replay_mass_fraction": float(split.masses.sum()/groups.masses.sum()),
            "card_configurations": len(key_ids), "card_fraction": len(key_ids)/len(keys),
            "card_keys_sha256": digest_arrays(keys[key_ids]),
            "preflop_cards": int((~flop[key_ids]).sum()), "flop_cards": int(flop[key_ids].sum()),
            "player_group_counts": [int((split.features[:, 104+p] == 1).sum()) for p in (0, 1)]}
    return splits, metadata
