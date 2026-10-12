"""Exact retained-replay targets, coverage and board-disjoint refitting data."""
from dataclasses import dataclass
import numpy as np

from unbiased_escher.lossless_replay import CODEBOOK, encode
from .panels import board_classes, categories, feature_hash


def dense(features, rows):
    if isinstance(features, dict):
        if not np.array_equal(features["codebook"], CODEBOOK) or features["tail"].shape[1]:
            raise ValueError("Unsupported feature coding")
        return CODEBOOK[features["codes"][rows]]
    return np.asarray(features[rows], dtype=np.float32)


@dataclass
class Groups:
    codes: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    masses: np.ndarray
    counts: np.ndarray
    ess: np.ndarray
    first_iteration: np.ndarray
    last_iteration: np.ndarray
    iteration_bins: np.ndarray
    rows: int

    def features(self, ids):
        return CODEBOOK[self.codes[ids]]

    def subset(self, ids):
        return Groups(*(getattr(self,k)[ids] for k in (
            "codes", "targets", "masks", "masses", "counts", "ess",
            "first_iteration", "last_iteration", "iteration_bins")),
            rows=int(self.counts[ids].sum()))


def group_replay(buffer, iteration, gamma):
    size = int(buffer["size"])
    if size <= 0 or iteration <= 0:
        raise ValueError("Empty replay or invalid iteration")
    coded = np.empty((size, 213), dtype=np.uint8)
    for start in range(0,size,16384):
        coded[start:start+16384] = encode(dense(buffer["infostate"], slice(start,min(size,start+16384))))
    probabilities = np.asarray(buffer["q_value"][:size], dtype=np.float64)
    masks = np.asarray(buffer["q_value_mask"][:size], dtype=np.float32)
    times = np.asarray(buffer["iteration"][:size], dtype=np.float64).reshape(-1)
    if (probabilities.shape != (size,3) or masks.shape != (size,3)
            or not all(np.isfinite(x).all() for x in (probabilities,masks,times))
            or np.any(times <= 0) or np.any(times > iteration)
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1),1,atol=2e-6)
            or not np.isin(masks,(0,1)).all() or np.any(probabilities[masks == 0] != 0)):
        raise ValueError("Invalid historical policy replay")
    weights = (2*times/iteration)**gamma
    unique, first, inverse, counts = np.unique(coded, axis=0, return_index=True,
                                             return_inverse=True, return_counts=True)
    n = len(unique)
    masses = np.bincount(inverse,weights=weights,minlength=n)
    squared = np.bincount(inverse,weights=weights**2,minlength=n)
    targets = np.column_stack([np.bincount(inverse,weights=weights*probabilities[:,a],minlength=n)
                              for a in range(3)]) / masses[:,None]
    if not np.array_equal(masks, masks[first][inverse]):
        raise ValueError("Different masks within a canonical information state")
    lo,hi = np.full(n,np.inf),np.zeros(n)
    np.minimum.at(lo,inverse,times)
    np.maximum.at(hi,inverse,times)
    bins = np.minimum((4*times/iteration).astype(int),3)
    hist = np.column_stack([np.bincount(inverse,weights=(bins==b).astype(int),minlength=n) for b in range(4)])
    return Groups(unique,targets.astype(np.float32),masks[first],masses,counts,
                  masses**2/squared,lo,hi,hist.astype(np.int64),size)


def coverage(groups, panel, stream_rows):
    wanted = {r["feature_hash"]:r["id"] for r in panel["rows"]}
    exact, categories_count = {}, {}
    for start in range(0,len(groups.codes),8192):
        ids = np.arange(start,min(len(groups.codes),start+8192))
        features = groups.features(ids)
        cats = categories(features)
        for cat in np.unique(cats):
            selected = ids[cats==cat]
            row = categories_count.setdefault(str(int(cat)),dict(groups=0,rows=0,mass=0.))
            row["groups"] += len(selected)
            row["rows"] += int(groups.counts[selected].sum())
            row["mass"] += float(groups.masses[selected].sum())
        for i,f in zip(ids,features):
            h = feature_hash(f)
            if h in wanted:
                exact[wanted[h]] = dict(group_index=int(i),count=int(groups.counts[i]),
                    effective_rows=float(groups.ess[i]),weighted_mass=float(groups.masses[i]),
                    target=groups.targets[i].tolist(), first_iteration=int(groups.first_iteration[i]),
                    last_iteration=int(groups.last_iteration[i]),iteration_quartile_rows=groups.iteration_bins[i].tolist())
    return dict(exact=exact,categories=categories_count,retained_rows=groups.rows,
        reservoir_stream_rows=int(stream_rows),unique_groups=len(groups.codes),
        missing_exact_states=len(panel["rows"])-len(exact),
        note="No retained row is not proof of no historical visit; target is an empirical replay surrogate, not exact CFR average.")


def training_groups(groups, panel, config):
    """Remove ALL holdout-board rows; target only supported, sensible rare labels.

    Predicate and thresholds are prespecified, not fitted to assessment losses.
    Random controls match group count and total added replay mass, not merely
    the number of SGD steps. Broad quads/straight-flush strata include all legal
    action histories; the primary diagnostic panel remains terminal calls.
    """
    holdout = {tuple(b) for b in panel["holdout_board_classes"]}
    keep = np.empty(len(groups.codes),dtype=bool)
    for start in range(0,len(keep),8192):
        stop = min(len(keep),start+8192)
        keep[start:stop] = [b not in holdout for b in board_classes(groups.features(slice(start,stop)))]
    result = groups.subset(np.flatnonzero(keep))
    cats = np.empty(len(result.codes),dtype=int)
    for start in range(0,len(cats),8192):
        cats[start:start+8192] = categories(result.features(slice(start,start+8192)))
    targeted = ((cats >= 7) & (result.counts >= config["minimum_target_count"])
                & (result.ess >= config["minimum_target_ess"])
                & (result.targets[:,0] <= config["maximum_target_fold"]))
    # Only upweight states with folding legal, so no artificial success from
    # selecting states where folding is masked out already.
    targeted &= result.masks[:,0] == 1
    candidates = np.flatnonzero(~targeted)
    n = int(targeted.sum())
    if n > len(candidates):
        raise ValueError("Cannot form disjoint equal-count random control")
    return result,targeted,dict(excluded_holdout_rows=int(groups.counts[~keep].sum()),
        excluded_holdout_groups=int((~keep).sum()),eligible_target_groups=n,
        eligibility="category 7/8, fold legal, count>=2, ESS>=1.5, replay target fold<=.05",
        targeted_available=bool(n),assessment_absent_from_refitting=True)


def critic_coverage(payload):
    """Count categories for BOTH hands in recent full-state critic replay.

    Critic data are circular/recent; these counts are not policy-reservoir
    retention estimates and not independent samples or exact-state visit logs.
    """
    result=[]
    for member in payload["q_ensemble"]["members"]:
        b=member["buffer"]; counts=np.zeros((2,10),dtype=np.int64)
        for start in range(0,b["size"],8192):
            f=dense(b["history"],slice(start,min(b["size"],start+8192)))
            # Base full-state layout ends in two nine-category vectors.
            for p,offset in enumerate((245,254)):
                cat=np.where(f[:,offset:offset+9].sum(1)>0,np.argmax(f[:,offset:offset+9],1)+1,0)
                counts[p]+=np.bincount(cat,minlength=10)
        result.append(dict(rows=b["size"],player_category_counts=counts.tolist()))
    return dict(members=result,category_index="0=preflop/partial board; 1..9=high card through straight flush",
                scope="recent circular critic replay, not lifetime visits")
