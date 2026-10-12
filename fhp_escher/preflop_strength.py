"""Versioned, exact FHP preflop equity distributions (not seven-card Hold'em).

Generate once with ``python -m fhp_escher.preflop_strength --output PATH``.
All possible flops and legal opponent holdings are enumerated, using public
board suit orbits and vectorized blockers. No RNG or learned policy is used.
"""
import argparse
from collections import Counter
from functools import lru_cache
from itertools import combinations
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .relative_strength import BoardTable, canonical_board

ASSET = Path(__file__).with_name("data") / "preflop_strength_v1.json"
FEATURE_NAMES = ("preflop_available", "mean_flop_equity", "std_flop_equity",
                 *(f"flop_equity_bin_{i}" for i in range(10)))


def starting_indices(hands):
    hands = np.asarray(hands)
    low, high = np.min(hands // 4, axis=1), np.max(hands // 4, axis=1)
    suited = hands[:, 0] % 4 == hands[:, 1] % 4
    return np.where(suited, high * 13 + low, low * 13 + high)


def generate_table():
    started = time.perf_counter()
    orbits = Counter(canonical_board(b)[0] for b in combinations(range(52), 3))
    hist = np.zeros((169, 10), dtype=np.int64)
    sums = np.zeros(169, dtype=np.int64)
    squares = np.zeros(169, dtype=np.int64)
    for i, (board, multiplicity) in enumerate(sorted(orbits.items())):
        table = BoardTable(board)
        index = starting_indices(table.hands)
        numerator = table.equity_twice_count.astype(np.int64)
        bins = np.minimum(numerator * 10 // 2162, 9)
        np.add.at(hist, (index, bins), multiplicity)
        np.add.at(sums, index, numerator * multiplicity)
        np.add.at(squares, index, numerator**2 * multiplicity)
        if i % 250 == 0:
            print(f"Exact preflop preparation: {i}/{len(orbits)} board orbits", flush=True)
    counts = hist.sum(axis=1)
    expected = np.asarray([19600 * (6 if a == b else 4 if a > b else 12)
                           for a in range(13) for b in range(13)])
    if not np.array_equal(counts, expected):
        raise RuntimeError("Incorrect suit-orbit weighting or starting-hand coverage")
    mean = sums / counts / 2162
    variance = np.maximum(0, squares / counts / 2162**2 - mean**2)
    rows = np.column_stack((np.ones(169), mean, np.sqrt(variance), hist / counts[:, None]))
    return {"schema": "exact_fhp_preflop_strength_v1", "features": list(FEATURE_NAMES),
            "definition": "Uniform flop given own cards; equity against 1081 uniform legal opponent hands on each flop",
            "bins": "[0,.1),...,[.9,1]; no turn/river", "board_orbits": len(orbits),
            "physical_boards": sum(orbits.values()), "sampled": False,
            "histogram_counts": hist.tolist(), "weighted_deal_counts": counts.tolist(),
            "equity_twice_count_sums": sums.tolist(), "equity_twice_count_square_sums": squares.tolist(),
            "rows": rows.astype(np.float32).tolist(),
            "generation_seconds": time.perf_counter() - started}


@lru_cache(maxsize=1)
def load_table():
    raw = ASSET.read_bytes()
    payload = json.loads(raw)
    if payload["schema"] != "exact_fhp_preflop_strength_v1" or payload["features"] != list(FEATURE_NAMES):
        raise ValueError("Incompatible preflop feature table")
    rows = np.asarray(payload["rows"], dtype=np.float32)
    if rows.shape != (169, len(FEATURE_NAMES)) or not np.isfinite(rows).all():
        raise ValueError("Invalid preflop table")
    rows.flags.writeable = False
    return rows, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace a versioned feature asset")
    table = generate_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2) + "\n")
    print(f"Wrote exact table in {table['generation_seconds']:.2f}s: {args.output}")


if __name__ == "__main__":
    main()
