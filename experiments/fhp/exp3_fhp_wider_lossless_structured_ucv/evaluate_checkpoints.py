"""Evaluate all four FHP Experiment 3 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.fhp.exp2_fhp_lossless_structured_ucv.evaluate_checkpoints import (
    run_evaluation,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rule-deals", type=int, default=10_000)
    parser.add_argument("--lbr-deals", type=int, default=1_000)
    parser.add_argument("--lbr-rollouts", type=int, default=4_096)
    parser.add_argument("--crossplay-deals", type=int, default=50_000)
    parser.add_argument("--base-seed", type=int, default=2026)
    args = parser.parse_args(argv)
    for name in ("rule_deals", "lbr_deals", "lbr_rollouts", "crossplay_deals"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    print(run_evaluation(args))


if __name__ == "__main__":
    main()
