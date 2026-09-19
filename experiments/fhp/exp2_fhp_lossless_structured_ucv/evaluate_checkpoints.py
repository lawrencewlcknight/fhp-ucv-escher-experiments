"""Evaluate all four FHP Experiment 2 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline import evaluate_checkpoints as _base
from fhp_escher.checkpointing import LoadedFHPPolicy

_ORIGINAL_RUN_EVALUATION = _base.run_evaluation


def run_evaluation(args):
    # The shared evaluator's public functions accept any OpenSpiel Policy, but
    # its generic checkpoint loader predates versioned feature encoders. Patch
    # only this call site by delegating the orchestration after installing our
    # encoder-aware loader in the imported module.
    _base._import_suite()
    from fhp_evaluation import loaders

    previous = loaders.load_checkpoint_policy
    loaders.load_checkpoint_policy = lambda game, path: LoadedFHPPolicy(game, path)
    try:
        return _ORIGINAL_RUN_EVALUATION(args)
    finally:
        loaders.load_checkpoint_policy = previous


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
