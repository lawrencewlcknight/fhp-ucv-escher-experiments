"""Evaluate all four FHP Experiment 1 checkpoints with the shared suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from fhp_escher.checkpointing import sha256_file
from fhp_escher.evaluation_adapter import _import_suite, evaluate_checkpoint
from fhp_escher.game import FHP_GAME_PARAMETERS, load_fhp_game


CHECKPOINTS = (
    ("time_06h", 6 * 60 * 60),
    ("time_12h", 12 * 60 * 60),
    ("time_18h", 18 * 60 * 60),
    ("time_24h", 24 * 60 * 60),
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _verified_checkpoints(source_run: Path) -> dict[str, Path]:
    rows = json.loads(
        (source_run / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if len(rows) != len(CHECKPOINTS):
        raise ValueError("Experiment 1 evaluation requires exactly four checkpoints")
    by_id = {str(row["checkpoint_id"]): row for row in rows}
    resolved = {}
    for checkpoint_id, target_seconds in CHECKPOINTS:
        row = by_id.get(checkpoint_id)
        if row is None or not np.isclose(
            float(row["checkpoint_target_seconds"]), float(target_seconds)
        ):
            raise ValueError(f"Missing or invalid {checkpoint_id}")
        path = source_run / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Missing or corrupt {checkpoint_id}: {path}")
        resolved[checkpoint_id] = path.resolve()
    return resolved


def run_evaluation(args) -> Path:
    _import_suite()
    from fhp_evaluation.duplicate import evaluate_cross_play
    from fhp_evaluation.loaders import load_checkpoint_policy

    source_run = args.source_run.resolve()
    run_manifest = json.loads(
        (source_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    checkpoints = _verified_checkpoints(source_run)
    output_dir = args.output_dir
    if output_dir is None:
        repository = Path(__file__).resolve().parents[3]
        output_dir = (
            repository
            / "outputs"
            / "evaluation"
            / str(run_manifest["experiment_name"])
            / source_run.name
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "started_utc": started.isoformat(),
        "source_run": str(source_run),
        "source_run_manifest_sha256": sha256_file(source_run / "run_manifest.json"),
        "experiment_id": int(run_manifest["experiment_id"]),
        "experiment_name": str(run_manifest["experiment_name"]),
        "seed": int(run_manifest["seed"]),
        "game": {"name": "FHP", "parameters": dict(FHP_GAME_PARAMETERS)},
        "evaluation": {
            "rule_deal_pairs_per_agent_per_checkpoint": int(args.rule_deals),
            "lbr_deal_pairs_per_checkpoint": int(args.lbr_deals),
            "lbr_preflop_rollout_samples": int(args.lbr_rollouts),
            "crossplay_deal_pairs_per_pair": int(args.crossplay_deals),
            "base_seed": int(args.base_seed),
            "exact_exploitability": False,
        },
        "checkpoints": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in checkpoints.items()
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)

    for label, _ in CHECKPOINTS:
        result = evaluate_checkpoint(
            checkpoints[label],
            num_deals=args.rule_deals,
            seed=args.base_seed,
            lbr_deals=args.lbr_deals,
            lbr_rollouts=args.lbr_rollouts,
        )
        _write_json(output_dir / label / "sampled_evaluation.json", result)

    game = load_fhp_game()
    policies = {
        label: load_checkpoint_policy(game, path)
        for label, path in checkpoints.items()
    }
    crossplay = [
        result.to_dict()
        for result in evaluate_cross_play(
            game,
            policies,
            num_deals=args.crossplay_deals,
            seed=args.base_seed + 2_000_000,
        )
    ]
    _write_json(
        output_dir / "checkpoint_crossplay.json",
        {
            "schema_version": 1,
            "interpretation": "positive mean favours policy_a",
            "results": crossplay,
        },
    )
    finished = datetime.now(timezone.utc)
    manifest.update(
        status="complete",
        finished_utc=finished.isoformat(),
        elapsed_seconds=(finished - started).total_seconds(),
        artifacts={
            "sampled_evaluations": {
                label: f"{label}/sampled_evaluation.json" for label, _ in CHECKPOINTS
            },
            "checkpoint_crossplay": "checkpoint_crossplay.json",
        },
    )
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    return output_dir


def main(argv=None) -> None:
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
