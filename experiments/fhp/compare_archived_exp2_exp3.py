"""Validate and compare archived Experiment 2 and Experiment 3 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_map(run_dir: Path) -> dict[float, dict]:
    rows = _read_json(run_dir / "checkpoint_rows.json")
    return {float(row["checkpoint_target_seconds"]): row for row in rows}


def compare_runs(sequential_run: Path, parallel_run: Path) -> dict:
    sequential_run = Path(sequential_run)
    parallel_run = Path(parallel_run)
    sequential_manifest = _read_json(sequential_run / "run_manifest.json")
    parallel_manifest = _read_json(parallel_run / "run_manifest.json")
    sequential_summary = _read_json(sequential_run / "summary.json")
    parallel_summary = _read_json(parallel_run / "summary.json")

    expected = (
        (sequential_manifest, 2, "sequential"),
        (parallel_manifest, 3, "ray_parallel"),
    )
    for manifest, experiment_id, backend in expected:
        if int(manifest["experiment_id"]) != experiment_id:
            raise ValueError(f"Expected Experiment {experiment_id} output")
        if manifest["execution_backend"] != backend:
            raise ValueError(f"Expected {backend!r} execution backend")
    matched_fields = (
        "seed",
        "checkpoint_training_seconds",
        "training_duration_seconds",
        "game",
        "training_config",
        "training_config_sha256",
    )
    for field in matched_fields:
        if sequential_manifest[field] != parallel_manifest[field]:
            raise ValueError(f"Comparison arms differ in {field!r}")

    sequential_checkpoints = _checkpoint_map(sequential_run)
    parallel_checkpoints = _checkpoint_map(parallel_run)
    if sequential_checkpoints.keys() != parallel_checkpoints.keys():
        raise ValueError("Comparison arms have different checkpoint targets")

    rows = []
    for target_seconds in sorted(sequential_checkpoints):
        sequential = sequential_checkpoints[target_seconds]
        parallel = parallel_checkpoints[target_seconds]
        sequential_nodes = int(sequential["nodes_touched"])
        parallel_nodes = int(parallel["nodes_touched"])
        rows.append(
            {
                "checkpoint_target_seconds": target_seconds,
                "checkpoint_target_hours": target_seconds / 3600.0,
                "sequential_nodes_touched": sequential_nodes,
                "parallel_nodes_touched": parallel_nodes,
                "parallel_over_sequential_nodes_ratio": (
                    parallel_nodes / sequential_nodes
                    if sequential_nodes > 0
                    else None
                ),
                "sequential_episodes": int(sequential["episode"]),
                "parallel_episodes": int(parallel["episode"]),
                "sequential_outer_iteration": int(sequential["iteration"]),
                "parallel_outer_iteration": int(parallel["iteration"]),
                "sequential_training_elapsed_seconds": float(
                    sequential["training_elapsed_seconds"]
                ),
                "parallel_training_elapsed_seconds": float(
                    parallel["training_elapsed_seconds"]
                ),
                "sequential_average_policy_loss": sequential.get(
                    "average_policy_loss"
                ),
                "parallel_average_policy_loss": parallel.get(
                    "average_policy_loss"
                ),
            }
        )

    return {
        "comparison": "archived_exp2_sequential_vs_exp3_ray_parallel",
        "seed": int(sequential_manifest["seed"]),
        "game": sequential_manifest["game"],
        "checkpoint_training_seconds": sequential_manifest[
            "checkpoint_training_seconds"
        ],
        "sequential_run": str(sequential_run.resolve()),
        "parallel_run": str(parallel_run.resolve()),
        "sequential_final_checkpoint": sequential_summary[
            "final_policy_checkpoint"
        ],
        "parallel_final_checkpoint": parallel_summary["final_policy_checkpoint"],
        "rows": rows,
        "interpretation": (
            "Node/episode throughput is directly comparable at matched training "
            "times. Policy quality requires sampled evaluation of the saved "
            "checkpoints; no exact FHP exploitability claim is made."
        ),
    }


def write_comparison(comparison: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "archived_exp2_exp3_comparison.json"
    csv_path = output_dir / "archived_exp2_exp3_checkpoint_comparison.csv"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    rows = comparison["rows"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-run", type=Path, required=True)
    parser.add_argument("--parallel-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    comparison = compare_runs(args.sequential_run, args.parallel_run)
    print(write_comparison(comparison, args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
