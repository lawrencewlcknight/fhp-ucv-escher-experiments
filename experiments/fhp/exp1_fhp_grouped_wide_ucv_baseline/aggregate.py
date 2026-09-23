"""Validate and aggregate the three independent Experiment 1 seed workers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from fhp_escher.checkpointing import sha256_file

from .config import (
    ALGORITHM_ID,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PRODUCTION_SEEDS,
    contract_manifest,
)
from .training_state import training_state_storage_info


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_name(index: int, seed: int) -> str:
    return f"task_{index:03d}_{ALGORITHM_ID}_seed_{seed}"


def aggregate_workers(
    output_root: Path,
    *,
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> Path:
    output_root = Path(output_root).resolve()
    workers_root = output_root / "workers"
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    checkpoints = []
    worker_artifacts = []
    for index, seed in enumerate(seeds):
        worker_dir = workers_root / task_name(index, int(seed))
        success_path = worker_dir / "SUCCESS.json"
        summary_path = worker_dir / "summary.json"
        manifest_path = worker_dir / "checkpoint_manifest.json"
        if not success_path.is_file() or not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete Experiment 1 worker: {worker_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "complete"
            or int(summary.get("seed", -1)) != int(seed)
            or int(summary.get("checkpoint_count", -1)) != 4
        ):
            raise ValueError(f"Invalid worker summary: {summary_path}")
        if len(rows) != 4:
            raise ValueError(f"Worker {seed} does not contain four checkpoints")
        continuation_rows = [row for row in rows if row.get("training_state_path")]
        if (
            len(continuation_rows) != 1
            or continuation_rows[0].get("checkpoint_id")
            != rows[-1].get("checkpoint_id")
        ):
            raise ValueError(
                f"Worker {seed} must retain only its latest continuation state"
            )
        for row in rows:
            checkpoint = worker_dir / row["path"]
            if sha256_file(checkpoint) != row["sha256"]:
                raise ValueError(f"Policy hash mismatch: {checkpoint}")
            # Cloud aggregation deliberately omits multi-gigabyte continuation
            # states. Verify them when present in a full local download; their
            # declared digest remains in the consolidated index either way.
            if row.get("training_state_path"):
                state = worker_dir / row["training_state_path"]
                if state.exists():
                    storage = training_state_storage_info(state, verify=True)
                    if storage["sha256"] != row["training_state_sha256"]:
                        raise ValueError(f"Training-state hash mismatch: {state}")
            checkpoints.append(
                {
                    "seed": int(seed),
                    "worker": worker_dir.name,
                    **dict(row),
                }
            )
        summaries.append(summary)
        worker_artifacts.append(
            {
                "seed": int(seed),
                "worker": worker_dir.name,
                "success_sha256": sha256_file(success_path),
                "summary_sha256": sha256_file(summary_path),
                "run_manifest_sha256": sha256_file(worker_dir / "run_manifest.json"),
                "checkpoint_manifest_sha256": sha256_file(manifest_path),
            }
        )

    _write_csv(analysis_dir / "seed_summaries.csv", summaries)
    _write_csv(analysis_dir / "checkpoint_index.csv", checkpoints)
    _write_json(
        analysis_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": ALGORITHM_ID,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "contract": contract_manifest(),
            "workers": worker_artifacts,
            "artifacts": {
                "seed_summaries": "seed_summaries.csv",
                "checkpoint_index": "checkpoint_index.csv",
            },
        },
    )
    _write_json(analysis_dir / "SUCCESS.json", {"status": "complete"})
    return analysis_dir


__all__ = ["aggregate_workers", "task_name"]
