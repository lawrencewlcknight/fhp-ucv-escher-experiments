"""CLI helpers for archived-state efficiency experiments (not new training)."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import numpy as np
import torch

from experiments.critic_target_cache_benchmark import (
    aggregate, benchmark, load_components, sha256, upload, write_json,
)


def source_paths(directory, spec, seed):
    directory = Path(directory)
    metadata = json.loads((directory / spec["manifest"]).read_text())
    if not (directory / "SUCCESS.json").is_file():
        raise ValueError("Source worker is not complete")
    if spec["game"] == "fhp":
        run = json.loads((directory / "run_manifest.json").read_text())
        if run["experiment_name"] != spec["source_experiment"] or int(run["seed"]) != seed:
            raise ValueError("Source experiment/seed differs")
        selected = [row for row in metadata if row["checkpoint_id"] == spec["checkpoint"]]
        if len(selected) != 1:
            raise ValueError("Missing or duplicated source endpoint")
        name, checksum = selected[0]["training_state_path"], selected[0]["training_state_sha256"]
    else:
        selected = [row for row in metadata["training_states"]
                    if row["filename"].endswith("_" + spec["checkpoint"] + ".pt")]
        if len(selected) != 1:
            raise ValueError("Missing or duplicated source endpoint")
        name, checksum = selected[0]["relative_path"], selected[0]["sha256"]
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Source path escapes its worker directory")
    return path, checksum


def fetch_source(bucket, run_id, index, directory, spec):
    seed = spec["seeds"][index]
    task = f"task_{index:03d}_{spec['source_algorithm']}_seed_{seed}"
    remote = f"{bucket.rstrip('/')}/{run_id}/workers/{task}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = {"SUCCESS.json", spec["manifest"]}
    if spec["game"] == "fhp":
        names.add("run_manifest.json")
    for name in sorted(names):
        subprocess.run(["gcloud", "storage", "cp", f"{remote}/{name}", str(directory/name)], check=True)
    path, _ = source_paths(directory, spec, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", f"{remote}/{path.relative_to(directory.resolve())}", str(path)], check=True)


def run_worker(source, output, index, spec, make_solver, *, smoke=False, make_smoke=None):
    torch.set_num_threads(1 if smoke and source is None else 8)
    seed = spec["seeds"][index]
    if source is None:
        if not smoke or make_smoke is None:
            raise ValueError("Production needs a saved source state")
        solver = make_smoke(seed)
        members, iteration = solver.q_value_trainer.members, solver.num_iteration
        config = {"synthetic_local_smoke": True}
        identity = {"synthetic": True}
    else:
        path, checksum = source_paths(source, spec, seed)
        if sha256(path) != checksum:
            raise ValueError("Source training-state checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload["type"] != spec["state_type"] or payload["seed"] != seed
                or payload["checkpoint_id"] != spec["checkpoint"] or payload["schema_version"] != 1):
            raise ValueError("Source type/seed/checkpoint/schema mismatch")
        for key, expected in {"q_ensemble_size": 2, "critic_target_average_window": 4,
                              "fixed_control_variate_beta": 1., "use_instantaneous_predictor": False,
                              "baseline_network_train_steps": 10000, "baseline_batch_size": 2048}.items():
            if payload["config"].get(key) != expected:
                raise ValueError(f"Source does not use the selected configuration: {key}")
        identity = {"training_state_sha256": checksum, "source_commit": payload["repository_commit"],
                    "source_checkpoint": payload["checkpoint_id"], "seed": seed}
        members, iteration, config = load_components(payload, make_solver)
        del payload
        gc.collect()
    if smoke:
        for member in members:
            member.train_steps = 4
    directory = Path(output) / "workers" / f"seed_{seed}"
    source_files = [Path(__file__), Path(__file__).with_name("critic_target_cache_benchmark.py"),
                    Path(__file__).parents[1] / "adaptive_escher/frozen_target_cache.py"]
    manifest = {
        "experiment": spec, "source": identity, "source_config": config,
        "smoke": smoke, "iteration": iteration, "threads": torch.get_num_threads(),
        "repetitions": 1 if smoke else 3,
        "atol": 1e-6, "rtol": 1e-5,
        "code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "measurement_scope": "frozen critic fitting, not a new policy training/evaluation run",
    }
    manifest = json.loads(json.dumps(manifest))
    benchmark(members, iteration, seed=seed, directory=directory, manifest=manifest,
              smoke=smoke, upload=lambda: upload(directory))


def main(spec, make_solver, make_smoke):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("worker", "smoke", "aggregate"):
        command = sub.add_parser(action)
        command.add_argument("--output-root", type=Path, required=True)
        if action != "aggregate":
            command.add_argument("--source-worker", type=Path)
            command.add_argument("--task-index", type=int, choices=range(3), default=0)
    fetch = sub.add_parser("fetch-source")
    for name in ("bucket", "source-run-id"):
        fetch.add_argument("--" + name, required=True)
    fetch.add_argument("--task-index", type=int, choices=range(3), required=True)
    fetch.add_argument("--directory", type=Path, required=True)
    sub.add_parser("contract")
    args = parser.parse_args()
    if args.action == "fetch-source":
        fetch_source(args.bucket, args.source_run_id, args.task_index, args.directory, spec)
    elif args.action == "contract":
        print(json.dumps(spec, indent=2))
    elif args.action == "aggregate":
        print(aggregate(args.output_root, spec["seeds"]))
    else:
        smoke = args.action == "smoke"
        run_worker(args.source_worker, args.output_root, args.task_index, spec,
                   make_solver, smoke=smoke, make_smoke=make_smoke)
        if smoke:
            print(aggregate(args.output_root, [spec["seeds"][args.task_index]], smoke=True))
