"""Offline matched reset fits; cloud smoke -> three source workers -> analysis."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from experiments.fhp.exp4_fhp_average_policy_audit.data import group_replay, group_diagnostics, split_groups
from experiments.fhp.exp13_fhp_policy_capacity.data import group_digest
from experiments.fhp.exp13_fhp_policy_capacity.fitting import fit_path, write_json, fitting_benchmark
from experiments.fhp.exp13_fhp_policy_capacity.run import sync_output, resources
from fhp_escher.checkpointing import sha256_file
from .config import contract, arm_name, fit_directory, SOURCE_ALGORITHM, SOURCE_EXPERIMENT, SOURCE_STATE_TYPE
from .data import fetch_source, load_source
from .evaluation import evaluate


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_worker(source, root, seed, *, smoke=False, include_lbr=False, evaluation_workers=8):
    config = contract(smoke, include_lbr)
    if seed not in config["seeds"] or not 1 <= evaluation_workers <= 8:
        raise ValueError("Unexpected seed or evaluation-worker count")
    root, source = Path(root).resolve(), Path(source).resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("Source replay must be outside the uploaded output tree")
    torch.set_num_threads(1)
    started = time.monotonic()
    directory = root / "workers" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    buffer, iteration, template, provenance = load_source(source, seed, smoke=smoke)
    repo = Path(__file__).resolve().parents[3]
    # Hash reused fitting/evaluation dependencies as well as the new experiment.
    paths = sorted((repo / "experiments/fhp").rglob("*.py"))
    paths += sorted((repo / "fhp_escher").rglob("*.py")) + sorted((repo / "fhp_evaluation").rglob("*.py"))
    paths += sorted((repo / "vr_deep_cfr").rglob("*.py"))
    manifest = {"config": config, "provenance": provenance,
                "audit_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "audit_source_sha256": hashlib.sha256(b"".join(
                    str(p.relative_to(repo)).encode() + p.read_bytes() for p in paths)).hexdigest(),
                "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Input/configuration/code/runtime mismatch; use a new RUN_ID")
    write_json(manifest_path, manifest)
    groups = group_replay(buffer, iteration, config["gamma"])
    del buffer
    gc.collect()
    full_info = group_diagnostics(groups)
    if smoke and not provenance["synthetic"]:
        benchmark = fitting_benchmark(groups, {**config, "batch_size": 2048, "control_recipe": "adam_003"})
        for row in benchmark["architectures"].values():
            row["projected_20000_update_seconds"] = row["seconds_per_update"] * 20000
        write_json(directory / "fitting_benchmark.json", benchmark)
        print("Fitting-only benchmark: " + json.dumps(benchmark), flush=True)
    if smoke and len(groups.features) > 128:
        groups = groups.subset(np.random.default_rng(902).choice(len(groups.features), 128, replace=False))
    training, heldout = split_groups(groups, config["heldout_group_fraction"], config["split_seed"] + seed)
    partitions = {"full": groups, "training": training, "heldout": heldout}
    replay = {"full_source": full_info, "splits": {
        name: {**group_diagnostics(g), "data_sha256": group_digest(g)} for name, g in partitions.items()}}
    replay_path = directory / "replay_diagnostics.json"
    if replay_path.exists() and json.loads(replay_path.read_text()) != replay:
        raise ValueError("Replay/split identity changed")
    write_json(replay_path, replay)
    archived = directory / "archived.pkl"
    # Preserve exact checkpoint bytes and their original identity.
    from .data import source_row
    from experiments.fhp.exp4_fhp_average_policy_audit.data import relative_path
    _, source_checkpoint = source_row(source)
    archived.write_bytes(relative_path(source, source_checkpoint["path"]).read_bytes())
    sync = lambda: sync_output(directory, config)
    rows, policies = [], {"archived": archived}
    for phase, data, validation in (("diagnostic", training, heldout), ("deployment", groups, None)):
        for replicate in config["replicates"]:
            matched = []
            for recipe in config["recipes"]:
                path = fit_directory(directory, phase, recipe, replicate)
                fits = fit_path(data, validation, config=config, seed=seed, replicate=replicate,
                    architecture="standard", recipe=recipe, updates=config["updates"],
                    directory=path, template=template, data_sha256=group_digest(data),
                    phase=phase, on_checkpoint=sync)
                matched.append(fits)
                rows.extend(fits)
                if phase == "deployment":
                    policies[arm_name(recipe, replicate)] = path / f"{config['updates'][-1]}.pkl"
            for a, b in zip(*matched, strict=True):
                for key in ("initial_state_sha256", "sample_sequence_sha256", "data_sha256",
                            "validation_data_sha256", "processed_examples", "updates"):
                    if a[key] != b[key]:
                        raise ValueError(f"Unmatched learning-rate arms: {key}")
    write_json(directory / "fit_metrics.json", rows)
    write_json(directory / "policy_index.json", {
        name: {"path": str(path.relative_to(directory)), "sha256": sha256_file(path)}
        for name, path in policies.items()})
    fit_elapsed = time.monotonic() - started
    del groups, training, heldout, partitions, template
    gc.collect()
    evaluate(policies, config, seed, directory / "evaluation_tasks", evaluation_workers, sync)
    resources(directory, "audit", started, fit_elapsed, evaluation_workers)
    sync()
    write_json(directory / "SUCCESS.json", {"seed": seed, "status": "complete", "smoke": smoke,
                                           "manifest_sha256": identity(manifest)})
    sync()
    return directory


def synthetic_source(directory, seed=0):
    """Small schema fixture using real information-state encodings, never evidence."""
    from experiments.fhp.exp4_fhp_average_policy_audit.run import synthetic_source as old_fixture
    directory = old_fixture(directory)
    policy_path, state_path = directory / "policy.pkl", directory / "state.pt"
    with policy_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload.update(seed=seed, experiment_id=9, experiment_name=SOURCE_EXPERIMENT, algorithm_id=SOURCE_ALGORITHM)
    with policy_path.open("wb") as handle:
        pickle.dump(payload, handle)
    state = torch.load(state_path, weights_only=False)
    state.update(type=SOURCE_STATE_TYPE, seed=seed)
    torch.save(state, state_path)
    write_json(directory / "run_manifest.json", {
        "experiment_id": 9, "experiment_name": SOURCE_EXPERIMENT, "algorithm_id": SOURCE_ALGORITHM,
        "seed": seed, "training_duration_seconds": 86400})
    rows = json.loads((directory / "checkpoint_manifest.json").read_text())
    rows[0].update(sha256=sha256_file(policy_path), training_state_sha256=sha256_file(state_path))
    write_json(directory / "checkpoint_manifest.json", rows)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("worker", "smoke", "aggregate"):
        command = sub.add_parser(action)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--include-lbr", action="store_true")
        if action != "aggregate":
            command.add_argument("--source-worker", type=Path)
            command.add_argument("--evaluation-workers", type=int, default=8)
        if action == "worker":
            command.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    fetch = sub.add_parser("fetch-source")
    fetch.add_argument("--bucket", required=True)
    fetch.add_argument("--run-id", required=True)
    fetch.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    fetch.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "fetch-source":
        fetch_source(args.bucket, args.run_id, args.seed, args.directory)
    elif args.action == "aggregate":
        from .aggregate import aggregate
        aggregate(args.output_root, include_lbr=args.include_lbr)
    elif args.action == "smoke":
        from .aggregate import aggregate
        with TemporaryDirectory(prefix="fhp-exp15-source-") as temporary:
            source = args.source_worker or synthetic_source(Path(temporary))
            run_worker(source, args.output_root, 0, smoke=True, include_lbr=args.include_lbr,
                       evaluation_workers=args.evaluation_workers)
            aggregate(args.output_root, smoke=True, include_lbr=args.include_lbr)
    else:
        if args.source_worker is None:
            parser.error("worker requires --source-worker")
        run_worker(args.source_worker, args.output_root, args.seed, include_lbr=args.include_lbr,
                   evaluation_workers=args.evaluation_workers)


if __name__ == "__main__":
    main()
