"""CLI for the frozen FHP average-policy audit and its cloud input staging."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fhp_escher.checkpointing import CHECKPOINT_TYPE, CHECKPOINT_VERSION, sha256_file
from fhp_escher.features import FHPFeatureEncoder
from fhp_escher.game import load_fhp_game, serialisable_game_definition
from .aggregate import aggregate
from .config import SEEDS, SOURCE_ALGORITHM, SOURCE_EXPERIMENT, contract, source_worker_name
from .data import group_diagnostics, group_replay, load_source, relative_path, source_row, split_groups
from .evaluation import evaluate
from .fitting import fit_path, new_model, write_json


def sync_output(directory):
    remote = os.environ.get("EXP4_REMOTE_WORKER")
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive",
                        "--exclude", r"\.pt$|\.tmp$", str(directory), remote], check=True)


def run_worker(source, root, seed, *, smoke=False, evaluation_workers=8):
    if seed not in SEEDS or evaluation_workers < 1 or evaluation_workers > 8:
        raise ValueError("Seed must be 0, 1 or 2; evaluation workers must be 1..8")
    config = contract(smoke)
    torch.set_num_threads(1)
    directory = Path(root) / "workers" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    buffer, iteration, template, provenance = load_source(source, seed)
    if template.get("audit_synthetic_fixture") and not smoke:
        raise ValueError("Synthetic smoke inputs cannot be used for production")
    manifest = {"config": config, "provenance": provenance,
                "audit_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "audit_source_sha256": hashlib.sha256(b"".join(
                    path.name.encode() + path.read_bytes()
                    for path in sorted(Path(__file__).parent.glob("*.py")))).hexdigest(),
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        # Cloud inputs can be relocated, but data/config/code identities must match.
        for candidate in (old, manifest):
            candidate["provenance"].pop("source_worker", None)
        if old != manifest:
            raise ValueError("Output directory belongs to a different input/configuration/code; use a new RUN_ID")
        if (directory / "SUCCESS.json").exists():
            print(f"Seed {seed} already complete", flush=True)
            return
    write_json(manifest_path, manifest)
    started = time.monotonic()
    groups = group_replay(buffer, iteration, config["gamma"])
    del buffer
    gc.collect()
    # A real-source cloud smoke verifies the entire million-row extraction and
    # grouping path, then uses a bounded group subset for its tiny optimiser test.
    replay_info = group_diagnostics(groups)
    if smoke and len(groups.features) > 128:
        ids = np.random.default_rng(902).choice(len(groups.features), 128, replace=False)
        groups = groups.subset(ids)
    train, heldout = split_groups(groups, config["heldout_group_fraction"], config["split_seed"] + seed)
    write_json(directory / "replay_diagnostics.json", {
        "full_source": replay_info, "fit_groups": group_diagnostics(groups),
        "diagnostic_train": group_diagnostics(train), "diagnostic_heldout": group_diagnostics(heldout),
        "grouping_seconds": time.monotonic() - started})
    archived = directory / "archived.pkl"
    with archived.open("wb") as handle:
        pickle.dump(template, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sync = lambda: sync_output(directory)
    for sampler in config["samplers"]:
        fit_path(train, heldout, config=config, seed=seed, sampler=sampler,
                 directory=directory / "diagnostic" / sampler, on_checkpoint=sync)
        fit_path(groups, None, config=config, seed=seed, sampler=sampler,
                 directory=directory / "deployment" / sampler, template=template, on_checkpoint=sync)
    fit_elapsed = time.monotonic() - started
    del groups, train, heldout, template
    gc.collect()
    policies = {"archived": archived}
    for sampler in config["samplers"]:
        for updates in config["updates"]:
            policies[f"{sampler}_{updates}"] = directory / "deployment" / sampler / f"{updates}.pkl"
    evaluate(policies, config, seed, directory / "evaluation_tasks", evaluation_workers, sync)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(directory / "resources.json", {
        "fit_and_group_wall_seconds_this_attempt": fit_elapsed,
        "total_wall_seconds_this_attempt": time.monotonic() - started,
        "main_process_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
        "note": "Process peak excludes spawned evaluation children; Batch resource snapshots cover the VM.",
        "evaluation_workers": evaluation_workers})
    # Publish data before the completion marker; a partial upload is never complete.
    sync()
    write_json(directory / "SUCCESS.json", {"seed": seed, "status": "complete", "smoke": smoke})
    sync()


def fetch_source(bucket, run_id, seed, directory):
    """Only the selected 24h state, policy and three tiny manifests; never 6/12/18h states."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    remote = f"{bucket.rstrip('/')}/{run_id}/workers/{source_worker_name(seed)}"
    for name in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"):
        subprocess.run(["gcloud", "storage", "cp", f"{remote}/{name}", str(directory / name)], check=True)
    _, row = source_row(directory)
    for key in ("path", "training_state_path"):
        target = relative_path(directory, row[key])
        target.parent.mkdir(parents=True, exist_ok=True)
        # Archived Experiment 2 uses monolithic .pt. Refuse an unexpected layout
        # rather than silently downloading an entire training-state tree.
        if key == "training_state_path" and not str(row[key]).endswith(".pt"):
            raise ValueError("Cloud staging expects the archived Experiment 2 .pt state")
        subprocess.run(["gcloud", "storage", "cp", f"{remote}/{row[key]}", str(target)], check=True)
    return directory


def synthetic_source(directory):
    """Real FHP encodings with artificial soft targets; never scientific output."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = contract(True)
    encoder = FHPFeatureEncoder()
    model = new_model(config, 55)
    game, rng = load_fhp_game(), np.random.default_rng(19)
    observations = []
    while len(observations) < 64:
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                state.apply_action(int(rng.choice([a for a, _ in outcomes], p=[p for _, p in outcomes])))
                continue
            mask = np.asarray(state.legal_actions_mask(), dtype=np.float32)
            probs = rng.random(3) * mask
            probs /= probs.sum()
            observations.append((encoder.information_state(state), probs, mask))
            state.apply_action(int(rng.choice(3, p=probs)))
    features, targets, masks = (np.stack([r[j] for r in observations]) for j in range(3))
    # Repeated observations with different iterations exercise sufficient statistics.
    features, targets, masks = (np.repeat(a, 3, axis=0) for a in (features, targets, masks))
    size = len(features)
    payload = {"type": CHECKPOINT_TYPE, "version": CHECKPOINT_VERSION,
               "game": serialisable_game_definition(), "seed": 0, "experiment_id": 2,
               "experiment_name": SOURCE_EXPERIMENT, "algorithm_id": SOURCE_ALGORITHM,
               "outer_iteration": 10, "nodes_touched": size, "input_size": 183,
               "num_actions": 3, "policy_network_layers": [192, 192],
               "feature_encoder": encoder.metadata(), "policy_model": model.checkpoint_metadata(),
               "policy_state_dict": model.state_dict(), "audit_synthetic_fixture": True}
    with (directory / "policy.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    state = {"type": "exp2_fhp_lossless_structured_full_training_state", "schema_version": 1,
             "seed": 0, "checkpoint_id": "time_24h", "solver": {"num_iteration": 10},
             "average_policy_trainer": {"model": model.state_dict(),
                                        "buffer": {"size": size, "cur_id": size, "infostate": features,
                                                   "q_value": targets, "q_value_mask": masks,
                                                   "iteration": rng.integers(1, 11, (size, 1))}}}
    torch.save(state, directory / "state.pt")
    write_json(directory / "run_manifest.json", {"experiment_name": SOURCE_EXPERIMENT,
                                                 "algorithm_id": SOURCE_ALGORITHM, "seed": 0})
    write_json(directory / "SUCCESS.json", {"smoke_only": True})
    write_json(directory / "checkpoint_manifest.json", [{"checkpoint_id": "time_24h",
               "checkpoint_target_seconds": 86400, "outer_iteration": 10, "nodes_touched": size,
               "path": "policy.pkl", "sha256": sha256_file(directory / "policy.pkl"),
               "training_state_path": "state.pt", "training_state_sha256": sha256_file(directory / "state.pt")}])
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("worker", "smoke", "aggregate"):
        command = sub.add_parser(action)
        command.add_argument("--output-root", type=Path, required=True)
        if action != "aggregate":
            command.add_argument("--source-worker", type=Path)
            command.add_argument("--evaluation-workers", type=int, default=8)
        if action == "worker":
            command.add_argument("--seed", type=int, required=True)
    fetch = sub.add_parser("fetch-source")
    fetch.add_argument("--bucket", required=True)
    fetch.add_argument("--run-id", required=True)
    fetch.add_argument("--seed", type=int, required=True)
    fetch.add_argument("--directory", type=Path, required=True)
    sub.add_parser("contract")
    args = parser.parse_args(argv)
    if args.action == "fetch-source":
        fetch_source(args.bucket, args.run_id, args.seed, args.directory)
    elif args.action == "contract":
        print(json.dumps(contract(), indent=2))
    elif args.action == "aggregate":
        print(aggregate(args.output_root))
    elif args.action == "smoke":
        with TemporaryDirectory(prefix="fhp-audit-input-") as temporary:
            source = args.source_worker or synthetic_source(Path(temporary))
            run_worker(source, args.output_root, 0, smoke=True, evaluation_workers=args.evaluation_workers)
        print(aggregate(args.output_root, smoke=True))
    else:
        if args.source_worker is None:
            parser.error("worker requires --source-worker")
        run_worker(args.source_worker, args.output_root, args.seed, evaluation_workers=args.evaluation_workers)


if __name__ == "__main__":
    main()
