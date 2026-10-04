"""Cloud smoke -> three frozen-data audit workers -> seed-level aggregation."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.critic_target_cache_benchmark import write_json
from fhp_escher.checkpointing import sha256_file
from .config import contract, TRAINING_REF
from .fitting import audit_member
from .diagnostics import evaluate
from .source import fetch_source, preflight, validate_local, validate_runtime


def sync(directory):
    remote = os.environ.get("EXP17_REMOTE_WORKER")
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(directory), remote], check=True)


def synthetic_solver(seed):
    """Tiny real-game fixture. No synthetic learning result enters production."""
    from experiments.fhp.exp10_fhp_hand_board_features.config import smoke_config
    from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import install_cache
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    from unbiased_escher.fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
    config = smoke_config()
    config.update(evaluation_frequency=0, evaluate_initial_policy=False, early_evaluation_node_thresholds=(),
                  baseline_network_train_steps=2)
    kwargs = _solver_kwargs(seed, config)
    kwargs.pop("cache_frozen_critic_targets")
    solver = StructuredFHPGroupedWideUCVEscher(**kwargs)
    install_cache(solver, enabled=True)
    solver.iteration()
    return solver


def load_solver(source, seed, smoke):
    from experiments.fhp.exp10_fhp_hand_board_features import config as c, training_state as ts
    from experiments.fhp.exp10_fhp_hand_board_features.worker import _make_solver
    state_path, provenance = validate_local(source, seed)
    # The source identity and learning config are checked by the native reader;
    # no permissive generic-resume override is introduced into the trainer.
    payload = ts.read_training_state(state_path)
    solver = _make_solver(seed, c.EXPERIMENT_CONFIG, smoke=False)
    try:
        ts.restore_training_state(solver, payload, seed=seed, repository_commit=TRAINING_REF,
                                  config=c.EXPERIMENT_CONFIG)
        if payload.get("checkpoint_id") != "time_24h" or len(payload["captured_checkpoints"]) != 4:
            raise ValueError("Expected the completed 24-hour Experiment 10 state")
    except BaseException:
        solver.close()
        raise
    del payload
    gc.collect()
    # Suppress only automatic policy evaluation/checkpoint side effects.
    solver.evaluation_frequency = 0
    solver._maybe_run_training_time_checkpoint = lambda: None
    if smoke:
        # Real-state cloud smoke retains the original model/replay dimensions and
        # all eight restored actors; only diagnostic workload is reduced.
        solver.num_traversals = 16
        for trainer in solver.regret_trainers:
            trainer.train_steps = 2
        solver.calibration_trainer.train_steps = 2
        for member in solver.q_value_trainer.members:
            member.train_steps = 2
    return solver, provenance


def fingerprint(repo):
    paths = []
    for directory in ("experiments/fhp", "fhp_escher", "unbiased_escher", "adaptive_escher", "vr_deep_cfr"):
        paths.extend((repo / directory).rglob("*.py"))
    return hashlib.sha256(b"".join(str(p.relative_to(repo)).encode() + p.read_bytes()
                                   for p in sorted(paths))).hexdigest()


def audit_boundary(solver, config, *, seed, boundary, directory):
    """Observe a normal complete iteration; never train on diagnostic samples."""
    ensemble = solver.q_value_trainer
    original_fit = ensemble.train_model
    results, snapshots, old_targets = [], [], []
    times = dict(collection=0., regret=0., calibration=0.)
    originals = []

    def timed(obj, name, category):
        original = getattr(obj, name)
        originals.append((obj, name, original))
        def call(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                times[category] += time.perf_counter() - started
        setattr(obj, name, call)

    def fit(iteration):
        if results:
            raise RuntimeError("Expected one critic-ensemble fit per iteration")
        for fold, member in enumerate(ensemble.members):
            metric, states, target = audit_member(member, iteration, updates=config["updates"],
                probe_rows=config["training_probe_rows"], diagnostic_seed=seed * 1000 + boundary * 10 + fold)
            metric["critic_fold"] = fold
            results.append(metric)
            snapshots.append(states)
            old_targets.append(target)
        return float(np.mean([r["full_native_final_loss"] for r in results]))

    start_nodes, start_episode = solver.nodes_touched, solver.episode
    timed(solver, "collect_training_data", "collection")
    timed(solver, "train_regret", "regret")
    timed(solver.calibration_trainer, "train_model", "calibration")
    ensemble.train_model = fit
    started = time.perf_counter()
    try:
        solver.iteration()
    finally:
        ensemble.train_model = original_fit
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
    observed_seconds = time.perf_counter() - started
    if len(results) != 2 or solver.episode - start_episode != 2 * solver.num_traversals:
        raise RuntimeError("Audit did not complete normal collection and both critic fits")
    times["critics"] = sum(r["full_native_fit_seconds"] for r in results)
    # Detailed data below is outside learning-phase timing. A local Generator
    # drives all rollouts; verify global and replay RNGs remain untouched.
    from experiments.critic_target_cache_benchmark import rng_hash
    before = [rng_hash(m) for m in ensemble.members]
    states_before = (solver.episode, solver.nodes_touched, solver.num_iteration,
                     [(m.buffer.cur_id, len(m.buffer)) for m in ensemble.members],
                     solver.calibration_trainer.buffer.cursor, solver.ave_policy_trainer.buffer.cur_id)
    diagnostic_start = time.perf_counter()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        probes = evaluate(solver, snapshots, old_targets, config, seed=seed, boundary=boundary)
    finally:
        torch.set_num_threads(threads)
    after = [rng_hash(m) for m in ensemble.members]
    states_after = (solver.episode, solver.nodes_touched, solver.num_iteration,
                    [(m.buffer.cur_id, len(m.buffer)) for m in ensemble.members],
                    solver.calibration_trainer.buffer.cursor, solver.ave_policy_trainer.buffer.cur_id)
    if before != after or states_before != states_after:
        raise RuntimeError("Diagnostic evaluation modified training state or RNG")
    directory = Path(directory)
    weights = directory / f"boundary_{boundary}_critics.pt"
    temporary = weights.with_suffix(".pt.tmp")
    torch.save(dict(type="exp17_critic_diagnostic_weights_not_resumable", iteration=solver.num_iteration,
                    arms=snapshots, frozen_td_targets=old_targets), temporary)
    temporary.replace(weights)
    result = dict(seed=seed, boundary=boundary, iteration=solver.num_iteration,
        added_nodes=solver.nodes_touched-start_nodes, added_trajectories=solver.episode-start_episode,
        phase_seconds=times, observed_iteration_seconds_including_audit=observed_seconds,
        diagnostic_seconds=time.perf_counter()-diagnostic_start, critics=results, probes=probes,
        diagnostic_state_and_rng_preserved=True,
        weights_path=weights.name, weights_sha256=sha256_file(weights))
    write_json(directory / f"boundary_{boundary}.json", result)
    sync(directory)
    return result


def run_worker(source, root, seed, *, smoke=False):
    config = contract(smoke)
    if seed not in config["seeds"]:
        raise ValueError("Unexpected seed")
    root = Path(root).resolve()
    if source:
        source = Path(source).resolve()
        if source.is_relative_to(root) or root.is_relative_to(source):
            raise ValueError("Source state must remain outside the uploaded output tree")
    elif not smoke:
        raise ValueError("Production requires a verified Experiment 10 source")
    torch.set_num_threads(config["learner_threads"])
    repo = Path(__file__).resolve().parents[3]
    runtime = (validate_runtime() if not smoke else dict(python_version=platform.python_version(),
               torch_version=str(torch.__version__), numpy_version=np.__version__))
    if not smoke:
        if torch.get_num_interop_threads() != 8:
            torch.set_num_interop_threads(8)
        # These learning kernels have not changed since the source training run.
        # Fail closed if a future algorithm change would silently alter the audit.
        subprocess.run(["git", "diff", "--exit-code", TRAINING_REF, "--", "unbiased_escher",
                        "adaptive_escher", "vr_deep_cfr"], cwd=repo, check=True, capture_output=True)
    runtime.update(torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                   logical_cpus_visible=os.cpu_count(), configured_machine="n2-standard-16" if source else "local_smoke",
                   collection_actors=8 if source else 0)
    provenance = validate_local(source, seed)[1] if source else dict(synthetic=True, seed=seed)
    manifest = dict(config=config, provenance=provenance, runtime=runtime, code_sha256=fingerprint(repo),
                    audit_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    directory = root / "workers" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Source/config/code/runtime changed; use a new RUN_ID")
    write_json(manifest_path, manifest)
    success = directory / "SUCCESS.json"
    if success.exists():
        complete = json.loads(success.read_text())
        if any(sha256_file(directory / name) != digest for name, digest in complete["artifacts"].items()):
            raise ValueError("Completed audit artifacts are corrupt")
        return directory
    solver = None
    started = time.perf_counter()
    try:
        solver = load_solver(source, seed, smoke)[0] if source else synthetic_solver(seed)
        initial_iteration = solver.num_iteration
        boundaries = [audit_boundary(solver, config, seed=seed, boundary=b, directory=directory)
                      for b in range(1, config["boundaries"] + 1)]
        summary = dict(seed=seed, smoke=smoke, source_iteration=initial_iteration,
            final_iteration=solver.num_iteration, boundaries=config["boundaries"],
            elapsed_seconds=time.perf_counter()-started,
            peak_driver_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
            driver_rss_note="driver process only; Batch resource diagnostics include child processes",
            diagnostic_state_and_rng_preserved=all(b["diagnostic_state_and_rng_preserved"] for b in boundaries))
        write_json(directory / "summary.json", summary)
        artifacts = [p for p in directory.iterdir() if p.suffix in (".json", ".pt") and p.name != "SUCCESS.json"]
        write_json(success, dict(seed=seed, smoke=smoke, status="complete",
                                 artifacts={p.name: sha256_file(p) for p in artifacts}))
        sync(directory)
    finally:
        if solver is not None and hasattr(solver, "close"):
            solver.close()
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("worker", "smoke", "aggregate"):
        command = sub.add_parser(action)
        command.add_argument("--output-root", type=Path, required=True)
        if action != "aggregate":
            command.add_argument("--source-worker", type=Path)
        if action == "worker":
            command.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    fetch = sub.add_parser("fetch-source")
    fetch.add_argument("--bucket", required=True)
    fetch.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    fetch.add_argument("--directory", type=Path, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--bucket", required=True)
    args = parser.parse_args(argv)
    if args.action == "preflight":
        print(json.dumps(preflight(args.bucket), indent=2))
    elif args.action == "fetch-source":
        fetch_source(args.bucket, args.seed, args.directory)
    elif args.action == "aggregate":
        from .aggregate import aggregate
        aggregate(args.output_root)
    else:
        smoke = args.action == "smoke"
        run_worker(args.source_worker, args.output_root, 0 if smoke else args.seed, smoke=smoke)
        if smoke:
            from .aggregate import aggregate
            aggregate(args.output_root, smoke=True)


if __name__ == "__main__":
    main()
