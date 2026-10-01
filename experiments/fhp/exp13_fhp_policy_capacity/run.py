"""Screen -> global validation selection -> test/refit/evaluate -> aggregate."""

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

from experiments.fhp.exp4_fhp_average_policy_audit.run import fetch_source, synthetic_source
from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file
from fhp_escher.game import load_fhp_game
from .config import contract, arm_specs
from .data import load_source, group_replay, group_diagnostics, group_digest, split_groups
from .evaluation import evaluate
from .fitting import fit_path, diagnostics, fitting_benchmark, inference_benchmark, write_json
from .selection import fit_directory, select, read_selection, screen_key, screen_identity


def sync_output(directory, config=None):
    remote = os.environ.get((config or {}).get("remote_worker_env", "EXP13_REMOTE_WORKER"))
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", "--exclude", r"\.pt$|\.tmp$",
                        str(directory), remote], check=True)


def resources(directory, stage, started, fitting_seconds=None, evaluation_workers=None):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(directory / f"{stage}_resources.json", {
        "total_wall_seconds_this_attempt": time.monotonic() - started,
        "fit_and_group_seconds_this_attempt": fitting_seconds,
        "main_process_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
        "evaluation_workers": evaluation_workers,
        "note": "Process peak excludes evaluation children. Resource logs cover the whole VM; fits/evaluation have separate timers.",
    })


def prepare(source, root, seed, config, *, splitter=None):
    if seed not in config["seeds"]:
        raise ValueError("Unexpected source seed")
    torch.set_num_threads(1)
    directory = Path(root) / "workers" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    buffer, iteration, template, provenance = load_source(source, seed)
    if template.get("audit_synthetic_fixture") and not config["smoke"]:
        raise ValueError("Synthetic source cannot be used for production")
    if not template.get("audit_synthetic_fixture") and int(buffer["size"]) != config["source_replay_rows"]:
        raise ValueError("Frozen audit requires exactly 1,000,000 source replay rows per seed")
    provenance.pop("source_worker", None)
    repo = Path(__file__).resolve().parents[3]
    code_paths = sorted(Path(__file__).parent.glob("*.py")) + sorted(
        (Path(__file__).parent.parent / "exp4_fhp_average_policy_audit").glob("*.py"))
    code_paths += sorted((repo / "fhp_escher").rglob("*.py"))
    code_paths += sorted((repo / "fhp_evaluation").rglob("*.py"))
    code_paths += [Path(__file__).parent.parent / "retrospective_exp2_exp3_evaluation" / "run.py"]
    if config["experiment_id"] != 13:
        code_paths += sorted((repo / "experiments/fhp" / config["experiment_name"]).glob("*.py"))
    manifest = {"config": config, "provenance": provenance,
                "audit_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "audit_source_sha256": hashlib.sha256(b"".join(
                    path.parent.name.encode() + path.name.encode() + path.read_bytes() for path in code_paths)).hexdigest(),
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    path = directory / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Source/configuration/code mismatch; use a new RUN_ID")
    write_json(path, manifest)
    started = time.monotonic()
    groups = group_replay(buffer, iteration, config["gamma"])
    del buffer
    gc.collect()
    full_info = group_diagnostics(groups)
    benchmark_path = directory / "capacity_benchmark.json"
    if config["smoke"] and not template.get("audit_synthetic_fixture") and not benchmark_path.exists():
        benchmark = fitting_benchmark(groups, {**config, "batch_size": 2048})
        write_json(benchmark_path, benchmark)
        print("Production-batch fitting benchmark: " + json.dumps(benchmark), flush=True)
    if config["smoke"] and len(groups.features) > 128:
        groups = groups.subset(np.random.default_rng(902).choice(len(groups.features), 128, replace=False))
    splits, metadata = (splitter or split_groups)(groups, config, seed)
    replay_info = {"full_source": full_info, "fit_groups": group_diagnostics(groups),
                   "data_sha256": group_digest(groups), "splits": metadata,
                   "grouping_seconds": time.monotonic() - started}
    path = directory / "replay_diagnostics.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "grouping_seconds"} != {
            k: v for k, v in replay_info.items() if k != "grouping_seconds"
        }:
            raise ValueError("Replay or split identity changed")
    else:
        write_json(path, replay_info)
    return directory, manifest, groups, splits, template, replay_info


def screen_worker(source, root, seed, *, smoke=False, config=None, splitter=None):
    started, config = time.monotonic(), contract(smoke) if config is None else config
    directory, _, groups, splits, template, info = prepare(source, root, seed, config, splitter=splitter)
    sync = lambda: sync_output(directory, config)
    # The test split is not passed to the fitter; no test metrics at this stage.
    del groups
    splits.pop("test")
    for architecture in config["architectures"]:
        for recipe in config["recipes"]:
            for replicate in config["replicates"]:
                fit_path(splits["training"], splits["validation"], config=config, seed=seed,
                         replicate=replicate, architecture=architecture, recipe=recipe,
                         updates=config["screen_updates"],
                         directory=fit_directory(directory, "screen", architecture, recipe, replicate),
                         template=template, data_sha256=info["splits"]["training"]["data_sha256"],
                         phase="screen", on_checkpoint=sync)
    resources(directory, "screen", started)
    sync()
    write_json(directory / "SCREEN_SUCCESS.json", {"seed": seed, "smoke": smoke, "status": "screen_complete"})
    sync()


def deploy_worker(source, root, seed, *, smoke=False, evaluation_workers=8, config=None, splitter=None):
    if not 1 <= evaluation_workers <= 8:
        raise ValueError("Evaluation workers must be 1..8")
    started, config = time.monotonic(), contract(smoke) if config is None else config
    directory, manifest, groups, splits, template, info = prepare(source, root, seed, config, splitter=splitter)
    selection = read_selection(Path(root) / "selection.json", config, manifest, seed)
    if not (directory / "SCREEN_SUCCESS.json").exists():
        raise ValueError("Screen must complete before deployment")
    sync = lambda: sync_output(directory, config)
    specs = arm_specs(config, selection)
    # Fixed controls and selected models only. Lock is copied alongside test results.
    test_path = directory / "diagnostic_test.json"
    test_rows, tested = [], {}
    for arm, spec in specs.items():
        key = (spec["architecture"], spec["recipe"], spec["replicate"], spec["updates"])
        path = fit_directory(directory, "screen", *key[:3]) / f"{key[3]}.pkl"
        if key not in tested:
            _, metrics, fit_hash = screen_identity(path.parent)
            if selection["screen_fit_hashes"][screen_key(seed, *key[:3])] != fit_hash:
                raise ValueError("Screen results changed after selection was locked")
            expected = next(row["policy_sha256"] for row in metrics if row["updates"] == key[3])
            if sha256_file(path) != expected:
                raise ValueError("Diagnostic checkpoint checksum mismatch")
            tested[key] = {"policy_sha256": expected,
                           "test": diagnostics(LoadedFHPPolicy(load_fhp_game(), path).model, splits["test"])}
        test_rows.append({"arm": arm, **spec, **tested[key]})
    tests = {"selection_sha256": selection["selection_sha256"], "seed": seed, "rows": test_rows,
             "test_data_sha256": info["splits"]["test"]["data_sha256"]}
    if test_path.exists() and json.loads(test_path.read_text()) != tests:
        raise ValueError("Locked test results differ")
    write_json(test_path, tests)
    del splits
    gc.collect()
    with (directory / "archived.pkl").open("wb") as handle:
        pickle.dump(template, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Share paths when tuned/control endpoints use the same recipe. All fits reset.
    paths = {}
    for spec in specs.values():
        key = (spec["architecture"], spec["recipe"], spec["replicate"])
        paths.setdefault(key, set()).add(spec["updates"])
    results = {}
    for (architecture, recipe, replicate), updates in paths.items():
        fits = fit_path(groups, None, config=config, seed=seed, replicate=replicate,
                        architecture=architecture, recipe=recipe, updates=updates,
                        directory=fit_directory(directory, "deployment", architecture, recipe, replicate),
                        template=template, data_sha256=info["data_sha256"], phase="deployment", on_checkpoint=sync)
        for fit in fits:
            results[(architecture, recipe, replicate, fit["updates"])] = fit
    policies, records, timings = {"archived": directory / "archived.pkl"}, [], {}
    for arm, spec in specs.items():
        key = (spec["architecture"], spec["recipe"], spec["replicate"], spec["updates"])
        path = fit_directory(directory, "deployment", *key[:3]) / f"{key[3]}.pkl"
        policies[arm] = path
        if key not in timings:
            timings[key] = inference_benchmark(path, groups.features)
        records.append({**results[key], "arm": arm, "path": str(path.relative_to(directory)),
                        "inference": timings[key]})
    write_json(directory / "deployment_metrics.json", records)
    write_json(directory / "locked_selection.json", selection)
    fit_elapsed = time.monotonic() - started
    del groups, template
    gc.collect()
    evaluate(policies, config, seed, directory / "evaluation_tasks", evaluation_workers, sync)
    resources(directory, "deployment", started, fit_elapsed, evaluation_workers)
    sync()
    write_json(directory / "SUCCESS.json", {"seed": seed, "smoke": smoke, "status": "complete",
                                            "selection_sha256": selection["selection_sha256"]})
    sync()


def main(argv=None, *, contract_factory=None, splitter=None, aggregate_fn=None):
    from .aggregate import aggregate
    make_config = contract if contract_factory is None else contract_factory
    run_aggregate = aggregate if aggregate_fn is None else aggregate_fn
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("screen", "deploy", "select", "aggregate", "smoke"):
        command = sub.add_parser(action)
        command.add_argument("--output-root", type=Path, required=True)
        if action in ("screen", "deploy", "smoke"):
            command.add_argument("--source-worker", type=Path)
        if action in ("screen", "deploy"):
            command.add_argument("--seed", type=int, required=True)
        if action in ("deploy", "smoke"):
            command.add_argument("--evaluation-workers", type=int, default=8)
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
        print(json.dumps(make_config(), indent=2))
    elif args.action == "select":
        select(args.output_root, make_config())
    elif args.action == "aggregate":
        print(run_aggregate(args.output_root, config=make_config()))
    elif args.action == "smoke":
        with TemporaryDirectory(prefix="fhp-capacity-source-") as temporary:
            source = args.source_worker or synthetic_source(temporary)
            config = make_config(True)
            screen_worker(source, args.output_root, 0, smoke=True, config=config, splitter=splitter)
            select(args.output_root, config)
            deploy_worker(source, args.output_root, 0, smoke=True, evaluation_workers=args.evaluation_workers,
                          config=config, splitter=splitter)
        print(run_aggregate(args.output_root, smoke=True, config=config))
    else:
        if args.source_worker is None:
            parser.error("screen/deploy requires --source-worker")
        if args.action == "screen":
            screen_worker(args.source_worker, args.output_root, args.seed, config=make_config(), splitter=splitter)
        else:
            deploy_worker(args.source_worker, args.output_root, args.seed, evaluation_workers=args.evaluation_workers,
                          config=make_config(), splitter=splitter)


if __name__ == "__main__":
    main()
