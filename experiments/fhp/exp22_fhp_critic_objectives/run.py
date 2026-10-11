"""Frozen-state worker, local smoke and aggregation; never runs solver.iteration in production."""
import argparse
from copy import deepcopy
import gc
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

from experiments.critic_target_cache_benchmark import write_json, weight_hash, rng_hash
from experiments.fhp.exp17_fhp_critic_budget.fitting import cpu_state, replay_digest
from experiments.fhp.exp17_fhp_critic_budget.run import fingerprint, synthetic_solver
from fhp_escher.checkpointing import sha256_file
from .config import contract, OBJECTIVES
from .data import frozen_sampler, generate_panel, assess
from .fitting import fit_objective, optimizer_hash
from .source import load_solver, fetch_source, preflight, validate_runtime


def sync(root):
    remote = os.environ.get("EXP22_REMOTE_WORKER")
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(root), remote], check=True)


def choose_candidates(rows):
    """Only selection-panel scores; identical decision rule for each objective."""
    scores = {}
    for row in rows:
        key = (row["objective"], row["updates"])
        scores.setdefault(key, []).append(row["selection"]["deployed"]["summary"]["mean_variance"])
    if any(len(v) != 2 for v in scores.values()):
        raise ValueError("Selection requires both critic folds")
    selected = {}
    for objective in OBJECTIVES:
        options = [k for k in scores if k[0] == objective]
        best = min(options, key=lambda k: (np.mean(scores[k]), k[1]))
        selected[objective] = dict(objective=best[0], updates=best[1],
                                    selection_variance=float(np.mean(scores[best])))
    selected["best_td_control"] = min((selected["replay_td"], selected["matched_td"]),
                                         key=lambda r: (r["selection_variance"], r["updates"], r["objective"]))
    return selected


def source_identity(solver):
    return dict(critics=[dict(online=weight_hash(cpu_state(m.model)),
                            target=weight_hash(cpu_state(m.target_model)), replay=replay_digest(m.buffer),
                            history=[weight_hash(s) for s in m.target_history], version=m.target_version,
                            optimizer=optimizer_hash(m.optimizer.state_dict()),
                            rng=rng_hash(m)) for m in solver.q_value_trainer.members],
                regret=[weight_hash(cpu_state(t.model)) for t in solver.regret_trainers],
                calibration=weight_hash(cpu_state(solver.calibration_trainer.target_model)),
                iteration=solver.num_iteration, nodes=solver.nodes_touched, episode=solver.episode)


def audit(solver, directory, config, seed, *, candidate_diagnostics=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = source_identity(solver)
    write_json(directory / "source_identity.json", original)
    sampler, seen, rows, panels = frozen_sampler(solver), set(), [], {}
    began, cpu_started = time.perf_counter(), time.process_time()
    for fold, member in enumerate(solver.q_value_trainer.members):
        torch.set_num_threads(config["diagnostic_threads"])
        split_panels = {}
        for split in ("fit", "selection"):
            panel, metadata = generate_panel(sampler, config, seed=seed, member=fold, split=split, seen=seen)
            path = directory / f"{split}_member_{fold}.npz"
            saving = time.perf_counter()
            panel.save(path)
            metadata.update(path=path.name, sha256=sha256_file(path),
                            save_seconds=time.perf_counter()-saving)
            panels[f"{fold}/{split}"] = metadata
            split_panels[split] = panel
        evaluation_model = deepcopy(member.model).eval()

        def candidate_callback(candidate, metrics):
            name = f"{candidate['objective']}_{candidate['updates']}_member_{fold}"
            path = directory / (name + ".pt")
            temporary = path.with_suffix(".pt.tmp")
            torch.save(candidate, temporary)
            temporary.replace(path)
            metrics.update(path=path.name, sha256=sha256_file(path), selection={})
            torch.set_num_threads(config["diagnostic_threads"])
            try:
                for representation in ("online", "deployed"):
                    evaluation_model.load_state_dict(candidate[representation])
                    metrics["selection"][representation] = assess(
                        evaluation_model, split_panels["selection"], config["evaluation_batch_size"])
                if candidate_diagnostics is not None:
                    metrics["fixed_fit_panel"] = candidate_diagnostics(
                        candidate, evaluation_model, split_panels["fit"], member, solver.num_iteration, config)
            finally:
                torch.set_num_threads(config["fit_threads"])

        # Rotate order across seeds/folds to reduce systematic timing order effects.
        offset = (seed+fold) % len(OBJECTIVES)
        order = OBJECTIVES[offset:] + OBJECTIVES[:offset]
        for objective in order:
            torch.set_num_threads(config["fit_threads"])
            fitted = fit_objective(member, solver.num_iteration, split_panels["fit"], objective,
                                   config, seed=seed, fold=fold, callback=candidate_callback)
            for row in fitted:
                data_cost = (0. if objective == "replay_td" or row["updates"] == 0 else
                             panels[f"{fold}/fit"]["generation_seconds"] + panels[f"{fold}/fit"]["save_seconds"])
                # Conservatively charge the entire shared panel cost to each method,
                # not a quarter just because the experimental implementation shares it.
                row["standalone_data_plus_fit_seconds"] = data_cost + row["fit_prefix_seconds"]
            rows.extend(fitted)
            write_json(directory / "fitting.json", rows)
            write_json(directory / "panels.json", panels)
            sync(directory)
        del split_panels, panel
        gc.collect()

    # Lock every model/budget decision before even constructing the test panels.
    decisions = choose_candidates(rows)
    write_json(directory / "selection_lock.json", dict(seed=seed, candidates=decisions,
                                                       fitting_sha256=sha256_file(directory / "fitting.json")))
    sync(directory)
    assessment_started = time.perf_counter()
    torch.set_num_threads(config["diagnostic_threads"])
    final, unchanged = [], []
    for fold, member in enumerate(solver.q_value_trainer.members):
        panel, metadata = generate_panel(sampler, config, seed=seed, member=fold, split="assessment", seen=seen)
        path = directory / f"assessment_member_{fold}.npz"
        panel.save(path)
        metadata.update(path=path.name, sha256=sha256_file(path))
        panels[f"{fold}/assessment"] = metadata
        model = deepcopy(member.model).eval()
        unchanged.append(dict(member=fold, assessment={
            "online": assess(deepcopy(member.model), panel, config["evaluation_batch_size"]),
            "deployed": assess(deepcopy(member.target_model), panel, config["evaluation_batch_size"])}))
        for row in [r for r in rows if r["member"] == fold]:
            path = directory / row["path"]
            if sha256_file(path) != row["sha256"]:
                raise ValueError("Candidate changed after selection")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            scores = {}
            started = time.perf_counter()
            for representation in ("online", "deployed"):
                model.load_state_dict(saved[representation])
                scores[representation] = assess(model, panel, config["evaluation_batch_size"])
            final.append(dict(objective=row["objective"], updates=row["updates"], member=fold,
                              assessment=scores, assessment_seconds=time.perf_counter()-started))
            write_json(directory / "assessment.json", final)
        del panel
        gc.collect()
    if source_identity(solver) != original:
        raise RuntimeError("Read-only audit modified source learning state or RNG")
    write_json(directory / "panels.json", panels)
    write_json(directory / "unchanged_assessment.json", unchanged)
    summary = dict(seed=seed, smoke=config["smoke"], source_state_preserved=True,
                   candidates=len(rows), elapsed_audit_seconds=time.perf_counter()-began,
                   final_assessment_seconds=time.perf_counter()-assessment_started,
                   observed_cpu_seconds=time.process_time()-cpu_started,
                   panel_generation_seconds=sum(p["generation_seconds"] for p in panels.values()),
                   all_objective_fit_seconds=sum(r["fit_prefix_seconds"] for r in rows
                                                 if r["updates"] == max(config["updates"])),
                   all_snapshot_and_selection_seconds=sum(r["snapshot_and_selection_seconds"] for r in rows),
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                       (1024**2 if sys.platform == "darwin" else 1024),
                   outcome="mechanism_screen_only_no_automatic_promotion",
                   limitation="late-stage conditional probe distribution; three source seeds maximum")
    write_json(directory / "summary.json", summary)
    return summary


def run_worker(source, output, seed, smoke=False):
    if seed not in (0, 1, 2) or (source is None and not smoke):
        raise ValueError("Production requires a verified source state and seed 0, 1 or 2")
    directory = Path(output).resolve()
    if source and (Path(source).resolve().is_relative_to(directory) or directory.is_relative_to(Path(source).resolve())):
        raise ValueError("Source and output directories must be disjoint")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh output directory; incomplete audits are not silently resumed")
    directory.mkdir(parents=True, exist_ok=True)
    config = contract(smoke)
    repo = Path(__file__).resolve().parents[3]
    torch.set_num_threads(config["fit_threads"])
    runtime = (dict(python_version=platform.python_version(), torch_version=str(torch.__version__),
                    numpy_version=np.__version__) if smoke else validate_runtime())
    if not smoke:
        if torch.get_num_interop_threads() != 8:
            torch.set_num_interop_threads(8)
        import importlib.metadata
        if importlib.metadata.version("open_spiel") != "1.6.3":
            raise ValueError("Production requires OpenSpiel 1.6.3")
    runtime.update(torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    started = time.perf_counter()
    solver, provenance = (load_solver(source, seed) if source else
                          (synthetic_solver(seed), dict(synthetic=True, seed=seed)))
    manifest = dict(config=config, seed=seed, runtime=runtime, provenance=provenance,
                    code_sha256=fingerprint(repo),
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    source_load_seconds=time.perf_counter()-started, collection_actors=0)
    write_json(directory / "manifest.json", manifest)
    try:
        audit(solver, directory, config, seed)
    except BaseException as error:
        import traceback
        write_json(directory / "failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    artifacts = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    write_json(directory / "SUCCESS.json", dict(status="complete", seed=seed, smoke=smoke, artifacts=artifacts))
    sync(directory)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("worker", "smoke"):
        p = sub.add_parser(action)
        p.add_argument("--source-worker", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    p = sub.add_parser("aggregate")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("fetch-source")
    p.add_argument("--bucket", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--bucket", required=True)
    args = parser.parse_args(argv)
    if args.action in ("worker", "smoke"):
        run_worker(args.source_worker, args.output, args.seed, smoke=args.action == "smoke")
    elif args.action == "fetch-source":
        fetch_source(args.bucket, args.seed, args.directory)
    elif args.action == "preflight":
        print(json.dumps(preflight(args.bucket), indent=2))
    else:
        from .aggregate import aggregate
        aggregate(args.root, pilot=args.pilot, smoke=args.smoke)


if __name__ == "__main__":
    main()
