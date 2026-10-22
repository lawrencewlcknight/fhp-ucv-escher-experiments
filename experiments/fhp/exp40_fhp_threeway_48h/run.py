"""One resumable CPU evaluation: validate, smoke, profile and full scoring."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from importlib.metadata import version
import multiprocessing
from pathlib import Path
import platform
import time

from . import engine as native, sources

ROOT = Path(__file__).resolve().parents[3]


def run_tasks(tasks, output, args, engine, deadline):
    """One atomic file per completed shard. Never rescore a valid cached shard."""
    output.mkdir(parents=True, exist_ok=True)
    results, pending = [], []
    for task in tasks:
        path = output/(task["task_id"]+".json")
        if path.exists():
            row = engine.read_json(path)
            engine.validate_result(row, task)
            results.append(row)
        else:
            pending.append(task)
    if not pending:
        return results
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=native.initialise_worker, initargs=(str(args.vr_repo), str(args.sd_repo), str(ROOT)))
    failed = False
    try:
        iterator, futures = iter(pending), {}
        def submit_one():
            task = next(iterator, None)
            if task is not None:
                futures[pool.submit(native.execute_task, task)] = task
        for _ in range(args.workers):
            submit_one()
        while futures:
            if time.monotonic() >= deadline:
                raise TimeoutError("Time ceiling reached; completed shards are retained for resume")
            done, _ = wait(futures, timeout=min(5, max(.001, deadline-time.monotonic())),
                           return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                row = future.result()
                engine.validate_result(row, task)
                engine.write_json(output/(task["task_id"]+".json"), row)
                results.append(row)
                engine.write_json(output.parent/"progress.json", dict(stage=task["stage"],
                    complete=len(results), total=len(tasks), updated_utc=datetime.now(timezone.utc).isoformat()))
                print(f"{task['stage']}: {len(results)}/{len(tasks)} shards complete", flush=True)
                submit_one()
    except BaseException:
        failed = True
        for process in tuple(pool._processes.values()):
            if process.is_alive():
                process.terminate()
        raise
    finally:
        pool.shutdown(wait=not failed, cancel_futures=True)
    return results


def implementation_digest(engine, args):
    files = {str(p.relative_to(ROOT)): engine.sha256(p)
        for base in (ROOT/"experiments/fhp/exp40_fhp_threeway_48h",
                     ROOT/"experiments/fhp/exp38_fhp_strength_critic2500_48h")
        for p in sorted(base.glob("*.py"))}
    files["requirements.txt"] = engine.sha256(ROOT/"requirements.txt")
    return engine.digest(dict(native=engine.implementation_digest(args.sd_repo, ROOT), local=files))


def report(results, output, records, costs, engine):
    summary = engine.report_main(results, output)
    summary.update(experiment_id=40, smoke=False, strategic_audit_evaluated=False,
        strategic_audit="not included", policy_labels=native.LABELS,
        selected_source_runs=native.RUN_IDS, exact_exploitability=False,
        policy_representations=dict(sd="full uniform historical trajectory mixture; no thinning",
            ucv="native saved average-policy network", vr="native saved average-policy network"))
    summary["caveats"] += [
        "Comparison of selected implementations, not an architecture or critic-budget ablation.",
        "Representations, network capacity, tuning effort and active-clock definitions differ.",
        "Three training seeds give exploratory intervals, not a definitive algorithm-wide ranking.",
        "Training-seed bootstrap and conditional deal uncertainty are separate, not a joint confidence region.",
        "Fresh deals do not undo historical model-selection bias; no automatic winner is promoted."]
    # MC uncertainty is conditional on these nine policies. Do not count the
    # nine cross-seed cells as nine independent training replications.
    for row in summary["direct_comparisons"]:
        se, mean = row["conditional_mc_se_mbb"], row["mean_mbb_per_hand"]
        row.update(conditional_mc_ci95_low=mean-1.96*se, conditional_mc_ci95_high=mean+1.96*se)
    engine.write_json(output/"summary.json", summary)
    engine.write_json(output/"training_cost_context.json", dict(records=costs,
        interpretation="Native counters retained without conflating different active-time definitions; not equal-total-compute evidence."))
    engine.write_csv(output/"checkpoint_index.csv", engine.portable(records))
    engine.write_json(output/"checkpoint_index.json", engine.portable(records))
    engine.write_csv(output/"direct_comparison_summary.csv", summary["direct_comparisons"])
    with (output/"analysis_summary.md").open("a") as stream:
        stream.write("\n## Frozen sources\n\n" + "\n".join(f"- {k}: {v}." for k,v in native.LABELS.items()))
        stream.write("\n\n## Interpretation\n\n" + "\n".join("- "+v for v in summary["caveats"]) + "\n")
        stream.write("\nRule-agent results are supporting evidence, not Nash distance. LBR and the strategic-position audit were not run.\n")
    return summary


def run(args):
    started = time.monotonic()
    remaining = args.max_hours*3600
    if args.deadline_unix is not None:
        remaining = min(remaining, args.deadline_unix-time.time())
    deadline = started + remaining
    engine = native.load(args.vr_repo, args.sd_repo, ROOT)
    import torch
    torch.set_num_threads(1)
    root, output = args.source_root.resolve(), args.output.resolve()
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Source and output directories must be disjoint")
    records = sources.validate(root, engine, args.sd_repo, ROOT)
    costs = sources.cost_context(root, records, engine)
    implementation = implementation_digest(engine, args)
    identity = dict(protocol=native.PROTOCOL, implementation_sha256=implementation,
        native_execution_refs=dict(vr=native.VR_REF, sd=native.SD_REF),
        sources=engine.portable(records), training_summaries_sha256=engine.digest(costs),
        direct_pairs_per_cell=engine.DIRECT_PAIRS, rule_pairs_per_cell=engine.RULE_PAIRS,
        base_seed=engine.BASE_SEED, smoke_only=args.smoke_only, source_uris=sorted(args.source_uri),
        python=platform.python_version(), dependencies={k: version(k) for k in
            ("torch", "numpy", "open_spiel", "scipy", "matplotlib")})
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"evaluation_manifest.json"
    if manifest.exists() and engine.read_json(manifest) != identity:
        raise ValueError("Incompatible resume contract; do not mix sources, code or deal streams")
    if engine._complete_stage(output, identity):
        return output
    engine.write_json(manifest, identity)
    engine.write_json(output/"STATUS.json", dict(status="running", stage="smoke"))
    try:
        smoke = run_tasks(engine.build_tasks(records, "smoke", implementation),
            output/"smoke/task_results", args, engine, deadline)
        engine.write_json(output/"smoke/SUCCESS.json", dict(status="complete", smoke=True,
            tasks=len(smoke), strength_evidence=False, identity_sha256=engine.digest(identity)))
        if args.smoke_only:
            summary = dict(experiment_id=40, status="complete", smoke=True, strength_evidence=False)
            engine.write_json(output/"summary.json", summary)
        else:
            tasks = engine.build_tasks(records, "production", implementation)
            engine.write_json(output/"STATUS.json", dict(status="running", stage="profiling"))
            probes = run_tasks(engine.build_tasks(records, "profile", implementation),
                output/"profile/task_results", args, engine, deadline)
            estimate = engine.cost_estimate(probes, tasks, args.workers)
            estimate.update(informational_only=True, cost_gate_enabled=False,
                note="No rejection based on this estimate. Cached timings may come from an earlier invocation.")
            engine.write_json(output/"timing_forecast.json", estimate)
            print(f"Informational timing forecast: {estimate}", flush=True)
            engine.write_json(output/"STATUS.json", dict(status="running", stage="scoring"))
            results = run_tasks(tasks, output/"task_results", args, engine, deadline)
            summary = report(results, output, records, costs, engine)
            summary.update(tasks=len(results), duplicate_pairs=sum(t["num_deals"] for t in tasks),
                hands=2*sum(t["num_deals"] for t in tasks),
                invocation_seconds=time.monotonic()-started,
                scoring_worker_seconds=sum(r["elapsed_seconds"] for r in results),
                profile_worker_seconds=sum(r["elapsed_seconds"] for r in probes),
                smoke_worker_seconds=sum(r["elapsed_seconds"] for r in smoke))
            summary["scoring_hands_per_worker_second"] = summary["hands"]/summary["scoring_worker_seconds"]
            engine.write_json(output/"summary.json", summary)
        files = {str(p.relative_to(output)): engine.sha256(p) for p in output.iterdir()
            if p.is_file() and p.name not in {"SUCCESS.json", "STATUS.json", "progress.json"}}
        engine.write_json(output/"STATUS.json", dict(status="complete", smoke=args.smoke_only))
        engine.write_json(output/"SUCCESS.json", dict(status="complete", experiment_id=40,
            smoke=args.smoke_only, identity_sha256=engine.digest(identity), files=files,
            tasks=summary.get("tasks", len(smoke)), duplicate_pairs=summary.get("duplicate_pairs", 144)))
    except BaseException as exc:
        engine.write_json(output/"STATUS.json", dict(status="incomplete", error_type=type(exc).__name__,
            message=str(exc), completed_shards_retained=True))
        raise
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "vr-repo", "sd-repo", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1,17), default=16)
    p.add_argument("--max-hours", type=float, default=23)
    p.add_argument("--deadline-unix", type=float)
    p.add_argument("--source-uri", action="append", default=[])
    p.add_argument("--smoke-only", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not 0 < args.max_hours <= 23:
        p.error("Use a positive evaluator allowance of at most 23h (24h Batch cap includes setup/upload)")
    run(args)


if __name__ == "__main__":
    main()
