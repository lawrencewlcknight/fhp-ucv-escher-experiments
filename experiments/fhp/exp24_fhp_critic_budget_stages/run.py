"""One shared reference, then immutable pre-fit audits and two analysis reports."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch

from experiments.critic_target_cache_benchmark import write_json
from experiments.fhp.exp17_fhp_critic_budget.source import validate_runtime
from experiments.fhp.exp22_fhp_critic_objectives.run import audit, fingerprint
from fhp_escher.checkpointing import sha256_file
from .config import contract, STAGES
from .diagnostics import FitPanelDiagnostics
from .states import load_audit_solver, restore_reference, commit
from .reference import collect, make_solver


def sync(root):
    remote = os.environ.get("EXP24_REMOTE_OUTPUT")
    if remote:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", "--exclude", r"(^|/)[^/]+\.tmp(/|$)",
                        str(root), remote], check=True)


def runtime(smoke):
    if smoke:
        result = dict(python_version=platform.python_version(), torch_version=str(torch.__version__), numpy_version=np.__version__)
    else:
        result = validate_runtime()
        import importlib.metadata
        if importlib.metadata.version("open_spiel") != "1.6.3":
            raise ValueError("Production requires OpenSpiel 1.6.3")
    torch.set_num_threads(1 if smoke else 8)
    if not smoke and torch.get_num_interop_threads() != 8:
        torch.set_num_interop_threads(8)
    return result


def reference(root, *, smoke=False, ray_smoke=False, source=None):
    root = Path(root)
    runtime(smoke)
    solver = None
    started = time.perf_counter()
    try:
        if source:
            from experiments.fhp.exp17_fhp_critic_budget.run import load_solver
            # Keep the FULL native late iteration even in cloud smoke. This
            # checks real source restoration, replay dimensions and eight actors.
            validate_runtime()
            torch.set_num_threads(8)
            if torch.get_num_interop_threads() != 8:
                torch.set_num_interop_threads(8)
            solver, lineage = load_solver(source, 0, smoke=False)
        else:
            solver = make_solver(smoke=smoke, ray_smoke=ray_smoke)
            lineage = dict(synthetic=smoke, new_reference=True, seed=0)
        write_json(root / ("late_source_load.json" if source else "reference_initialisation.json"), dict(seconds=time.perf_counter()-started,
            lineage=lineage, runtime=runtime_info(), commit=commit()))
        directory = root / ("late_reference" if source else "reference")
        return collect(solver, directory, smoke=smoke, late=bool(source), lineage=lineage, sync=lambda: sync(root))
    finally:
        if solver is not None and hasattr(solver, "close"):
            solver.close()


def runtime_info():
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())


def verify_checkpoint(root, *, smoke=False, ray_smoke=False):
    runtime(smoke)
    solver = make_solver(smoke=smoke, ray_smoke=ray_smoke)
    try:
        metadata = restore_reference(Path(root)/"reference/checkpoints/six_hours", solver)
        from .states import read_state
        from experiments.fhp.exp22_fhp_critic_objectives.fitting import optimizer_hash
        payload, _ = read_state(Path(root)/"reference/checkpoints/six_hours", pre_fit=False)
        # Native restore validates config/commit; additionally verify restored
        # actor streams and every critical critic/replay-RNG/optimizer component.
        if optimizer_hash(solver._capture_rng_state()) != optimizer_hash(payload["rng"]):
            raise RuntimeError("Training RNG failed checkpoint round-trip")
        if hasattr(solver, "_workers"):
            actors = solver._ray.get([w.continuation_state.remote() for w in solver._workers])
            if optimizer_hash(actors) != optimizer_hash(payload["parallel"]["actors"]):
                raise RuntimeError("Actor state failed checkpoint round-trip")
        for member, saved in zip(solver.q_value_trainer.members, payload["q_ensemble"]["members"], strict=True):
            if (optimizer_hash(member.optimizer.state_dict()) != optimizer_hash(saved["optimizer"])
                    or optimizer_hash(member.target_model.state_dict()) != optimizer_hash(saved["target_model"])):
                raise RuntimeError("Critic checkpoint round-trip failed")
        write_json(Path(root)/"checkpoint_verification.json", dict(status="complete", smoke=smoke,
            stage=metadata["stage"], ray_actors=metadata["ray_actors"], iteration=solver.num_iteration))
    finally:
        if hasattr(solver, "close"):
            solver.close()


def audit_stage(root, stage, *, smoke=False):
    root = Path(root)
    observed_runtime = runtime(smoke)
    directory = root / "audits" / stage
    if directory.exists():
        raise FileExistsError("Use a fresh audit directory; no silent reuse")
    source = root / ("late_reference" if stage == STAGES[-1] else "reference") / "pre_fit" / stage
    started = time.perf_counter()
    solver, metadata = load_audit_solver(source)
    if metadata["smoke"] != smoke:
        raise ValueError("Smoke and production cannot be mixed")
    config = contract(smoke, stage)
    manifest = dict(config=config, stage=stage, seed=0, runtime=observed_runtime,
        provenance=metadata, commit=commit(), code_sha256=fingerprint(Path(__file__).resolve().parents[3]),
        source_load_seconds=time.perf_counter()-started, collection_actors=0)
    write_json(directory / "manifest.json", manifest)
    # Override any unrelated inherited Experiment-22 output destination.
    old_remote = os.environ.pop("EXP22_REMOTE_WORKER", None)
    try:
        audit(solver, directory, config, 0, candidate_diagnostics=FitPanelDiagnostics())
    finally:
        if old_remote is not None:
            os.environ["EXP22_REMOTE_WORKER"] = old_remote
    summary = json.loads((directory/"summary.json").read_text())
    summary.update(experiment_id=24, linked_experiment_id=25, stage=stage,
                   limitation="one-seed development evidence; pre-fit stages; historical late trajectory")
    write_json(directory/"summary.json", summary)
    write_json(directory/"SUCCESS.json", dict(status="complete", seed=0, smoke=smoke,
        artifacts={p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}))
    sync(root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("reference", "late", "audit", "verify-checkpoint", "aggregate", "smoke-local"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ray-smoke", action="store_true")
    p.add_argument("--source-worker", type=Path)
    p.add_argument("--stage", choices=STAGES)
    args = p.parse_args(argv)
    if args.action == "smoke-local" and os.environ.get("EXP24_REMOTE_OUTPUT"):
        p.error("Unset EXP24_REMOTE_OUTPUT for the local-only smoke")
    if args.action == "late" and args.source_worker is None:
        p.error("Late reference requires the verified historical source worker")
    if args.action == "audit" and args.stage is None:
        p.error("Audit requires --stage")
    try:
        if args.action == "smoke-local":
            args.smoke = True
            reference(args.output, smoke=True, ray_smoke=args.ray_smoke)
            verify_checkpoint(args.output, smoke=True, ray_smoke=args.ray_smoke)
            for stage in STAGES[:2]:
                audit_stage(args.output, stage, smoke=True)
                gc.collect()
            from .aggregate import aggregate
            aggregate(args.output, smoke=True, local=True)
        elif args.action in ("reference", "late"):
            reference(args.output, smoke=args.smoke, ray_smoke=args.ray_smoke,
                      source=args.source_worker if args.action == "late" else None)
        elif args.action == "verify-checkpoint":
            verify_checkpoint(args.output, smoke=args.smoke, ray_smoke=args.ray_smoke)
        elif args.action == "audit":
            audit_stage(args.output, args.stage, smoke=args.smoke)
        else:
            from .aggregate import aggregate
            aggregate(args.output, smoke=args.smoke)
        sync(args.output)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(action=args.action, stage=args.stage,
            error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
