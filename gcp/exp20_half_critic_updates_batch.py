#!/usr/bin/env python3
"""Experiment 20 training jobs; preserve the proven Experiment 10 lifecycle."""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp import exp10_hand_board_features_batch as training
from gcp import retrospective_exp7_exp8_evaluation_batch as evaluation

REPO_URL = training.REPO_URL
MODULE = "experiments.fhp.exp20_fhp_half_critic_updates.run"
EVALUATION_MODULE = "experiments.fhp.exp20_fhp_half_critic_updates.evaluate"
BASELINE_RUN_ID = "exp10-features-20261001-161740"


def validate(args):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,33}[a-z0-9]", args.run_id):
        raise ValueError("RUN_ID must be 2–35 lowercase letters, digits or hyphens")
    if args.run_id == BASELINE_RUN_ID:
        raise ValueError("Never overwrite the historical baseline")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("REPO_REF must be a full pushed commit SHA")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", args.bucket_root):
        raise ValueError("Expected a gs:// bucket without a path")
    if not 1 <= args.parallelism <= 3:
        raise ValueError("Parallelism must be 1–3")
    if not 1 <= args.max_hours <= 72:
        raise ValueError("Evaluation wall limit must be 1–72 hours")
    if args.kind == "controller" and (not args.project_id or not args.region):
        raise ValueError("Controller requires project and region")
    if getattr(args, "source_run_id", "") == BASELINE_RUN_ID:
        raise ValueError("Experiment 20 must not warm-start from Experiment 10")
    if args.kind == "evaluate" and (args.total_hours != 24 or args.source_run_id):
        raise ValueError("This evaluator compares original 24h runs, not extensions")


def _replace(text, old, new):
    if old not in text:
        raise RuntimeError(f"Inherited Batch template changed: {old}")
    return text.replace(old, new)


def build_job(args):
    validate(args)
    if args.kind == "evaluate":
        inherited = copy(args)
        inherited.exp7_run_id, inherited.exp8_run_id = args.run_id, BASELINE_RUN_ID
        inherited.smoke_only = False
        # Separate evaluation output subtree; never write into either input.
        script = evaluation._script(inherited)
        script = _replace(script, "fhp-eval78", "fhp-exp20-eval")
        script = _replace(script, "experiments.fhp.retrospective_exp7_exp8_evaluation.run", EVALUATION_MODULE)
        script = _replace(script, "--exp7-run", "--exp20-run")
        script = _replace(script, "--exp8-run", "--exp10-run")
        script = _replace(script, '"$BUCKET_ROOT/$RUN_ID"', '"$BUCKET_ROOT/$RUN_ID/evaluation"')
        script = script.replace(".*training_states.*|.*reservoir", ".*training_states.*|.*continuation_inputs.*|.*reservoir")
        job = {
            "taskGroups": [{"taskCount": 1, "parallelism": 1, "taskCountPerNode": 1,
                           "taskSpec": {"runnables": [{"script": {"text": script}}],
                                        "computeResource": {"cpuMilli": 16000, "memoryMib": 62000},
                                        "maxRetryCount": 0, "maxRunDuration": f"{args.max_hours * 3600}s"}}],
            "allocationPolicy": {"serviceAccount": {"email": args.service_account},
                                 "instances": [{"policy": {"machineType": "n2-standard-16",
                                                            "provisioningModel": "STANDARD",
                                                            "bootDisk": {"sizeGb": 100, "type": "pd-balanced"}}}]},
            "logsPolicy": {"destination": "CLOUD_LOGGING"},
        }
    else:
        inherited = copy(args)
        inherited.controller_action = "orchestrate"
        job = training.build_job(inherited)
        spec = job["taskGroups"][0]["taskSpec"]
        script = spec["runnables"][0]["script"]["text"]
        for old, new in (
            ("exp10_fhp_hand_board_features", "exp20_fhp_half_critic_updates"),
            ("exp10_hand_board_features", "exp20_half_critic_updates"),
            ("hand_board_cached_parallel_ucv_escher", "half_critic_hand_board_ucv_escher"),
            ("EXP10", "EXP20"), ("exp10-", "exp20-"), ("Experiment 10", "Experiment 20"),
        ):
            script = script.replace(old, new)
        spec["environment"]["variables"] = {
            k.replace("EXP10", "EXP20"): v for k, v in spec["environment"]["variables"].items()}
        if args.kind != "controller":
            script = _replace(script, "uv python install 3.11\n", "uv python install 3.11.16\n")
            script = _replace(script, "uv venv --python 3.11 ", "uv venv --python 3.11.16 ")
            check = """python - <<'EXP20_RUNTIME'
import platform, subprocess, numpy, torch, ray
from importlib.metadata import version
assert platform.python_version() == '3.11.16'
assert (numpy.__version__, str(torch.__version__), ray.__version__) == ('1.26.4', '2.7.0+cpu', '2.51.2')
assert version('open_spiel') == '1.6.3'
EXP20_RUNTIME
"""
            check = check.replace("EXP20_RUNTIME\n", f"assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == '{args.repo_ref}'\nEXP20_RUNTIME\n")
            if args.kind == "train":
                check = check.replace("EXP20_RUNTIME\n", "assert (torch.get_num_threads(), torch.get_num_interop_threads()) == (8, 8)\nEXP20_RUNTIME\n")
            script = _replace(script, 'cd "$REPOSITORY"\n', 'cd "$REPOSITORY"\n' + check)
        spec["runnables"][0]["script"]["text"] = script
        spec["maxRetryCount"] = 0
        if args.kind == "controller":
            waves = (3 + args.parallelism - 1) // args.parallelism
            training_hours = 72 if args.source_run_id else 36
            spec["maxRunDuration"] = f"{(training_hours * waves + 12) * 3600}s"
    job["labels"] = {"experiment": "exp20-fhp-half-critic-updates", "stage": args.kind}
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("controller", "smoke", "train", "aggregate", "evaluate"))
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--total-hours", type=int, default=24)
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2) + "\n")


if __name__ == "__main__":
    main()
