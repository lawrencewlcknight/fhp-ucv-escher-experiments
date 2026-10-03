#!/usr/bin/env python3
"""Card-audit jobs with a pinned runtime and provenance-preserving recovery."""

import argparse
from copy import copy
import json
from pathlib import Path
import re

import exp13_policy_capacity_batch as capacity


def runtime_gate(args):
    """Embed launcher code without altering the frozen scientific checkout."""
    seed = ' --seed "$SEED"' if args.kind in ("smoke", "screen", "train") else ""
    command = (
        'python - --bucket-root "$BUCKET_ROOT" --run-id "$RUN_ID" '
        '--source-run-id "$EXP2_RUN_ID" '
        f'--audit-ref {capacity.base._q(args.audit_ref)} '
        f'--python-version {capacity.base._q(args.python_version)} --stage {args.kind}'
        f'{seed} --repository "$REPOSITORY" '
        '--failure-file "$DIAGNOSTIC_DIR/failure.json"'
    )
    source = Path(__file__).with_name("exp14_runtime_preflight.py").read_text()
    return command + " <<'EXP14_RUNTIME_GATE'\n" + source + "\nEXP14_RUNTIME_GATE\n"


def build_job(args):
    args = copy(args)
    args.audit_ref = getattr(args, "audit_ref", None) or args.repo_ref
    args.python_version = getattr(args, "python_version", "3.11.16")
    if not re.fullmatch(r"3\.11\.\d+", args.python_version):
        raise ValueError("Python must be pinned to a full 3.11 patch version")
    if args.audit_ref != args.repo_ref and args.controller_action != "orchestrate-recover":
        raise ValueError("Separate launcher/audit commits are only allowed for explicit recovery")
    checkout = copy(args)
    if args.kind != "controller":
        checkout.repo_ref = args.audit_ref
    job = capacity.build_job(checkout)
    spec = job["taskGroups"][0]["taskSpec"]
    text = spec["runnables"][0]["script"]["text"]
    for old, new in (
        ("exp13_fhp_policy_capacity", "exp14_fhp_card_architecture"),
        ("run_exp13_policy_capacity.sh", "run_exp14_card_architecture.sh"),
        ("exp13-capacity", "exp14-cards"), ("exp13-controller", "exp14-controller"),
        ("EXP13_REMOTE", "EXP14_REMOTE"),
    ):
        text = text.replace(old, new)
    if "exp13" in text or "EXP13" in text:
        raise ValueError("Experiment 14 job contains a stale Experiment 13 route")
    if args.kind == "controller":
        exports = (
            f"export EXP14_AUDIT_REF={capacity.base._q(args.audit_ref)}\n"
            f"export EXP14_PYTHON_VERSION={capacity.base._q(args.python_version)}\n"
            f"export RESUME_TAG={capacity.base._q(getattr(args, 'resume_tag', ''))}\n"
        )
        text = text.replace("exec bash gcp/run_exp14_card_architecture.sh", exports +
                            "exec bash gcp/run_exp14_card_architecture.sh")
    else:
        text = text.replace("uv python install 3.11\n", f"uv python install {args.python_version}\n")
        text = text.replace("uv venv --python 3.11 ", f"uv venv --python {args.python_version} ")
        gate = runtime_gate(args)
        if args.kind in ("smoke", "screen", "train"):
            # The resource monitor/traps are already installed at this point.
            text = text.replace('SOURCE="$WORK_ROOT/source"', gate + 'SOURCE="$WORK_ROOT/source"')
        else:
            # Selection and aggregation previously had no diagnostics wrapper.
            start = text.index('mkdir -p "$OUTPUT_ROOT/workers"')
            wrapped = capacity.base._diagnostic_wrapper(
                gate + text[start:], f"$OUTPUT_ROOT/diagnostics/{args.kind}",
                f"$BUCKET_ROOT/$RUN_ID/diagnostics/{args.kind}", 30000)
            text = text[:start] + wrapped + "\n"
    spec["runnables"][0]["script"]["text"] = text
    waves = (3 + args.parallelism - 1) // args.parallelism
    spec["maxRunDuration"] = {"controller": f"{2 * 172800 * waves + 21600}s",
        "smoke": "14400s", "screen": "172800s", "select": "3600s",
        "train": "172800s", "aggregate": "3600s"}[args.kind]
    job["labels"] = {"experiment": "exp14-fhp-cards", "stage": args.kind}
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=capacity.KINDS)
    parser.add_argument("--output", required=True, type=Path)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "source-run-id", "project-id", "region"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--audit-ref", help="Original scientific commit; defaults to repo-ref")
    parser.add_argument("--python-version", default="3.11.16")
    parser.add_argument("--repo-url", default=capacity.base.REPO_URL)
    parser.add_argument("--parallelism", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--resume-tag", default="")
    parser.add_argument("--controller-action", default="orchestrate",
                        choices=("orchestrate", "orchestrate-resume", "orchestrate-recover"))
    args = parser.parse_args()
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2) + "\n")


if __name__ == "__main__":
    main()
