"""Read-only compatibility gate, embedded outside the frozen audit source tree.

Keep this standalone: recovery executes the ORIGINAL audit checkout, not this
launcher's commit. Never rewrite manifests, selection locks or their hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

EXPERIMENT = "exp14_fhp_card_architecture"
IDENTITY_FIELDS = ("audit_commit", "audit_source_sha256", "python", "torch", "numpy")


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def differences(expected, actual, prefix=""):
    """Return useful field-level evidence, including full interpreter builds."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = {}
        for key in sorted(expected.keys() | actual.keys()):
            field = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                result[field] = {"saved": expected.get(key), "current": actual.get(key),
                                 "missing_from": "saved" if key not in expected else "current"}
            else:
                result.update(differences(expected[key], actual[key], field))
        return result
    return {} if expected == actual else {prefix: {"saved": expected, "current": actual}}


def require_equal(expected, actual, description):
    mismatch = differences(expected, actual)
    if mismatch:
        raise ValueError(f"{description}: " + json.dumps(mismatch, sort_keys=True))


def fetch_json(uri, *, optional=False):
    result = subprocess.run(["gcloud", "storage", "cat", uri], text=True, capture_output=True)
    if result.returncode:
        # Authentication, network and permission errors must not masquerade as
        # a fresh run. Only the CLI's explicit missing-object response is optional.
        if optional and "matched no objects" in result.stderr.lower():
            return None
        raise RuntimeError(f"Cannot read {uri}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def audit_source_hash(repository):
    """Mirror the original prepare() fingerprint without importing replay data."""
    repository = Path(repository)
    base = repository / "experiments/fhp/exp13_fhp_policy_capacity"
    paths = sorted(base.glob("*.py"))
    paths += sorted((base.parent / "exp4_fhp_average_policy_audit").glob("*.py"))
    paths += sorted((repository / "fhp_escher").rglob("*.py"))
    paths += sorted((repository / "fhp_evaluation").rglob("*.py"))
    paths += [base.parent / "retrospective_exp2_exp3_evaluation/run.py"]
    paths += sorted((base.parent / EXPERIMENT).glob("*.py"))
    return hashlib.sha256(b"".join(
        path.parent.name.encode() + path.name.encode() + path.read_bytes() for path in paths
    )).hexdigest()


def current_identity(repository, smoke=False):
    # No load_source/group_replay/torch.load call: this gate precedes the
    # multi-gigabyte source download and unpickling.
    sys.path.insert(0, str(Path(repository).resolve()))
    import numpy
    import torch
    from experiments.fhp.exp14_fhp_card_architecture.config import contract
    return {
        "audit_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
        "audit_source_sha256": audit_source_hash(repository),
        "python": sys.version, "torch": str(torch.__version__), "numpy": numpy.__version__,
        "config": contract(smoke),
    }


def verify_selection(selection, manifests):
    identity = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if (selection.get("selection_sha256") != object_hash(identity)
            or selection.get("selection_uses_test_or_gameplay") is not False):
        raise ValueError("Invalid or modified locked selection")
    for seed, manifest in manifests.items():
        require_equal(manifest["config"], selection["config"], "Selected configuration mismatch")
        require_equal(selection["source_manifest_hashes"].get(str(seed)), object_hash(manifest),
                      f"Locked source manifest differs for seed {seed}")


def verify_source(args, seed, manifest, fetch):
    source = (f"{args.bucket_root}/{args.source_run_id}/workers/"
              f"task_{seed:03d}_lossless_structured_ucv_escher_seed_{seed}")
    run = fetch(source + "/run_manifest.json")
    rows = fetch(source + "/checkpoint_manifest.json")
    final = [row for row in rows if row["checkpoint_id"] == "time_24h"]
    if len(final) != 1:
        raise ValueError(f"Source seed {seed} does not have one 24-hour endpoint")
    row = final[0]
    expected = {
        "seed": seed, "checkpoint_id": "time_24h",
        "source_commit": run["repository_commit"],
        "source_policy_sha256": row["sha256"],
        "source_state_sha256": row["training_state_sha256"],
        "outer_iteration": row["outer_iteration"], "nodes_touched": row["nodes_touched"],
    }
    if (run.get("seed") != seed or run.get("experiment_name") != "exp2_fhp_lossless_structured_ucv"
            or row["checkpoint_target_seconds"] != 86400):
        raise ValueError(f"Unexpected source experiment/seed/checkpoint: {source}")
    require_equal(manifest["provenance"], expected, f"Source provenance differs for seed {seed}")


def check(args, *, fetch=fetch_json, observed=None):
    if not re.fullmatch(r"[0-9a-f]{40}", args.audit_ref):
        raise ValueError("audit-ref must be a full commit SHA")
    if not re.fullmatch(r"3\.11\.\d+", args.python_version):
        raise ValueError("Pin a complete Python patch version, e.g. 3.11.16")
    smoke = args.stage == "smoke"
    if args.repository is not None and observed is None:
        observed = current_identity(args.repository, smoke)
    if observed is not None:
        require_equal(args.audit_ref, observed["audit_commit"], "Wrong audit checkout")
        require_equal(args.python_version, observed["python"].split()[0], "Wrong Python patch")
    prefix = f"{args.bucket_root}/{args.run_id}" + ("/smoke" if smoke else "")
    seeds = [args.seed] if args.seed is not None else [0, 1, 2]
    require_screen = args.stage in ("train", "select", "aggregate", "recover")
    manifests = {}
    for seed in seeds:
        worker = f"{prefix}/workers/seed_{seed}"
        manifest = fetch(worker + "/manifest.json", optional=not require_screen)
        if manifest is None:
            continue
        expected = {"experiment_id": 14, "experiment_name": EXPERIMENT, "smoke": smoke}
        require_equal(expected, {k: manifest["config"].get(k) for k in expected},
                      f"Wrong source audit for seed {seed}")
        require_equal(args.audit_ref, manifest["audit_commit"], "Saved audit commit differs")
        require_equal(args.python_version, manifest["python"].split()[0], "Saved Python patch differs")
        if observed is not None:
            require_equal({k: manifest[k] for k in (*IDENTITY_FIELDS, "config")}, observed,
                          f"Runtime/code/configuration mismatch for seed {seed}")
        if manifests:
            first = next(iter(manifests.values()))
            require_equal({k: first[k] for k in (*IDENTITY_FIELDS, "config")},
                          {k: manifest[k] for k in (*IDENTITY_FIELDS, "config")},
                          "Screening seeds used different runtimes/code")
        verify_source(args, seed, manifest, fetch)
        if require_screen:
            require_equal({"seed": seed, "smoke": False, "status": "screen_complete"},
                          fetch(worker + "/SCREEN_SUCCESS.json"), "Incomplete screening")
        manifests[seed] = manifest
    if args.stage in ("train", "aggregate", "recover"):
        selection = fetch(prefix + "/selection.json")
        verify_selection(selection, manifests)
    return {"status": "compatible", "stage": args.stage, "audit_commit": args.audit_ref,
            "python_version": args.python_version, "checked_seeds": sorted(manifests)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket-root", "run-id", "source-run-id", "audit-ref", "python-version"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=("smoke", "screen", "train", "select", "aggregate", "recover"),
                        required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--failure-file", type=Path)
    args = parser.parse_args(argv)
    args.bucket_root = args.bucket_root.rstrip("/")
    try:
        result = check(args)
    except Exception as error:
        failure = {"status": "failed", "stage": "runtime_preflight",
                   "error_type": type(error).__name__, "error": str(error),
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        if args.failure_file:
            args.failure_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.failure_file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(failure, indent=2) + "\n")
            temporary.replace(args.failure_file)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
