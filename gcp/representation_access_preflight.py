#!/usr/bin/env python3
"""Read-only, local access checks before allocating an representation-comparison VM.

Inspect explicit bucket IAM grants for the runner, not permissions of the
submitting user. This intentionally needs neither impersonation nor a new API.
It is a conservative allow-policy check, not an effective-access simulator:
inherited/custom/conditional grants and organisation-level denies are not resolved.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


READ_ROLES = {"roles/storage.objectViewer", "roles/storage.objectUser",
              "roles/storage.objectAdmin", "roles/storage.admin"}
WRITE_ROLES = {"roles/storage.objectUser", "roles/storage.objectAdmin", "roles/storage.admin"}


def parse_object_list(output):
    # gcloud storage ls emits one JSON array per URL argument, not a single
    # array for the entire request. Parse that JSON stream without accepting
    # warning text, malformed data or an unexpected object schema.
    decoder, rows, remaining = json.JSONDecoder(), [], output.strip()
    while remaining:
        value, end = decoder.raw_decode(remaining)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError("Unexpected gcloud storage ls JSON format")
        rows.extend(value)
        remaining = remaining[end:].lstrip()
    return rows


def cloud_json(arguments, project):
    result = subprocess.run(["gcloud", *arguments, "--project", project, "--quiet"],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Cloud preflight failed: {' '.join(arguments)}\n{result.stderr.strip()}")
    if arguments[:2] == ["storage", "ls"]:
        return parse_object_list(result.stdout)
    return json.loads(result.stdout)


def require_grant(policy, bucket, service_account, *, write=False):
    member = "serviceAccount:" + service_account
    roles = {binding.get("role") for binding in policy.get("bindings", [])
             if member in binding.get("members", []) and not binding.get("condition")}
    acceptable = WRITE_ROLES if write else READ_ROLES
    if not roles.intersection(acceptable):
        recommended = "roles/storage.objectAdmin" if write else "roles/storage.objectViewer"
        raise ValueError(
            f"Cannot verify {'read/write' if write else 'read/list'} access for VM runner "
            f"{service_account} on {bucket}. Need a direct, unconditional {recommended} "
            "grant (or a supported stronger role). Your local login's access is not proof "
            "of runner access. No IAM changes or job submission were made. "
            "If access relies on inherited, custom or conditional policies, have an "
            "administrator verify that configuration before adapting this check.")
    return sorted(roles.intersection(acceptable))


def metadata_uris(builder, args):
    return builder.metadata_uris(args)


def check(builder, args, project, cloud=cloud_json):
    builder.validate(args)
    receipt = dict(status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
                   service_account=args.service_account, repo_ref=args.repo_ref,
                   method="direct_unconditional_bucket_allow_grants",
                   effective_access_verified=False, bucket_grants=[], source_metadata=[])
    # Do not infer VM access from a successful local ls/cat, or accept a grant
    # belonging to another runner. Check the output bucket too: uploads overwrite
    # progress markers and therefore require more than objectCreator.
    buckets = sorted({args.bucket, getattr(args, args.algorithm+"_bucket")})
    for bucket in buckets:
        policy = cloud(["storage", "buckets", "get-iam-policy", bucket, "--format=json"], project)
        roles = require_grant(policy, bucket, args.service_account, write=bucket == args.bucket)
        receipt["bucket_grants"].append(dict(bucket=bucket, roles=roles, etag=policy.get("etag")))
        print(f"Verified runner bucket grant: {bucket} ({', '.join(roles)})", flush=True)
    # Source existence checks still use the submitting login; they are deliberately
    # separate from the runner IAM checks. No models/replay are downloaded.
    for key, uris in metadata_uris(builder, args):
        records = cloud(["storage", "ls", "--json", *uris], project)
        found = {row.get("url", "").split("#", 1)[0] for row in records
                 if row.get("type") == "cloud_object"}
        missing = set(uris) - found
        if missing:
            raise ValueError("Missing required source metadata: " + ", ".join(sorted(missing)))
        receipt["source_metadata"].append(dict(source=key, objects=len(uris)))
        print(f"Verified all three source seeds: {key} ({len(uris)} metadata files)", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    # Use the pinned job builder's source IDs/configuration even when resuming an
    # older worker commit. The local preflight never changes scoring code.
    spec = importlib.util.spec_from_file_location("representation_pinned_builder", options.builder)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    args = builder.parser().parse_args(remaining)
    try:
        receipt = check(builder, args, options.project)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"{error}\nPreflight stopped before VM submission.", file=sys.stderr)
        return 1
    options.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Access preflight passed; receipt: {options.receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
