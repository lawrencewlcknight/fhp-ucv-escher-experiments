"""Stdlib-only successful cloud-smoke artifact gate; uses gcloud's TLS stack."""
import argparse
import hashlib
import json
import subprocess


def validate(success_raw, summary_raw, commit):
    success, summary = json.loads(success_raw), json.loads(summary_raw)
    stages = ["one_hour", "six_hours", "late_24h_plus_one_iteration"]
    if (success.get("status") != "complete" or success.get("smoke") is not True
            or success.get("local_smoke") is not False or success.get("commit") != commit
            or success.get("experiments") != [24, 25] or success.get("stages") != stages
            or success["artifacts"].get("analysis/summary.json") != hashlib.sha256(summary_raw).hexdigest()):
        raise ValueError("Required smoke artifacts/commit differ")
    if (summary["stages"] != stages or set(summary["audits"]) != set(stages)
            or summary["checkpoint_verification"].get("ray_actors") != 8
            or not all(r["source_state_preserved"] and r["candidates"] == 32 for r in summary["audits"].values())
            or summary["references"]["late_reference"]["stages"][0]["pre_fit"]["lineage"].get("synthetic")
            or summary["references"]["late_reference"]["stages"][0]["pre_fit"]["lineage"].get("source_run_id")
                != "exp10-features-20261001-161740"):
        raise ValueError("Smoke omitted real source, eight-actor restore or candidate curves")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", required=True)
    p.add_argument("--commit", required=True)
    args = p.parse_args()
    data = [subprocess.check_output(["gcloud", "storage", "cat", args.uri.rstrip("/")+"/"+name])
            for name in ("SUCCESS.json", "analysis/summary.json")]
    validate(*data, args.commit)


if __name__ == "__main__":
    main()
