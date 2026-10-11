"""Dependency-free, checksum-verified execution gate; never a scientific significance gate."""
import argparse
import hashlib
import json
import subprocess


def validate(success_bytes, manifest_bytes, summary_bytes, commit, smoke):
    success, manifest, summary = map(json.loads, (success_bytes, manifest_bytes, summary_bytes))
    if (success.get("status") != "complete" or success.get("seed") != 0 or success.get("smoke") != smoke
            or manifest.get("commit") != commit or manifest.get("config", {}).get("experiment_id") != 22
            or manifest.get("config", {}).get("smoke") != smoke
            or not summary.get("source_state_preserved") or summary.get("candidates") != 24
            or manifest.get("provenance", {}).get("synthetic")):
        raise ValueError("Gate requires completed real-source seed 0 with identical commit/mode")
    for name, value in (("manifest.json", manifest_bytes), ("summary.json", summary_bytes)):
        if hashlib.sha256(value).hexdigest() != success.get("artifacts", {}).get(name):
            raise ValueError("Gate artifact checksum mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--smoke", choices=("true", "false"), required=True)
    args = p.parse_args()
    raw = [subprocess.check_output(["gcloud", "storage", "cat", args.uri+"/"+name])
           for name in ("SUCCESS.json", "manifest.json", "summary.json")]
    validate(*raw, args.commit, args.smoke == "true")
    print("Execution gate passed; inspect scientific results and costs before choosing to continue.")


if __name__ == "__main__":
    main()
