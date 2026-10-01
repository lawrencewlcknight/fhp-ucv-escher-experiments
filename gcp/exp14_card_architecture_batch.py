#!/usr/bin/env python3
"""Reuse the capacity audit's staging/retention; change only experiment identity/caps."""

import exp13_policy_capacity_batch as capacity


def build_job(args):
    job = capacity.build_job(args)
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
    spec["runnables"][0]["script"]["text"] = text
    waves = (3 + args.parallelism - 1) // args.parallelism
    spec["maxRunDuration"] = {"controller": f"{2 * 172800 * waves + 21600}s",
        "smoke": "14400s", "screen": "172800s", "select": "3600s",
        "train": "172800s", "aggregate": "3600s"}[args.kind]
    job["labels"] = {"experiment": "exp14-fhp-cards", "stage": args.kind}
    return job


if __name__ == "__main__":
    capacity.main(builder=build_job)
