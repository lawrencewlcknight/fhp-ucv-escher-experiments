# Experiment 21 — adversarial policy improvement with preservation

Starting from **Experiment 10's three 24-hour policies**, compare unchanged
policies, compute-matched additional grouped fitting, adversarial payoff
fine-tuning, and payoff fine-tuning with replay/blueprint preservation. Student
network size and the lossless hand/board representation remain fixed.

Implementation lives in the sibling **fhp-evaluation-suite**, under
`fhp_evaluation/adversarial/`, with the complete frozen protocol and step-by-step
instructions in `docs/exp21_adversarial_preservation.md`. This repository provides
the root launcher `gcp/run_exp21_adversarial_preservation.sh` and native encoders.
Keep both checkouts up to date. Source bundles capture both repositories.

No original UCV training is repeated. A separate attacker-strength qualification
must pass before repair is allowed. Three repair rounds retain previous opponents;
independent evaluation includes fresh learned attackers, head-to-head play, frozen
UCV reference policies, rule agents, LBR and the specialised full-flop responder.
VR-Deep/SD-CFR cross-family evaluation is a separate follow-up, not part of this
pilot's runtime budget. Results cannot establish exact Nash distance.

## Cloud smoke (from this repository root)

Assume `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` are already set:

```bash
export RUN_ID="exp21-adv-smoke-$(date -u +%Y%m%d-%H%M%S)"
export SMOKE_RUN_ID="$RUN_ID"
bash gcp/run_exp21_adversarial_preservation.sh prepare-smoke
bash gcp/run_exp21_adversarial_preservation.sh qualify-smoke
```

Wait for each preceding job to show `SUCCEEDED` before submitting the next:

```bash
bash gcp/run_exp21_adversarial_preservation.sh train-smoke
# Wait for SUCCEEDED.
bash gcp/run_exp21_adversarial_preservation.sh evaluate-smoke
# Wait for SUCCEEDED.
bash gcp/run_exp21_adversarial_preservation.sh aggregate-smoke
```

The smoke tests execution only; it cannot pass the scientific qualification gate.
It reads the real historical seed-0 training state (~5.4 GB), extracts replay once,
then reuses a much smaller, checksummed policy-only cache.

## Full pilot

Preserve `SMOKE_RUN_ID`; all four smoke jobs must succeed with the same source
bundle. Changes to bundled code require repeating smoke.

```bash
export RUN_ID="exp21-adv-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp21_adversarial_preservation.sh prepare
bash gcp/run_exp21_adversarial_preservation.sh qualify
bash gcp/run_exp21_adversarial_preservation.sh status
```

Review qualification results. If all three attacker-strength gates pass:

```bash
bash gcp/run_exp21_adversarial_preservation.sh train
# Wait for SUCCEEDED before submitting evaluation.
bash gcp/run_exp21_adversarial_preservation.sh evaluate
# Wait for SUCCEEDED before submitting aggregation.
bash gcp/run_exp21_adversarial_preservation.sh aggregate
```

`prepare` is offline. Every other command except `status` submits one paid Batch
job, never its successors. Do not rerun an inconclusive gate until it passes.
Qualification/training use three `n2-standard-16` VMs; evaluation uses twelve
`n2-standard-4` VMs. Both arrays require 48 regional N2 vCPUs. Watchdogs, resource
logs, periodic uploads, final diagnostics and zero automatic retries bound costs.

Allow roughly 12–24 elapsed hours across the full sequence, subject to
profiling and VM availability; the limits are safeguards, not predictions.
Full outputs retain source identities, final native-playable policies, intermediate
round policies/attackers, optimizer metadata and metrics. These are post-trained
policies, **not resumable full UCV solver states**. Aggregate analysis lives at
`$BUCKET/$RUN_ID/analysis/summary.json`.
