# Experiment 34 — exact terminal-call safeguard

Evaluation-only experiment on all three frozen Experiment 16 48-hour policies.
No retraining: transfer fold probability to call only at a terminal call/fold
decision where exact enumeration certifies calling is no worse against every
compatible opponent holding. All other action probabilities remain unchanged.

One million duplicate pairs per seed against its own unchanged version, plus
rule agents, predecessor checkpoints, a frozen neural panel, target-aware LBR and
full-flop response cross-probes, and the complete development strategic audit.
The evaluator records rare-event activation, sampled and analytically integrated
payoff differences, cold/cached verifier latency and separate training-seed
uncertainty. Checkpoint SHA-256 values are pinned in the shared suite.

Implementation/protocol: `../../fhp-evaluation-suite/docs/exp34_terminal_call_safeguard.md`
relative to the UCV repository root. Launch with
`bash gcp/run_exp34_terminal_call_safeguard.sh run`; `EXP34_MODE=smoke` is default,
and `EXP34_MODE=full` explicitly selects the large study. One n2-standard-16 VM;
full-mode 48h hard cap and a pre-gameplay timing gate. Nothing is launched by
creating or testing this experiment. No new training-state artifacts.
