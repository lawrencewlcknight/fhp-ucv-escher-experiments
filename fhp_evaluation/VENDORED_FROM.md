# Shared FHP evaluation snapshot

This package is a byte-for-byte source snapshot of the locally validated
`fhp-evaluation-suite/fhp_evaluation` package as of 2026-09-22. It is vendored
so that cloud evaluation jobs can reconstruct the evaluator from the pinned
`fhp-ucv-escher-experiments` commit without depending on files from the
submitting laptop.

The original suite was validated jointly against UCV-ESCHER, VR-Deep and Deep
CFR checkpoints. Its provenance includes the five rule agents and pre-flop
lookup from the released DeepPDCFR artefact; the LooseAggressive thresholds are
corrected to the intended increasing order `(-300, -100)`. Local Best Response
is reported only as a lower bound on best-response value, never as exact
exploitability.
