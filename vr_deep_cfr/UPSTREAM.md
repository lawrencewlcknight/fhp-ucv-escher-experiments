# Upstream provenance

The algorithm implementation in `solver.py` and `variants.py` was adapted from
[`rpSebastian/DeepPDCFR`](https://github.com/rpSebastian/DeepPDCFR), commit
`9f156c9fcdac7f8c9bd0debf94c9432d222858d3`, retrieved on 16 July 2026.

The upstream repository did not contain a licence file at retrieval time. This
copy is retained for research reproducibility and attribution; redistribution
rights should be confirmed with the authors before publishing this repository.

Integration changes include direct OpenSpiel FHP loading, structured metrics,
training-time stopping, deterministic checkpoint isolation,
and an obvious optimiser reset correction in `VRPDCFRPlusRegretTrainer.reset`
(the upstream reset incorrectly attached the immediate-regret optimiser to the
cumulative-regret model parameters). The integration also corrects the swapped
`reinitialize_imm_regret_networks` and `use_regret_matching_argmax` positional
arguments in the upstream VR-DeepPDCFR+ trainer construction. Finally, it makes
the paper's immediate-regret reinitialisation independent of cumulative-regret
reinitialisation (the released control flow otherwise never applies the former
when the latter is false) and tracks circular-buffer occupancy separately from
the wrapped write index.
