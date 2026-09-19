# Parallel UCV-ESCHER provenance

`parallel_solver.py` and `parallel_utils.py` are derived from the synchronous
Ray implementation in
[`lawrencewlcknight/leduc-poker-escher-architecture-experiments`](https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments),
commit `c09bbe5a9adc6b495c9a0f74c2993dcb682ea754` (`Add parallel
UCV-ESCHER equivalence experiment`).

The retained architecture has one authoritative driver learner, persistent
CPU-only Ray traversal actors, frozen inference snapshots, exact traversal
budget partitioning, driver-side replay/reservoir decisions, deterministic
merge order, disjoint cross-fitted Q replay, and concurrent updates only for
independent Q-fold/calibration learners.

FHP-specific integration replaces Leduc game loading with the canonical
OpenSpiel `universal_poker` definition, uses compact worker scratch storage
sized from FHP's maximum game length, records parallel timing/payload metrics,
and divides each frozen traverser phase into bounded synchronized dispatches.
The bounded dispatches allow the inherited archived Experiment 1 training-time mechanism
to save at the first safe merge boundary after 6 and 12 hours and to stop after
the final checkpoint. They do not introduce asynchronous gradients or multiply
the configured traversal budget.
