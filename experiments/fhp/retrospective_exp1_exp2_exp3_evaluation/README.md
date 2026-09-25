# Retrospective evaluation of FHP Experiments 1, 2 and 3

This is an evaluation extension, not another training experiment. It evaluates
the frozen Experiment 1 checkpoints and combines the result with the completed,
validated Experiment 2/3 retrospective evaluation.

The Experiment 1 evaluation deliberately retains the earlier protocol:

- seeds 0, 1 and 2 at 6, 12, 18 and 24 hours;
- 10,000 duplicate deal pairs against each of five fixed rule agents;
- 1,000 LBR duplicate deal pairs with 4,096 preflop rollouts;
- 50,000 duplicate deal pairs for temporal and direct cross-play;
- common deal and action seeds, including across all three same-time pairwise
  comparisons; and
- uncertainty across independent training-seed means.

Experiment 2/3 measurements are imported rather than recomputed. Before any
new evaluation starts, the runner checks the reference protocol and confirms
that every Experiment 2/3 checkpoint hash and node count matches the frozen
reference manifest.

The output contains unified rule-agent, LBR and temporal tables and charts for
all three experiments; direct Experiment 1 versus 2, Experiment 1 versus 3,
and Experiment 2 versus 3 cross-play; and approximately node-matched cross-play
for all three pairs. LBR is not exact exploitability.

One `n2-standard-8` VM uses eight evaluation workers. A real-checkpoint smoke
test runs first. The job has an 18-hour hard ceiling and stops immediately when
finished; only the new Experiment 1 diagnostics and missing cross-play are
computed.
