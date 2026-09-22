# Retrospective evaluation of FHP Experiments 2 and 3

This is an evaluation extension to Experiments 2 and 3, not a new training
experiment. It reloads the frozen policies at 6, 12, 18 and 24 hours for all
three seeds and evaluates them using the shared FHP evaluation contract.

The production analysis reports:

- duplicate matches against the five corrected published rule agents;
- the Local Best Response lower bound;
- all-pairs temporal checkpoint cross-play within each configuration;
- matched-seed direct Experiment 2 versus Experiment 3 cross-play;
- uncertainty across independently trained seed means; and
- CSV tables, a concise Markdown summary, and wall-clock and node-budget
  charts.

LBR is not exact exploitability. Deal-level confidence intervals describe
match-sampling uncertainty for a fixed policy. The aggregate tables instead
use the three independent training-seed means as the inferential sample.

The GCP launcher uses one `n2-standard-8` VM and eight worker processes. A
small real-checkpoint smoke test runs on the cloud VM immediately before the
production analysis.
