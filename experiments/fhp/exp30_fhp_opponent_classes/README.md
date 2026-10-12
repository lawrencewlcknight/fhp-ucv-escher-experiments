# Experiment 30: A+B: relative strength and opponent-class profiles

Inherits Experiment 10; trains three seeds for 24 active hours on
`n2-standard-16`, with eight collection actors per learner. Automatic final
policy evaluation uses archived Experiment 10, without baseline retraining.
Retains policies at 6/12/18/24h and one final full resumable state per seed.

Use `bash gcp/run_exp30_opponent_classes.sh run` from the repository root after
setting the required environment and pushing the pinned source commit.
See the [shared specification and launch guide](../strategic_feature_suite/README.md)
for feature definitions, validation, caveats, recovery and continuation.
