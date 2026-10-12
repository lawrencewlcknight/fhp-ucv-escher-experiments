"""Frozen Experiment 32 contract; all learning settings inherit Experiment 10."""
from experiments.fhp.strategic_feature_suite.config import get_experiment
EXPERIMENT = get_experiment(32)
globals().update(vars(EXPERIMENT))
