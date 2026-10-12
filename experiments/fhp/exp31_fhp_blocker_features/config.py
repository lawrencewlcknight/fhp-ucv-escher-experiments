"""Frozen Experiment 31 contract; all learning settings inherit Experiment 10."""
from experiments.fhp.strategic_feature_suite.config import get_experiment
EXPERIMENT = get_experiment(31)
globals().update(vars(EXPERIMENT))
