"""Experiment 31 task names and validated aggregation."""
from functools import partial
from .config import EXPERIMENT
from experiments.fhp.strategic_feature_suite import aggregate as shared

task_name = partial(shared.task_name, experiment=EXPERIMENT)
aggregate_workers = partial(shared.aggregate_workers, experiment=EXPERIMENT)
