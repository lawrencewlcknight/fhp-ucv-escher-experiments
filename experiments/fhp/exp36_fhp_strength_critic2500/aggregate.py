"""Aggregate three seeds after verifying actual critic budgets and checkpoints."""
from functools import partial
from experiments.fhp.strategic_feature_suite.aggregate import aggregate_workers as _aggregate, task_name as _task_name
from .config import EXPERIMENT

aggregate_workers = partial(_aggregate, experiment=EXPERIMENT)
task_name = partial(_task_name, experiment=EXPERIMENT)
