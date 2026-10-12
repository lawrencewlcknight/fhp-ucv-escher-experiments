"""Frozen-policy evaluation against archived Experiment 10 only."""
from .config import EXPERIMENT
from experiments.fhp.strategic_feature_suite.evaluate import main as _main

def main(argv=None):
    return _main(argv, experiment=EXPERIMENT)

if __name__ == "__main__":
    main()
