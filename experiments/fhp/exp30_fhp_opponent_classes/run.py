"""Training, smoke, aggregation and continuation CLI for Experiment 30."""
from .config import EXPERIMENT
from experiments.fhp.strategic_feature_suite.run import main as _main

def main(argv=None):
    return _main(argv, experiment=EXPERIMENT)

if __name__ == "__main__":
    main()
