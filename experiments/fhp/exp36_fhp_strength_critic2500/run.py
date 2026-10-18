"""Shared native Ray training, cache, feature, capacity and continuation checks."""
from experiments.fhp.strategic_feature_suite.run import main as shared_main
from .config import EXPERIMENT


def main(argv=None):
    return shared_main(argv, experiment=EXPERIMENT)


if __name__ == "__main__":
    main()
