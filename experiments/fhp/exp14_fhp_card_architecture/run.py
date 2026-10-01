"""Run the card audit using the validated frozen-replay workflow, not new CFR."""

from experiments.fhp.exp13_fhp_policy_capacity.run import main as audit_main
from .config import contract
from .data import split_groups
from .aggregate import aggregate


def main(argv=None):
    return audit_main(argv, contract_factory=contract, splitter=split_groups, aggregate_fn=aggregate)


if __name__ == "__main__":
    main()
