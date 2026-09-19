from experiments.fhp.compare_evaluated_experiments import _matrix


def test_crossplay_matrix_is_antisymmetric():
    pairs = []
    for left in (1, 2, 3, 4):
        for right in (1, 2, 3, 4):
            if left < right:
                pairs.append(
                    {
                        "checkpoint": "checkpoint_12h",
                        "policy_a": f"experiment_{left}",
                        "policy_b": f"experiment_{right}",
                        "mean_mbb_per_hand": float(10 * left + right),
                    }
                )
    rows = _matrix(pairs, "checkpoint_12h")
    for left in range(4):
        assert rows[left][f"vs_experiment_{left + 1}_mbb_per_hand"] == 0.0
        for right in range(4):
            assert rows[left][f"vs_experiment_{right + 1}_mbb_per_hand"] == -rows[
                right
            ][f"vs_experiment_{left + 1}_mbb_per_hand"]
