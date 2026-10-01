from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments.fhp.exp4_fhp_average_policy_audit.run import synthetic_source
from experiments.fhp.exp13_fhp_policy_capacity import run as workflow
from experiments.fhp.exp13_fhp_policy_capacity.data import load_source, group_replay, group_digest
from experiments.fhp.exp13_fhp_policy_capacity.fitting import new_model, fit_path, state_hash, write_json
from experiments.fhp.exp13_fhp_policy_capacity.selection import select
from experiments.fhp.exp13_fhp_policy_capacity.evaluation import build_tasks, expected_metrics
from experiments.fhp.exp14_fhp_card_architecture.config import contract
from experiments.fhp.exp14_fhp_card_architecture.data import card_keys, split_groups
from experiments.fhp.exp14_fhp_card_architecture import aggregate as aggregation
from fhp_escher.card_policy import ResidualCardPolicy, suit_tokens, invariant_context
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload, sha256_file
from fhp_escher.game import load_fhp_game


@pytest.fixture
def data(tmp_path):
    torch.set_num_threads(1)
    source = synthetic_source(tmp_path / "source")
    buffer, iteration, template, _ = load_source(source, 0)
    return source, group_replay(buffer, iteration), template


def test_prespecified_counts_shared_base_and_rng_preservation():
    config = contract()
    assert config["experiment_id"] == 14 and config["seeds"] == [0, 1, 2]
    assert config["control_updates"] == [20000, 60000] and config["replicates"] == [0, 1]
    assert len(config["recipes"]) == 4 and config["batch_size"] == 2048
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state()
    base = new_model(config["architectures"]["standard"], 19)
    counts = []
    for name, settings in config["architectures"].items():
        model = new_model(settings, 19)
        assert sum(p.numel() for p in model.parameters()) == settings["parameters"]
        if name != "standard":
            counts.append(settings["parameters"])
            assert state_hash(model.base) == state_hash(base)
            assert all(p.requires_grad for p in model.base.parameters())
            assert torch.count_nonzero(model.residual_logits(torch.randn(7, 183))) == 0
    assert max(counts)/min(counts) - 1 < .003
    # Do not advance global RNGs while creating candidates (torch.randn above
    # is deliberately outside the factory and checked separately below).
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], numpy_before[1])
    assert after[0] == numpy_before[0] and after[2:] == numpy_before[2:]
    torch.set_rng_state(torch_before)
    new_model(config["architectures"]["attention"], 72)
    assert torch.equal(torch.get_rng_state(), torch_before)


def permute_suits(inputs, order):
    result = inputs.clone()
    result[..., :52] = inputs[..., :52].reshape(*inputs.shape[:-1], 4, 13)[..., order, :].reshape(*inputs.shape[:-1], 52)
    result[..., 52:104] = inputs[..., 52:104].reshape(*inputs.shape[:-1], 4, 13)[..., order, :].reshape(*inputs.shape[:-1], 52)
    for start in (164, 168):
        result[..., start:start+4] = inputs[..., start:start+4][..., order]
    return result


@pytest.mark.parametrize("architecture", ["deepsets", "attention"])
def test_suit_branch_invariance_equivariance_and_gradients(data, architecture):
    _, groups, _ = data
    inputs = torch.from_numpy(groups.features[:12]).double()
    model = new_model(contract()["architectures"][architecture], 31).double()
    with torch.no_grad():
        head = model.residual[-1]
        head.weight.copy_(torch.linspace(-.3, .5, head.weight.numel()).reshape_as(head.weight))
    reference = model.residual_logits(inputs)
    assert reference.abs().max() > 0  # Not a vacuous zero-head invariance check.
    for order in permutations(range(4)):
        changed = permute_suits(inputs, order)
        torch.testing.assert_close(invariant_context(changed), invariant_context(inputs), rtol=0, atol=0)
        torch.testing.assert_close(suit_tokens(changed), suit_tokens(inputs)[..., list(order), :], rtol=0, atol=0)
        torch.testing.assert_close(model.residual_logits(changed), reference, rtol=1e-10, atol=1e-10)
    params = [p for name, p in model.named_parameters() if not name.startswith("base.")]
    gradients = torch.autograd.grad(reference.square().sum(), params)
    changed = model.residual_logits(permute_suits(inputs, (2, 0, 3, 1)))
    changed_gradients = torch.autograd.grad(changed.square().sum(), params)
    for a, b in zip(gradients, changed_gradients):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    if architecture == "attention":
        tokens = model.suit_encoder(suit_tokens(inputs))
        order = [3, 1, 0, 2]
        torch.testing.assert_close(model.attention(tokens[..., order, :]), model.attention(tokens)[..., order, :],
                                   rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("architecture", ["dense", "deepsets", "attention"])
def test_vectorised_forward_matches_individual_examples(data, architecture):
    _, groups, _ = data
    model = new_model(contract()["architectures"][architecture], 17).double()
    with torch.no_grad():
        model.residual[-1].weight.fill_(.1)
    inputs = torch.from_numpy(groups.features[:8]).double()
    torch.testing.assert_close(model(inputs), torch.stack([model(row) for row in inputs]), rtol=1e-10, atol=1e-10)
    assert suit_tokens(inputs).shape == (8, 4, 26)
    # Rank order and hole/board roles are not pooled away inside each suit.
    torch.testing.assert_close(suit_tokens(inputs)[..., :13], inputs[..., :52].reshape(8, 4, 13))
    torch.testing.assert_close(suit_tokens(inputs)[..., 13:], inputs[..., 52:104].reshape(8, 4, 13))


def test_card_split_keeps_betting_and_player_variants_together(data):
    _, groups, _ = data
    config = contract(True)
    splits, metadata = split_groups(groups, config, 0)
    _, again = split_groups(groups, config, 0)
    assert metadata == again
    keys = {name: set(map(bytes, card_keys(part.features))) for name, part in splits.items()}
    assert not keys["training"] & keys["validation"]
    assert not keys["training"] & keys["test"]
    assert not keys["validation"] & keys["test"]
    assert sum(len(k) for k in keys.values()) == len(set(map(bytes, card_keys(groups.features))))
    assert sum(part.rows for part in splits.values()) == groups.rows
    assert sum(part.masses.sum() for part in splits.values()) == pytest.approx(groups.masses.sum())
    # Change only player/context of every group: the same card partition must
    # still result. Targets are NEVER averaged across these context variants.
    changed = deepcopy(groups)
    changed.features[:, 104:106] = changed.features[:, 104:106][:, ::-1]
    changed.features[:, 108:138] = np.roll(changed.features[:, 108:138], 1, axis=1)
    other, _ = split_groups(changed, config, 0)
    assert keys == {name: set(map(bytes, card_keys(part.features))) for name, part in other.items()}
    invalid = groups.features.copy()
    invalid[0, 0] = .5
    with pytest.raises(ValueError, match="binary"):
        card_keys(invalid)


def fit_candidates(data, root):
    _, groups, template = data
    config = contract(True)
    rows = {}
    for architecture in config["architectures"]:
        rows[architecture] = fit_path(groups, None, config=config, seed=0, replicate=0,
            architecture=architecture, recipe="adam_003", updates=config["control_updates"],
            directory=root / architecture, template=template, data_sha256=group_digest(groups), phase="deployment")
    return rows


def test_fits_train_residuals_match_sampling_and_reload(data, tmp_path):
    source, groups, _ = data
    before = sha256_file(source / "state.pt"), group_digest(groups)
    config = contract(True)
    rows = fit_candidates(data, tmp_path / "fits")
    assert len({r[-1]["sample_sequence_sha256"] for r in rows.values()}) == 1
    for architecture, settings in config["architectures"].items():
        path = tmp_path / "fits" / architecture / "4.pkl"
        loaded = LoadedFHPPolicy(load_fhp_game(), path)
        assert loaded.checkpoint["experiment_id"] == 14
        assert loaded.checkpoint["experiment_name"] == config["experiment_name"]
        assert loaded.model.input_size == 183 and loaded.model.output_size == 3
        if architecture != "standard":
            assert isinstance(loaded.model, ResidualCardPolicy)
            assert loaded.model.kind == architecture
            assert torch.count_nonzero(loaded.model.residual[-1].weight) > 0
            if architecture != "dense":
                initial = new_model(settings, config["initialisation_seed_base"])
                assert not torch.equal(initial.suit_encoder[0].weight, loaded.model.suit_encoder[0].weight)
        payload = load_checkpoint_payload(path)
        if architecture != "standard":
            payload["policy_model"]["version"] = 999
            bad = tmp_path / "bad.pkl"
            with bad.open("wb") as handle:
                pickle.dump(payload, handle)
            with pytest.raises(ValueError, match="metadata"):
                LoadedFHPPolicy(load_fhp_game(), bad)
    assert before == (sha256_file(source / "state.pt"), group_digest(groups))
    assert not list((tmp_path / "fits").rglob("*.pt"))


def decision(player, opponent):
    holes = ["Kc Kd", opponent] if player == 0 else [opponent, "Kc Kd"]
    state = load_fhp_game().new_initial_state()
    def card(token):
        return "23456789TJQKA".index(token[0])*4 + "cdhs".index(token[1])
    for token in " ".join(holes).split():
        state.apply_action(card(token))
    state.apply_action(1); state.apply_action(1)
    for token in "9h 5s 2c".split():
        state.apply_action(card(token))
    if player == 0:
        state.apply_action(1)
    return state


def test_no_opponent_private_information_in_fitted_candidates(data, tmp_path):
    fit_candidates(data, tmp_path)
    class OwnInformation:
        def __init__(self, state, player):
            self.state, self.player = state, player
        def information_state_tensor(self, player):
            assert player == self.player, "Opponent actual cards requested"
            return self.state.information_state_tensor(player)
        def __getattr__(self, name):
            return getattr(self.state, name)
    for architecture in contract()["architectures"]:
        policy = LoadedFHPPolicy(load_fhp_game(), tmp_path / architecture / "4.pkl")
        for player in (0, 1):
            probabilities = []
            for opponent in ("Qc Qd", "Ac Ad"):
                state = OwnInformation(decision(player, opponent), player)
                policy.feature_encoder._information_cache.clear()
                probs = policy.action_probabilities(state)
                assert set(probs) == set(state.legal_actions())
                assert sum(probs.values()) == pytest.approx(1)
                probabilities.append(probs)
            assert probabilities[0] == probabilities[1]


def test_comparators_matched_by_budget_replicate_and_deals(data):
    from experiments.fhp.exp13_fhp_policy_capacity.config import arm_specs
    source, _, _ = data
    config = contract(True)
    selected = {"selected": {a: {"recipe": "adam_003", "updates": 2} for a in config["architectures"]}}
    policies = {name: source / "policy.pkl" for name in ["archived", *arm_specs(config, selected)]}
    tasks = build_tasks(policies, config, 0)
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    assert len(direct) == 54  # 24 vs archived, plus five contrasts x six endpoints.
    assert len({t["evaluation_seed"] for t in direct}) == 1
    expected = expected_metrics(config)
    actual = {(t["policy_a_name"], "lbr_mbb_per_hand" if t["kind"] == "lbr" else f"{t['kind']}_{t['opponent']}") for t in tasks}
    assert actual == expected
    for task in direct:
        if task["opponent"] != "archived":
            assert task["policy_a_name"].split("_", 1)[1] == task["policy_b_name"].split("_", 1)[1]


def test_evaluation_aliases_equal_full_evaluation_and_resume(data, tmp_path, monkeypatch):
    from experiments.fhp.exp13_fhp_policy_capacity import evaluation
    from experiments.fhp.exp13_fhp_policy_capacity.config import arm_specs
    source, _, _ = data
    config = contract(True)
    selected = {"selected": {a: {"recipe": "adam_003", "updates": 2} for a in config["architectures"]}}
    policies = {name: source / "policy.pkl" for name in ["archived", *arm_specs(config, selected)]}
    calls = []
    def fake_worker(task):
        calls.append(task)
        result = {**task, "num_deal_pairs": task["num_deals"], "num_games": 2*task["num_deals"],
                  "mean_mbb_per_hand": 5.0}
        if task["kind"] == "lbr":
            result.update({f"_{key}_values": [1., 2.] for key in ("paired", "player_zero", "player_one")})
        return evaluation.relabel_result(result, task)
    monkeypatch.setattr(evaluation, "_evaluation_worker", fake_worker)
    compact = evaluation.evaluate(policies, config, 0, tmp_path / "compact/evaluation_tasks", workers=1)
    assert len(calls) == 7  # Five rules, one LBR shard, one direct matchup.
    evaluation.evaluate(policies, config, 0, tmp_path / "compact/evaluation_tasks", workers=1)
    assert len(calls) == 7
    full = evaluation.evaluate(policies, {**config, "deduplicate_identical_evaluation": False}, 0,
                               tmp_path / "full/evaluation_tasks", workers=1)
    key = lambda row: (row["arm"], row["metric"])
    assert sorted(compact, key=key) == sorted(full, key=key)


@pytest.mark.parametrize("kind", ["controller", "smoke", "screen", "select", "train", "aggregate"])
def test_cloud_routes_only_exp14_and_retains_no_training_states(tmp_path, kind):
    repo = Path(__file__).parents[1]
    output = tmp_path / "job.json"
    subprocess.run([sys.executable, str(repo / "gcp/exp14_card_architecture_batch.py"), "--kind", kind,
        "--output", str(output), "--run-id", "exp14-test", "--bucket-root", "gs://bucket",
        "--service-account", "runner@test.iam.gserviceaccount.com", "--repo-ref", "abc",
        "--source-run-id", "exp2-source", "--project-id", "test", "--region", "europe-west1"], check=True)
    job = json.loads(output.read_text())
    group = job["taskGroups"][0]
    text = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "exp13" not in text and "EXP13" not in text
    assert "exp14" in text and group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskCount"] == (3 if kind in ("screen", "train") else 1)
    if kind in ("screen", "train"):
        assert group["taskCountPerNode"] == 1 and group["parallelism"] == 3
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
        assert "EXP14_REMOTE_WORKER" in text and r"\.pt$" in text
    script = tmp_path / "job.sh"
    script.write_text(text)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_three_source_card_audit_pipeline(data, tmp_path, monkeypatch):
    config = contract(True)
    config["seeds"] = [0, 1, 2]
    def fake_evaluate(policies, config, seed, directory, workers, sync):
        rows = [{"arm": arm, "metric": metric, "mean_mbb_per_hand": 1 + seed}
                for arm, metric in expected_metrics(config)]
        write_json(directory.parent / "evaluation_summary.json", rows)
    monkeypatch.setattr(workflow, "evaluate", fake_evaluate)
    monkeypatch.setattr(aggregation, "plot", lambda *args: None)
    root = tmp_path / "outputs"
    sources = []
    for seed in config["seeds"]:
        source = synthetic_source(tmp_path / f"source{seed}")
        payload = load_checkpoint_payload(source / "policy.pkl")
        payload["seed"] = seed
        with (source / "policy.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        state = torch.load(source / "state.pt", weights_only=False)
        state["seed"] = seed
        torch.save(state, source / "state.pt")
        manifest = json.loads((source / "run_manifest.json").read_text())
        manifest["seed"] = seed
        write_json(source / "run_manifest.json", manifest)
        rows = json.loads((source / "checkpoint_manifest.json").read_text())
        rows[0].update(sha256=sha256_file(source / "policy.pkl"), training_state_sha256=sha256_file(source / "state.pt"))
        write_json(source / "checkpoint_manifest.json", rows)
        workflow.screen_worker(source, root, seed, smoke=True, config=config, splitter=split_groups)
        assert not list(root.rglob("diagnostic_test.json"))
        if seed < 2:
            with pytest.raises(FileNotFoundError):
                select(root, config)
        sources.append(source)
    select(root, config)
    for seed, source in enumerate(sources):
        workflow.deploy_worker(source, root, seed, smoke=True, config=config, splitter=split_groups, evaluation_workers=1)
    analysis = aggregation.aggregate(root, smoke=True, config=config)
    results = json.loads((analysis / "summary.json").read_text())
    assert all(r["n_source_seeds"] == 3 for r in results["summaries"])
    assert "deepsets_minus_dense_2" in {r["contrast"] for r in results["paired_contrasts"]}
    assert not list(root.rglob("*.pt"))
    wrong = root / "workers/seed_1/diagnostic_test.json"
    results = json.loads(wrong.read_text())
    results["test_data_sha256"] = "wrong cards"
    write_json(wrong, results)
    with pytest.raises(ValueError, match="test results"):
        aggregation.aggregate(root, smoke=True, config=config)
