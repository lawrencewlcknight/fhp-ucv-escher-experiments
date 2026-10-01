import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import EXPERIMENT_CONFIG as EXP2
from experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.config import EXPERIMENT_CONFIG as EXP3
from experiments.fhp.exp4_fhp_average_policy_audit.fitting import fit_path as old_fit
from experiments.fhp.exp4_fhp_average_policy_audit.run import synthetic_source
from experiments.fhp.exp13_fhp_policy_capacity.config import contract, arm_specs
from experiments.fhp.exp13_fhp_policy_capacity.data import group_replay, group_digest, load_source, split_groups
from experiments.fhp.exp13_fhp_policy_capacity.fitting import fit_path, new_model, optimizer_for, state_hash, write_json, fitting_benchmark
from experiments.fhp.exp13_fhp_policy_capacity.selection import select, read_selection, fit_directory
from experiments.fhp.exp13_fhp_policy_capacity.evaluation import build_tasks, task_fingerprint, validate_result
from experiments.fhp.exp13_fhp_policy_capacity.aggregate import collapse_replicates, paired_contrasts
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload, sha256_file
from fhp_escher.game import load_fhp_game


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    return synthetic_source(tmp_path / "source")


@pytest.fixture
def grouped(source):
    buffer, iteration, template, _ = load_source(source, 0)
    return group_replay(buffer, iteration), template


def test_frozen_contract_uses_only_source_policy_architectures():
    config = contract()
    assert config["seeds"] == [0, 1, 2]
    assert config["replicates"] == [0, 1]
    assert config["control_updates"] == [20000, 60000]
    assert config["screen_updates"] == [5000, 10000, 20000, 40000, 60000]
    assert len(config["recipes"]) == 4
    assert config["training_state_retention"] == "none"
    for name, source in (("standard", EXP2), ("wide", EXP3)):
        architecture = config["architectures"][name]
        assert architecture["network_layers"] == list(source["average_policy_network_layers"])
        assert architecture["branch_width"] == source["structured_branch_width"]
        model = new_model(architecture, 1)
        assert model.input_size == 183
        assert sum(p.numel() for p in model.parameters()) == architecture["parameters"]


def test_split_canonical_groups_no_leakage_and_stable_identity(grouped):
    groups, _ = grouped
    config = contract(True)
    splits, metadata = split_groups(groups, config, 0)
    again, meta_again = split_groups(groups, config, 0)
    assert metadata == meta_again
    ids = {k: {row.tobytes() for row in v.features} for k, v in splits.items()}
    assert not ids["training"] & ids["validation"]
    assert not ids["training"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == len(groups.features)
    assert sum(v.rows for v in splits.values()) == groups.rows
    assert sum(v.masses.sum() for v in splits.values()) == pytest.approx(groups.masses.sum())
    for k in splits:
        assert group_digest(splits[k]) == group_digest(again[k])


def fitting_kwargs(groups, template, directory, architecture="standard", replicate=0):
    config = contract(True)
    return dict(config=config, seed=0, replicate=replicate, architecture=architecture,
                recipe="adam_003", updates=[2, 4], directory=directory, template=template,
                data_sha256=group_digest(groups), phase="deployment")


@pytest.mark.parametrize("architecture", ["standard", "wide"])
def test_fit_matches_existing_objective_nested_endpoints_and_reload(grouped, tmp_path, architecture):
    groups, template = grouped
    kwargs = fitting_kwargs(groups, template, tmp_path / "new", architecture)
    config = kwargs["config"]
    old_config = {**config["architectures"][architecture], "initialisation_seed_base": config["initialisation_seed_base"],
                  "batch_size": config["batch_size"], "learning_rate": .003, "updates": [2, 4]}
    config["sampler_seed_base"] = config["initialisation_seed_base"] + 1000
    rows = fit_path(groups, None, **kwargs)
    old_fit(groups, None, config=old_config, seed=0, sampler="uniform", directory=tmp_path / "old", template=template)
    for update in (2, 4):
        payload = load_checkpoint_payload(tmp_path / "new" / f"{update}.pkl")
        old = load_checkpoint_payload(tmp_path / "old" / f"{update}.pkl")
        for key in payload["policy_state_dict"]:
            assert torch.equal(payload["policy_state_dict"][key], old["policy_state_dict"][key])
        policy = LoadedFHPPolicy(load_fhp_game(), tmp_path / "new" / f"{update}.pkl")
        assert policy.model.hidden_layers == config["architectures"][architecture]["network_layers"]
        assert payload["feature_encoder"] == template["feature_encoder"]
        assert payload["policy_model"]["branch_width"] == config["architectures"][architecture]["branch_width"]
        assert payload["experiment_id"] == 13
    short = fitting_kwargs(groups, template, tmp_path / "short", architecture)
    short["config"] = config
    short["updates"] = [2]
    fit_path(groups, None, **short)
    assert state_hash(LoadedFHPPolicy(load_fhp_game(), tmp_path / "new/2.pkl").model) == state_hash(
        LoadedFHPPolicy(load_fhp_game(), tmp_path / "short/2.pkl").model)
    assert not list((tmp_path / "new").glob("*.pt"))
    assert all("test" not in row for row in rows)


def test_sampling_independent_of_architecture_recipe_and_initialisation(grouped, tmp_path):
    groups, template = grouped
    rows = []
    for architecture in ("standard", "wide"):
        for recipe in ("adam_003", "adamw_001"):
            kwargs = fitting_kwargs(groups, template, tmp_path / architecture / recipe, architecture)
            kwargs["recipe"] = recipe
            rows.append(fit_path(groups, None, **kwargs)[-1])
    assert len({r["sample_sequence_sha256"] for r in rows}) == 1
    assert rows[0]["initial_state_sha256"] == rows[1]["initial_state_sha256"]
    assert rows[2]["initial_state_sha256"] == rows[3]["initial_state_sha256"]
    assert rows[0]["initial_state_sha256"] != rows[2]["initial_state_sha256"]
    model = new_model(contract()["architectures"]["wide"], 1)
    optimizer = optimizer_for(model, contract()["recipes"]["adamw_001"])
    for group in optimizer.param_groups:
        assert all((p.ndim >= 2) == (group["weight_decay"] > 0) for p in group["params"])


def test_interrupted_fit_repeats_from_reset_and_rejects_wrong_inputs(grouped, tmp_path):
    groups, template = grouped
    kwargs = fitting_kwargs(groups, template, tmp_path / "resume")
    def interrupt():
        raise RuntimeError("interrupt")
    with pytest.raises(RuntimeError, match="interrupt"):
        fit_path(groups, None, **kwargs, on_checkpoint=interrupt)
    fit_path(groups, None, **kwargs)
    clean = fitting_kwargs(groups, template, tmp_path / "clean")
    fit_path(groups, None, **clean)
    assert state_hash(LoadedFHPPolicy(load_fhp_game(), tmp_path / "resume/4.pkl").model) == state_hash(
        LoadedFHPPolicy(load_fhp_game(), tmp_path / "clean/4.pkl").model)
    fit_path(groups, None, **kwargs, on_checkpoint=interrupt)  # complete fits are reused
    with pytest.raises(ValueError, match="checksum"):
        fit_path(groups, None, **{**kwargs, "data_sha256": "changed"})


def synthetic_screens(root, config):
    for seed in config["seeds"]:
        worker = root / "workers" / f"seed_{seed}"
        write_json(worker / "manifest.json", {"config": config, "seed": seed,
                   **{k: "test" for k in ("audit_commit", "audit_source_sha256", "python", "torch", "numpy")}})
        split = {name: {"data_sha256": f"{seed}-{name}"} for name in ("training", "validation", "test")}
        write_json(worker / "replay_diagnostics.json", {"splits": split})
        write_json(worker / "SCREEN_SUCCESS.json", {"seed": seed, "smoke": config["smoke"], "status": "screen_complete"})
        for architecture in config["architectures"]:
            for recipe in config["recipes"]:
                for replicate in config["replicates"]:
                    rows = [{"seed": seed, "replicate": replicate, "architecture": architecture, "recipe": recipe,
                             "phase": "screen", "updates": updates,
                             "data_sha256": split["training"]["data_sha256"],
                             "validation_data_sha256": split["validation"]["data_sha256"],
                             "sample_sequence_sha256": f"{seed}-{replicate}-{updates}",
                             "initial_state_sha256": f"{seed}-{replicate}-{architecture}",
                             "validation": {"all": {"weighted_ce": 1 + seed*.1 + replicate*.01
                                + .05*(recipe != "adam_001") + .02*(updates != config["screen_updates"][1])}}}
                            for updates in config["screen_updates"]]
                    path = fit_directory(worker, "screen", architecture, recipe, replicate)
                    write_json(path / "metrics.json", rows)
                    write_json(path / "fit_contract.json", {"config": config, "updates": config["screen_updates"],
                               **{k: rows[0][k] for k in ("seed", "replicate", "architecture", "recipe", "phase",
                                   "data_sha256", "validation_data_sha256", "initial_state_sha256")}})


def test_global_selection_validation_only_complete_sources_and_lock(tmp_path):
    config = contract()
    synthetic_screens(tmp_path, config)
    result = select(tmp_path, config)
    assert all(r["recipe"] == "adam_001" and r["updates"] == 10000 for r in result["selected"].values())
    assert not result["selection_uses_test_or_gameplay"]
    assert len(result["source_manifest_hashes"]) == 3
    write_json(tmp_path / "workers/seed_0/diagnostic_test.json", {"deliberately": "not read by selection"})
    assert select(tmp_path, config) == result
    read_selection(tmp_path / "selection.json", config)
    with pytest.raises(ValueError, match="source/code"):
        read_selection(tmp_path / "selection.json", config, {"changed": True}, 0)
    path = fit_directory(tmp_path / "workers/seed_0", "screen", "standard", "adam_003", 0) / "metrics.json"
    rows = json.loads(path.read_text())
    rows[0]["test"] = {}
    write_json(path, rows)
    with pytest.raises(ValueError, match="test access"):
        select(tmp_path, config)
    rows[0].pop("test")
    rows[0]["validation"]["all"]["weighted_ce"] = -100
    write_json(path, rows)
    with pytest.raises(ValueError, match="locked"):
        select(tmp_path, config)


def test_evaluation_has_matched_capacity_and_common_deals(source):
    config = contract(True)
    selection = {"selected": {a: {"recipe": "adam_003", "updates": 2} for a in config["architectures"]}}
    specs = arm_specs(config, selection)
    policies = {name: source / "policy.pkl" for name in ["archived", *specs]}
    tasks = build_tasks(policies, config, 0)
    assert len([t for t in tasks if t["kind"] == "lbr"]) == 13
    matched = [t for t in tasks if t.get("opponent") == "matched_standard"]
    assert len(matched) == 6
    for task in matched:
        assert task["policy_b_name"] == task["policy_a_name"].replace("wide_", "standard_", 1)
    assert len({t["evaluation_seed"] for t in tasks if t["kind"] == "direct_crossplay"}) == 1
    task = tasks[0]
    assert task_fingerprint(task) == task_fingerprint({**task, "policy_a_path": "/relocated"})
    assert task_fingerprint(task) != task_fingerprint({**task, "policy_a_sha256": "changed"})


def test_source_size_enforced_before_fitting(source, tmp_path, monkeypatch):
    from experiments.fhp.exp13_fhp_policy_capacity import run
    buffer, iteration, template, provenance = load_source(source, 0)
    with pytest.raises(ValueError, match="Synthetic"):
        run.prepare(source, tmp_path / "real", 0, contract())
    template.pop("audit_synthetic_fixture")
    monkeypatch.setattr(run, "load_source", lambda *args: (buffer, iteration, template, provenance))
    # A cloud smoke must also reject undersized real inputs.
    for smoke in (False, True):
        with pytest.raises(ValueError, match="1,000,000"):
            run.prepare(source, tmp_path / str(smoke), 0, contract(smoke))


def test_validation_identity_cannot_change_on_retry(grouped, tmp_path):
    from copy import deepcopy
    groups, template = grouped
    kwargs = fitting_kwargs(groups, template, tmp_path / "fit")
    fit_path(groups, groups, **kwargs)
    changed = deepcopy(groups)
    changed.targets = changed.targets[:, ::-1].copy()
    with pytest.raises(ValueError, match="Incompatible fit"):
        fit_path(groups, changed, **kwargs)


def test_selection_rejects_mixed_data_and_code(tmp_path):
    config = contract()
    synthetic_screens(tmp_path, config)
    worker = tmp_path / "workers/seed_1"
    path = fit_directory(worker, "screen", "wide", "adam_003", 0) / "metrics.json"
    rows = json.loads(path.read_text())
    original = rows[0]["data_sha256"]
    rows[0]["data_sha256"] = "wrong replay"
    write_json(path, rows)
    with pytest.raises(ValueError, match="Screen metric data"):
        select(tmp_path, config)
    rows[0]["data_sha256"] = original
    write_json(path, rows)
    manifest = json.loads((worker / "manifest.json").read_text())
    manifest["audit_commit"] = "wrong commit"
    write_json(worker / "manifest.json", manifest)
    with pytest.raises(ValueError, match="different code"):
        select(tmp_path, config)


def test_boundary_warning_includes_unselected_improving_recipe(tmp_path):
    config = contract()
    synthetic_screens(tmp_path, config)
    for seed in config["seeds"]:
        for replicate in config["replicates"]:
            path = fit_directory(tmp_path / "workers" / f"seed_{seed}", "screen", "wide", "adam_0003", replicate) / "metrics.json"
            rows = json.loads(path.read_text())
            rows[-1]["validation"]["all"]["weighted_ce"] -= .01
            write_json(path, rows)
    selected = select(tmp_path, config)["selected"]
    assert selected["wide"]["updates"] == 10000
    assert selected["wide"]["budget_boundary_warning"]
    assert not selected["standard"]["budget_boundary_warning"]


def test_cached_evaluation_result_must_match_actual_task(source):
    config = contract(True)
    selection = {"selected": {a: {"recipe": "adam_003", "updates": 2} for a in config["architectures"]}}
    policies = {name: source / "policy.pkl" for name in ["archived", *arm_specs(config, selection)]}
    task = build_tasks(policies, config, 0)[0]
    result = {**task, "num_deal_pairs": task["num_deals"], "num_games": 2*task["num_deals"], "mean_mbb_per_hand": 0}
    validate_result(task, result)
    for bad in ({"policy_a_sha256": "stale"}, {"num_deal_pairs": 1}, {"mean_mbb_per_hand": float("nan")}):
        with pytest.raises(ValueError, match="task identity"):
            validate_result(task, {**result, **bad})


def test_production_batch_benchmark_does_not_change_replay_or_source(grouped, source):
    groups, _ = grouped
    before = group_digest(groups), sha256_file(source / "state.pt"), sha256_file(source / "policy.pkl")
    result = fitting_benchmark(groups, contract(True), warmup_steps=1, timed_steps=2)
    assert set(result["architectures"]) == {"standard", "wide"}
    assert all(r["batch_size"] == 8 and r["seconds_per_update"] > 0 for r in result["architectures"].values())
    assert before == (group_digest(groups), sha256_file(source / "state.pt"), sha256_file(source / "policy.pkl"))


@pytest.mark.parametrize("architecture", ["standard", "wide"])
def test_fitted_policy_predictions_reload_and_ignore_opponent_cards(grouped, source, tmp_path, architecture):
    from fhp_escher.features import FHPFeatureEncoder
    groups, template = grouped
    before = sha256_file(source / "state.pt"), sha256_file(source / "policy.pkl")
    fit_path(groups, None, **fitting_kwargs(groups, template, tmp_path / "fit", architecture))
    loaded = LoadedFHPPolicy(load_fhp_game(), tmp_path / "fit/4.pkl")
    own_encoder = FHPFeatureEncoder()
    class OwnInformationOnly:
        def __init__(self, state, player):
            self.state, self.player = state, player
        def information_state_tensor(self, player):
            assert player == self.player, "Privileged opponent cards requested"
            return self.state.information_state_tensor(player)
        def __getattr__(self, name):
            return getattr(self.state, name)
    for player in (0, 1):
        probabilities = []
        for opponent in ("Qc Qd", "Ac Ad"):
            holes = ["Kc Kd", opponent] if player == 0 else [opponent, "Kc Kd"]
            state = load_fhp_game().new_initial_state()
            for card in " ".join(holes).split():
                state.apply_action("23456789TJQKA".index(card[0])*4 + "cdhs".index(card[1]))
            state.apply_action(1); state.apply_action(1)
            for card in "9h 5s 2c".split():
                state.apply_action("23456789TJQKA".index(card[0])*4 + "cdhs".index(card[1]))
            if player == 0:
                state.apply_action(1)
            assert state.current_player() == player
            view = OwnInformationOnly(state, player)
            loaded.feature_encoder._information_cache.clear()
            actual = loaded.action_probabilities(view)
            encoded = torch.from_numpy(own_encoder.information_state(view, player))
            mask = torch.as_tensor(state.legal_actions_mask(), dtype=torch.bool)
            with torch.no_grad():
                expected = torch.softmax(loaded.model(encoded).masked_fill(~mask, -1e20), -1)
            assert actual == {a: expected[a].item() for a in state.legal_actions()}
            probabilities.append(actual)
        assert probabilities[0] == probabilities[1]
    assert before == (sha256_file(source / "state.pt"), sha256_file(source / "policy.pkl"))


def test_replicates_not_pseudoreplicated_and_factorial_contrast():
    config = contract()
    rows = [{"seed": seed, "replicate": rep, "arm": f"{arch}_{u}", "metric": "quality",
             "value": seed + rep*.2 - (arch == "wide")*.4 - (u == 60000)*.1}
            for seed in config["seeds"] for rep in config["replicates"]
            for arch in config["architectures"] for u in config["control_updates"]]
    reduced = collapse_replicates(rows, config)
    assert len(reduced) == 12 and all(r["n_fitting_replicates"] == 2 for r in reduced)
    values = {}
    for row in reduced:
        values.setdefault((row["metric"], row["arm"]), {})[row["seed"]] = row["value"]
    results = {r["contrast"]: r for r in paired_contrasts(values, config)}
    assert results["wide_minus_standard_20000"]["mean"] == pytest.approx(-.4)
    assert results["wide_minus_standard_20000"]["n_source_seeds"] == 3
    assert results["wide_minus_standard_20000"]["exact_two_sided_sign_flip_p"] == .25
    assert results["capacity_by_budget_interaction"]["mean"] == pytest.approx(0)
    with pytest.raises(ValueError, match="Missing"):
        collapse_replicates(rows[:-1], config)


@pytest.mark.parametrize("kind", ["controller", "smoke", "screen", "select", "train", "aggregate"])
def test_cloud_stage_routing_resources_and_retention(tmp_path, kind):
    root = Path(__file__).parents[1]
    path = tmp_path / f"{kind}.json"
    subprocess.run([sys.executable, str(root / "gcp/exp13_policy_capacity_batch.py"), "--kind", kind,
                    "--output", str(path), "--run-id", "exp13-test", "--bucket-root", "gs://bucket",
                    "--service-account", "runner@test.iam.gserviceaccount.com", "--repo-ref", "abc",
                    "--source-run-id", "exp2-fhp-source", "--project-id", "test", "--region", "europe-west1"], check=True)
    job = json.loads(path.read_text())
    group = job["taskGroups"][0]
    text = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == (3 if kind in ("screen", "train") else 1)
    assert group["taskCountPerNode"] == 1 and group["taskSpec"]["maxRetryCount"] == 0
    assert "exp1_fhp_grouped_wide_ucv_baseline.run" not in text
    if kind in ("screen", "train"):
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
        assert "EXP13_REMOTE_WORKER" in text and "fetch-source" in text
        assert group["taskSpec"]["computeResource"]["memoryMib"] == 30000
        assert "training_states" in text and r"\.pt$" in text
    if kind == "train":
        assert 'deploy --seed "$SEED"' in text and "selection.json" in text
    if kind == "select":
        assert " select --output-root" in text and "fetch-source" not in text
    shell = tmp_path / f"{kind}.sh"
    shell.write_text(text)
    subprocess.run(["bash", "-n", str(shell)], check=True)


def test_launcher_order_and_syntax():
    path = Path(__file__).parents[1] / "gcp/run_exp13_policy_capacity.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    commands = ['ensure "$SMOKE_JOB" smoke', 'ensure "$SCREEN_JOB" screen', 'ensure "$SELECT_JOB" select',
                'ensure "$TRAIN_JOB" train', 'ensure "$AGGREGATE_JOB" aggregate']
    assert [text.index(c) for c in commands] == sorted(text.index(c) for c in commands)
    assert "Pull the pushed code and reset REPO_REF" in text


def test_three_source_pipeline_and_aggregate_completeness(tmp_path, monkeypatch):
    """Real grouping/fits/reloads on three synthetic sources; stub expensive play."""
    import pickle
    from fhp_escher.checkpointing import sha256_file
    from experiments.fhp.exp13_fhp_policy_capacity import run, aggregate as aggregation
    config = contract(True)
    config["seeds"] = [0, 1, 2]
    monkeypatch.setattr(run, "contract", lambda smoke=False: config)
    monkeypatch.setattr(aggregation, "contract", lambda smoke=False: config)
    monkeypatch.setattr(aggregation, "plot", lambda *args: None)
    def fake_evaluate(policies, config, seed, directory, workers, sync):
        tasks = build_tasks(policies, config, seed)
        rows = []
        for task in tasks:
            metric = "lbr_mbb_per_hand" if task["kind"] == "lbr" else f"{task['kind']}_{task['opponent']}"
            rows.append({"arm": task["policy_a_name"], "metric": metric, "mean_mbb_per_hand": 1 + seed})
        write_json(directory.parent / "evaluation_summary.json", rows)
    monkeypatch.setattr(run, "evaluate", fake_evaluate)
    root = tmp_path / "output"
    sources = []
    for seed in config["seeds"]:
        source = synthetic_source(tmp_path / f"input_{seed}")
        with (source / "policy.pkl").open("rb") as handle:
            policy = pickle.load(handle)
        policy["seed"] = seed
        with (source / "policy.pkl").open("wb") as handle:
            pickle.dump(policy, handle)
        state = torch.load(source / "state.pt", weights_only=False)
        state["seed"] = seed
        torch.save(state, source / "state.pt")
        manifest = json.loads((source / "run_manifest.json").read_text())
        manifest["seed"] = seed
        write_json(source / "run_manifest.json", manifest)
        checkpoints = json.loads((source / "checkpoint_manifest.json").read_text())
        checkpoints[0]["sha256"] = sha256_file(source / "policy.pkl")
        checkpoints[0]["training_state_sha256"] = sha256_file(source / "state.pt")
        write_json(source / "checkpoint_manifest.json", checkpoints)
        run.screen_worker(source, root, seed, smoke=True)
        sources.append(source)
        if seed < 2:
            with pytest.raises(FileNotFoundError):
                select(root, config)
    select(root, config)
    for seed, source in enumerate(sources):
        run.deploy_worker(source, root, seed, smoke=True, evaluation_workers=1)
    assert not list(root.rglob("*.pt"))
    output = aggregation.aggregate(root, smoke=True)
    report = json.loads((output / "summary.json").read_text())
    assert all(r["n_source_seeds"] == 3 for r in report["summaries"])
    assert len(report["selection"]["source_manifest_hashes"]) == 3
    path = root / "workers/seed_2/evaluation_summary.json"
    rows = json.loads(path.read_text())
    write_json(path, rows[:-1])
    with pytest.raises(ValueError, match="Incomplete gameplay"):
        aggregation.aggregate(root, smoke=True)
