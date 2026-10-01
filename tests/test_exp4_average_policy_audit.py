import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fhp.exp4_fhp_average_policy_audit.config import contract
from experiments.fhp.exp4_fhp_average_policy_audit.aggregate import aggregate
from experiments.fhp.exp4_fhp_average_policy_audit.data import (
    group_replay, load_source, relative_path, split_groups,
)
from experiments.fhp.exp4_fhp_average_policy_audit.evaluation import build_tasks, task_fingerprint
from experiments.fhp.exp4_fhp_average_policy_audit.fitting import GroupSampler, fit_path, new_model, state_hash
from experiments.fhp.exp4_fhp_average_policy_audit.run import synthetic_source
from fhp_escher.checkpointing import LoadedFHPPolicy
from fhp_escher.game import load_fhp_game
from unbiased_escher.grouped_wide_solver import GroupedSoftTargetCrossEntropyAvePolicyTrainer


@pytest.fixture
def source(tmp_path):
    return synthetic_source(tmp_path / "source")


@pytest.fixture
def grouped(source):
    buffer, iteration, template, _ = load_source(source, 0)
    return group_replay(buffer, iteration), template


def test_source_checksum_and_seed_rejection(source):
    with pytest.raises(ValueError, match="seed"):
        load_source(source, 1)
    with (source / "state.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_source(source, 0)
    with pytest.raises(ValueError, match="escapes"):
        relative_path(source, "../escape.pt")


def test_grouping_identical_to_production_and_split_no_leakage(source):
    buffer, iteration, _, _ = load_source(source, 0)
    data = group_replay(buffer, iteration)
    fake_buffer = SimpleNamespace(cur_id=buffer["size"], buffer_size=buffer["size"],
                                  infostate_buf=buffer["infostate"], q_value_buf=buffer["q_value"],
                                  q_value_mask_buf=buffer["q_value_mask"], iteration_buf=buffer["iteration"])
    trainer = SimpleNamespace(buffer=fake_buffer, gamma=2.0, device="cpu")
    old = GroupedSoftTargetCrossEntropyAvePolicyTrainer._grouped_training_data(trainer, iteration)
    new = (data.features, data.targets, data.masks, (data.masses * len(data.features)/data.rows).astype(np.float32))
    for expected, observed in zip(old, new):
        np.testing.assert_array_equal(expected.numpy(), observed)
    training, heldout = split_groups(data, .1, 42)
    assert training.rows + heldout.rows == data.rows
    assert not ({row.tobytes() for row in training.features} & {row.tobytes() for row in heldout.features})


def test_invalid_masks_rejected(source):
    buffer, iteration, _, _ = load_source(source, 0)
    buffer["q_value_mask"][0] = 0
    with pytest.raises(ValueError, match="Invalid replay"):
        group_replay(buffer, iteration)


def test_mass_sampler_expected_gradient_and_loss_scale(grouped):
    groups, _ = grouped
    logits = torch.randn((len(groups.features), 3), dtype=torch.float64, requires_grad=True)
    targets = torch.tensor(groups.targets, dtype=torch.float64)
    losses = -(targets * torch.log_softmax(logits, -1)).sum(1)
    masses = torch.tensor(groups.masses)
    uniform_loss = (losses * masses * len(masses) / groups.rows).mean()
    mass_loss = (masses / masses.sum() * losses * masses.sum() / groups.rows).sum()
    np.testing.assert_allclose(uniform_loss.detach(), mass_loss.detach(), atol=1e-14)
    grad_a = torch.autograd.grad(uniform_loss, logits, retain_graph=True)[0]
    grad_b = torch.autograd.grad(mass_loss, logits)[0]
    torch.testing.assert_close(grad_a, grad_b, rtol=1e-13, atol=1e-14)
    draw = GroupSampler(groups, "mass", 9, 8)
    observed = np.zeros(len(groups.features))
    for _ in range(10_000):
        indices, weights = draw.sample()
        np.add.at(observed, indices, 1)
        np.testing.assert_allclose(weights, groups.masses.sum()/groups.rows)
    np.testing.assert_allclose(observed/observed.sum(), groups.masses/groups.masses.sum(), atol=.003)


def test_uniform_fit_matches_production_loop(grouped, tmp_path):
    torch.set_num_threads(1)
    groups, template = grouped
    config = contract(True)
    seed = config["initialisation_seed_base"]
    model = new_model(config, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    tensors = tuple(torch.from_numpy(a) for a in (groups.features, groups.targets, groups.masks,
                    (groups.masses * len(groups.features)/groups.rows).astype(np.float32)))
    trainer = SimpleNamespace(_grouped_training_data=lambda iteration: tensors,
                              model=model, optimizer=optimizer, device="cpu", batch_size=8, train_steps=4,
                              logger=SimpleNamespace(info=lambda message: None))
    random.seed(seed + 1000)
    GroupedSoftTargetCrossEntropyAvePolicyTrainer.train_model(trainer, 10)
    fit_path(groups, None, config=config, seed=0, sampler="uniform", directory=tmp_path / "fit", template=template)
    loaded = LoadedFHPPolicy(load_fhp_game(), tmp_path / "fit" / "4.pkl")
    assert state_hash(model) == state_hash(loaded.model)


@pytest.mark.parametrize("sampler", ["uniform", "mass"])
def test_resume_identical_to_uninterrupted(grouped, tmp_path, sampler):
    torch.set_num_threads(1)
    groups, template = grouped
    config = contract(True)
    def interruption():
        raise RuntimeError("Simulated interruption")
    with pytest.raises(RuntimeError, match="Simulated"):
        fit_path(groups, None, config=config, seed=0, sampler=sampler,
                 directory=tmp_path / "resume", template=template, on_checkpoint=interruption)
    # No optimizer state is saved. An incomplete path repeats deterministically.
    assert not list((tmp_path / "resume").glob("*.pt"))
    (tmp_path / "resume" / "2.pkl").unlink()
    fit_path(groups, None, config=config, seed=0, sampler=sampler, directory=tmp_path / "resume", template=template)
    assert (tmp_path / "resume" / "2.pkl").exists()
    fit_path(groups, None, config=config, seed=0, sampler=sampler, directory=tmp_path / "clean", template=template)
    a = LoadedFHPPolicy(load_fhp_game(), tmp_path / "resume" / "4.pkl")
    b = LoadedFHPPolicy(load_fhp_game(), tmp_path / "clean" / "4.pkl")
    assert state_hash(a.model) == state_hash(b.model)
    rows = json.loads((tmp_path / "resume" / "metrics.json").read_text())
    assert [r["updates"] for r in rows] == [2, 4]
    assert not list((tmp_path / "resume").glob("*.pt"))
    before = (tmp_path / "resume" / "metrics.json").read_bytes()
    fit_path(groups, None, config=config, seed=0, sampler=sampler,
             directory=tmp_path / "resume", template=template, on_checkpoint=interruption)
    assert (tmp_path / "resume" / "metrics.json").read_bytes() == before


def test_evaluation_common_random_numbers_and_cache_identity(source):
    config = contract(True)
    policies = {name: source / "policy.pkl" for name in ("archived", "uniform_2", "uniform_4", "mass_2", "mass_4")}
    tasks = build_tasks(policies, config, 0)
    lbr = [t for t in tasks if t["kind"] == "lbr"]
    assert len(lbr) == 5
    assert len({t["evaluation_seed"] for t in lbr}) == 1
    relocated = dict(lbr[0], policy_a_path="/different/location.pkl")
    assert task_fingerprint(relocated) == task_fingerprint(lbr[0])
    changed = dict(lbr[0], policy_a_sha256="different")
    assert task_fingerprint(changed) != task_fingerprint(lbr[0])
    assert len([t for t in tasks if t["kind"] == "direct_crossplay"]) == 7


def test_cloud_jobs_safe_routing_and_caps(tmp_path):
    root = Path(__file__).parents[1]
    for kind in ("controller", "smoke", "train", "aggregate"):
        target = tmp_path / f"{kind}.json"
        subprocess.run([sys.executable, str(root / "gcp/exp4_average_policy_audit_batch.py"),
                        "--kind", kind, "--output", str(target), "--run-id", "exp4-test", "--bucket-root", "gs://bucket",
                        "--service-account", "runner@test.iam.gserviceaccount.com", "--repo-ref", "abc",
                        "--source-run-id", "exp2-fhp-source", "--project-id", "test", "--region", "europe-west1"], check=True)
        job = json.loads(target.read_text())
        group = job["taskGroups"][0]
        text = group["taskSpec"]["runnables"][0]["script"]["text"]
        assert group["taskSpec"]["maxRetryCount"] == 0
        assert group["taskCount"] == (3 if kind == "train" else 1)
        assert group["taskCountPerNode"] == 1
        assert "exp1_fhp_grouped_wide_ucv_baseline.run" not in text
        assert "run_exp1_grouped_wide.sh" not in text
        if kind == "train":
            assert "EXP4_REMOTE_WORKER" in text
            assert "fetch-source" in text
            assert group["taskSpec"]["maxRunDuration"] == "86400s"
        if kind == "controller":
            assert "export EXP2_RUN_ID=exp2-fhp-source" in text
        script = tmp_path / f"{kind}.sh"
        script.write_text(text)
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_three_seed_aggregation_and_factorial_estimands(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    config = contract()
    for seed in (0, 1, 2):
        worker = tmp_path / "workers" / f"seed_{seed}"
        worker.mkdir(parents=True)
        def save(name, value):
            path = worker / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        save("manifest.json", {"config": config})
        save("SUCCESS.json", {"seed": seed})
        save("replay_diagnostics.json", {})
        save("resources.json", {})
        for phase in ("deployment", "diagnostic"):
            for sampler in config["samplers"]:
                rows = []
                for updates in config["updates"]:
                    value = 1 + seed*.01 - (sampler == "mass")*.2 - (updates == 60_000)*.1
                    metrics = {"all": {"weighted_ce": value, "weighted_kl": value, "weighted_l1": value}}
                    rows.append({"arm": f"{sampler}_{updates}", "seed": seed,
                                 "fitting_seconds": 1, "diagnostic_seconds": 1, "processed_examples": updates*2048,
                                 "training": metrics, "validation": metrics if phase == "diagnostic" else None})
                save(f"{phase}/{sampler}/metrics.json", rows)
        save("evaluation_summary.json", [{"arm": arm, "metric": "lbr_mbb_per_hand", "mean_mbb_per_hand": 10+seed}
                                         for arm in ("archived", "uniform_20000", "uniform_60000", "mass_20000", "mass_60000")])
    analysis = aggregate(tmp_path)
    payload = json.loads((analysis / "summary.json").read_text())
    assert all(row["n_source_seeds"] == 3 for row in payload["summaries"])
    effects = {row["candidate"]: row for row in payload["paired_contrasts"]
               if row["metric"] == "deployment_training_weighted_ce"}
    assert effects["sampler_main_effect_mass_minus_uniform"]["mean"] == pytest.approx(-.2)
    assert effects["update_main_effect_high_minus_low"]["mean"] == pytest.approx(-.1)
    assert effects["interaction_mass_increment_minus_uniform_increment"]["mean"] == pytest.approx(0)
    assert effects["sampler_main_effect_mass_minus_uniform"]["exact_two_sided_sign_flip_p"] == .25
    (tmp_path / "workers/seed_2/SUCCESS.json").unlink()
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path)
