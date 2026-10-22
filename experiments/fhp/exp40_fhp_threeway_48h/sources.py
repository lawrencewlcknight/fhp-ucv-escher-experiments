"""Native endpoint checks, including Experiment 36 -> 38 continuation history."""
from pathlib import Path

from experiments.fhp.exp38_fhp_strength_critic2500_48h import contract
from .engine import LABELS, RUN_IDS


def validate_ucv(root, source, engine):
    from fhp_escher.checkpointing import LoadedFHPPolicy
    from fhp_escher.game import FHP_GAME_PARAMETERS, load_fhp_game
    game, records = load_fhp_game(), []
    for seed in contract.SEEDS:
        worker = Path(root) / "workers" / contract.worker_name(seed)
        parent = Path(source) / "workers" / contract.worker_name(seed)
        data = []
        for directory, hours in ((parent, 24), (worker, 48)):
            if (directory / "FAILURE.json").exists():
                raise ValueError("Failed UCV source worker")
            raw = {name: (directory / name).read_bytes() for name in contract.METADATA_FILES}
            data.append(contract.validate_metadata(raw, seed=seed, hours=hours))
        before, after = data
        rows, old = after["checkpoint_manifest.json"], before["checkpoint_manifest.json"]
        lineage = engine.read_json(worker / "continuation_source.json")
        expected = dict(seed=seed, source_total_hours=24, total_hours=48,
            source_commit=contract.TRAINING_REF, source_state_path=old[-1]["training_state_path"],
            source_state_sha256=old[-1]["training_state_sha256"],
            source_summary_sha256=before["SUCCESS.json"]["summary_sha256"])
        if (any(lineage.get(k) != v for k, v in expected.items()) or not
                lineage.get("source_worker", "").endswith(f"/{RUN_IDS['ucv_source36']}/workers/{parent.name}")):
            raise ValueError("Invalid Experiment 36 -> 38 continuation lineage")
        for previous, imported in zip(old, rows[:4]):
            for key in ("sha256", "outer_iteration", "nodes_touched", "actual_training_elapsed_seconds"):
                if previous[key] != imported[key]:
                    raise ValueError("UCV imported 6-24h policy history differs from its source")
        final = rows[-1]
        if any(final[k] <= old[-1][k] for k in ("nodes_touched", "outer_iteration")):
            raise ValueError("No additional training after the 24h restore")
        path = engine.contained(worker, final["path"])
        if engine.sha256(path) != final["sha256"]:
            raise ValueError("UCV final policy checksum mismatch")
        if "size_bytes" in final and path.stat().st_size != final["size_bytes"]:
            raise ValueError("UCV final policy size mismatch")
        manifest = after["run_manifest.json"]
        payload = LoadedFHPPolicy(game, path).checkpoint
        fields = dict(seed=seed, algorithm_id=contract.ALGORITHM_ID,
            experiment_name=contract.LEARNER_NAME, checkpoint_target_seconds=48*3600,
            nodes_touched=final["nodes_touched"], outer_iteration=final["outer_iteration"],
            feature_encoder=manifest["feature_encoder"])
        if (any(payload.get(k) != v for k, v in fields.items()) or
                contract.digest(payload.get("training_config")) != contract.CONFIG_SHA256 or
                manifest.get("game", {}).get("parameters") != dict(FHP_GAME_PARAMETERS)):
            raise ValueError("UCV native checkpoint payload/configuration mismatch")
        records.append(dict(id=f"ucv_s{seed}_48h", algorithm="ucv", label=LABELS["ucv"],
            seed=seed, training_hours=48, active_seconds=final["actual_training_elapsed_seconds"],
            nodes_touched=final["nodes_touched"], outer_iteration=final["outer_iteration"],
            path=str(path), sha256=final["sha256"], source_run=RUN_IDS["ucv"],
            source_commit=contract.TRAINING_REF, policy_size_bytes=path.stat().st_size,
            metadata_sha256={name: engine.sha256(worker/name) for name in contract.METADATA_FILES},
            parent_metadata_sha256={name: engine.sha256(parent/name) for name in contract.METADATA_FILES},
            continuation_source_sha256=engine.sha256(worker/"continuation_source.json")))
    return records


def validate(root, engine, sd_repo, ucv_repo):
    # Reuse the native SD/VR validators, not the old Exp16 UCV validator.
    original = engine.ALGORITHMS
    try:
        engine.ALGORITHMS = ("sd", "vr")
        records = engine.validate_sources(root, sd_repo, ucv_repo)
    finally:
        engine.ALGORITHMS = original
    records += validate_ucv(Path(root)/"ucv", Path(root)/"ucv_source36", engine)
    for row in records:
        path = Path(row["path"])
        if row["algorithm"] == "sd":
            archive = engine.read_json(path)
            row["archive_models_per_player"] = row["outer_iteration"]
            row["policy_size_bytes"] = path.stat().st_size + sum(
                engine.contained(path.parent, c["path"]).stat().st_size for c in archive["chunks"])
        else:
            row["policy_size_bytes"] = path.stat().st_size
    if sorted((r["algorithm"], r["seed"]) for r in records) != [
            (a, s) for a in ("sd", "ucv", "vr") for s in (0, 1, 2)]:
        raise ValueError("Exactly three seeds per algorithm are required")
    return records


def cost_context(root, records, engine):
    result = []
    for record in records:
        algorithm, seed = record["algorithm"], record["seed"]
        name = contract.worker_name(seed) if algorithm == "ucv" else engine._worker(root, algorithm, seed).name
        worker = Path(root)/algorithm/"workers"/name
        # SD stores its native timing/progress summary directly in SUCCESS;
        # unlike UCV/VR, its completed workers have no summary.json.
        summary_name = "SUCCESS.json" if algorithm == "sd" else "summary.json"
        summary = engine.read_json(worker/summary_name)
        result.append(dict(algorithm=algorithm, seed=seed, source_run=record["source_run"],
            policy_size_bytes=record["policy_size_bytes"],
            source_summary_file=summary_name, source_summary_sha256=engine.sha256(worker/summary_name),
            native_training_summary=summary))
    return result
