"""Hash-pinned metadata, configuration audit and complete native policy validation."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

PINS = json.loads(Path(__file__).with_name("source_pins.json").read_text())


@contextmanager
def cohort_contract(engine, algorithm, cohort):
    pin = PINS[algorithm][cohort]
    replacements = dict(ALGORITHMS=(algorithm,),
        SOURCE_NAMES={algorithm: tuple(pin["source_names"])},
        SOURCE_COMMITS={algorithm: pin["source_commit"]},
        CONFIG_SHA256={algorithm: pin["config_sha256"]},
        RUN_IDS={algorithm: pin["run_id"]}, LABELS={algorithm: f"{algorithm.upper()} {cohort} 48h"})
    saved = {key: getattr(engine, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(engine, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(engine, key, value)


def validate_metadata(root, engine):
    algorithm = engine.REPRESENTATION_ALGORITHM
    manifests = {}
    context = []
    for cohort in ("original", "rich"):
        pin = PINS[algorithm][cohort]
        with cohort_contract(engine, algorithm, cohort):
            for seed in range(3):
                worker = engine._worker(Path(root)/cohort/algorithm, algorithm, seed)
                if (worker/"FAILURE.json").exists():
                    raise ValueError("Failed source worker")
                for name, expected in pin["metadata"][str(seed)].items():
                    if engine.sha256(worker/name) != expected:
                        raise ValueError(f"Pinned {algorithm}/{cohort}/{seed} metadata mismatch: {name}")
                m = engine.read_json(worker/"run_manifest.json")
                engine._validate_common(m, algorithm, seed)
                row = engine._final_row(engine.read_json(worker/"checkpoint_manifest.json"), algorithm)
                if (m.get("smoke", m.get("is_smoke", False)) or m.get("feature_encoder") != pin["feature_encoder"]):
                    raise ValueError("Unexpected smoke flag or feature representation")
                manifests[cohort, seed] = m
                context.append(dict(algorithm=algorithm, cohort=cohort, seed=seed, source_run=pin["run_id"],
                    source_commit=pin["source_commit"], nodes_touched=row["nodes_touched"],
                    outer_iteration=row.get("outer_iteration", row.get("iteration")),
                    active_seconds=row["actual_training_elapsed_seconds"],
                    feature_encoder=m["feature_encoder"],
                    clock={k: m[k] for k in ("time_excludes", "training_clock", "checkpoint_boundary",
                        "archive_capture_included_in_training_time", "parallel_startup_included_in_training_time") if k in m},
                    runtime={k: m[k] for k in ("runtime", "python_version", "torch_version", "numpy_version",
                        "torch_threads", "interop_threads", "ray_version") if k in m}))
    for seed in range(3):
        old, rich = (manifests[c, seed] for c in ("original", "rich"))
        old_config, rich_config = (dict(m.get("config", m.get("training_config"))) for m in (old, rich))
        if old_config.pop("feature_encoder_id", None) not in (None, "fhp_lossless_suit_canonical_v1"):
            raise ValueError("Unexpected original encoder")
        if rich_config.pop("feature_encoder_id", None) != "fhp_hand_board_strategic_ae_v1" or old_config != rich_config:
            raise ValueError("Training configuration differs beyond representation")
        for key in ("time_excludes", "training_clock", "checkpoint_boundary", "torch_threads",
                    "parallel_traversal_workers", "parallel_execution", "strategy_weighting",
                    "archive_capture_included_in_training_time", "parallel_startup_included_in_training_time"):
            if old.get(key) != rich.get(key):
                raise ValueError(f"Training execution contract changed: {key}")
    return context


def validate(root, engine, sd_repo, ucv_repo):
    validate_metadata(root, engine)
    algorithm, records = engine.REPRESENTATION_ALGORITHM, []
    for cohort in ("original", "rich"):
        with cohort_contract(engine, algorithm, cohort):
            rows = engine.validate_sources(Path(root)/cohort, sd_repo, ucv_repo)
        for row in rows:
            row.update(cohort=cohort, id=f"{algorithm}_{cohort}_s{row['seed']}_48h",
                feature_encoder_id=PINS[algorithm][cohort]["feature_encoder"]["id"],
                player_features=PINS[algorithm][cohort]["feature_encoder"]["policy_layout"]["total_size"])
        records.extend(rows)
    return records


def cost_context(root, records, engine):
    return validate_metadata(root, engine)


def main():
    from . import engine as native
    from .run import ROOT
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "vr-repo", "sd-repo"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--algorithm", choices=("sd", "vr"), required=True)
    args = p.parse_args()
    engine = native.load(args.vr_repo, args.sd_repo, ROOT, args.algorithm)
    validate_metadata(args.source_root, engine)
    print("All six pinned sources verified; training configs differ only by representation.")


if __name__ == "__main__":
    main()
