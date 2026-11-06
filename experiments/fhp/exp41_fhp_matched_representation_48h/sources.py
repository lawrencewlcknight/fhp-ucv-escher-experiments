"""Freeze Exp8 cohorts and verify native full policies before any scoring."""
import argparse
import json
from pathlib import Path

from experiments.fhp.exp40_fhp_threeway_48h import sources as previous

PINS = json.loads(Path(__file__).with_name("source_pins.json").read_text())


def validate_metadata(root, engine):
    """Cheap rejection before downloading the SD historical archives."""
    from fhp_escher.features import make_feature_encoder
    expected_encoder = make_feature_encoder("fhp_hand_board_strategic_ae_v1").metadata()
    root = Path(root)
    for algorithm in ("sd", "vr"):
        pin = PINS[algorithm]
        if pin["run_id"] != engine.RUN_IDS[algorithm]:
            raise ValueError("Pinned source run differs from evaluator")
        for seed in (0, 1, 2):
            worker = engine._worker(root / algorithm, algorithm, seed)
            if (worker / "FAILURE.json").exists():
                raise ValueError("Failed source worker")
            for name, expected in pin["metadata"][str(seed)].items():
                if engine.sha256(worker / name) != expected:
                    raise ValueError(f"Pinned {algorithm} seed {seed} metadata mismatch: {name}")
            manifest = engine.read_json(worker / "run_manifest.json")
            engine._validate_common(manifest, algorithm, seed)
            engine._final_row(engine.read_json(worker / "checkpoint_manifest.json"), algorithm)
            if manifest.get("feature_encoder") != expected_encoder:
                raise ValueError("Source does not use the matched rich AE representation")
            context = manifest.get("experiment_metadata", manifest)
            if context.get("initialisation") != "fresh_from_scratch_no_warm_start":
                raise ValueError("SD/VR cohorts must be fresh training runs")
    for seed in (0, 1, 2):
        worker = root / "ucv" / "workers" / previous.contract.worker_name(seed)
        if engine.read_json(worker / "run_manifest.json").get("feature_encoder") != expected_encoder:
            raise ValueError("UCV encoder differs from matched SD/VR encoders")


def validate(root, engine, sd_repo, ucv_repo):
    validate_metadata(root, engine)
    # The existing validator checks SD archive coverage, every chunk checksum,
    # uniform weighting and native payloads, plus UCV's full 36 -> 38 lineage.
    records = previous.validate(root, engine, sd_repo, ucv_repo)
    for record in records:
        record["label"] = engine.LABELS[record["algorithm"]]
        record["feature_encoder_id"] = "fhp_hand_board_strategic_ae_v1"
        record["matched_player_features"] = 231
    return records


cost_context = previous.cost_context


def main():
    from . import engine as native
    from .run import ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "vr-repo", "sd-repo"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    validate_metadata(args.source_root, native.load(args.vr_repo, args.sd_repo, ROOT))
    print("Pinned SD/VR metadata and shared rich AE representation verified.")


if __name__ == "__main__":
    main()
