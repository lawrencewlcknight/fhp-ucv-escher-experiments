"""Local entry point; production GCP submission uses the gcp shell wrapper."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=ROOT.parents[1] / "fhp-evaluation-suite")
    parser.add_argument("--input-root", required=True, help="Nine trusted, hash-pinned playable .pkl files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    suite = args.suite_root.resolve()
    if not (suite / "fhp_evaluation/resolving/runner.py").is_file():
        parser.error("The shared evaluation checkout must contain the Experiment 23 implementation")
    command = [sys.executable, "-m", "fhp_evaluation.resolving.runner", "--input-root",
               str(Path(args.input_root).resolve()), "--native-root", str(ROOT),
               "--output", str(Path(args.output).resolve())]
    if args.preflight_only:
        command.append("--preflight-only")
    env = dict(os.environ, PYTHONPATH=str(suite), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
    subprocess.run(command, cwd=suite, env=env, check=True)


if __name__ == "__main__":
    main()
