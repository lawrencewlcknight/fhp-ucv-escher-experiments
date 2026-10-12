"""Evaluate frozen Exp16 policies with/without an exact terminal-call safeguard."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=ROOT.parents[1]/"fhp-evaluation-suite")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if not (args.suite_root/"fhp_evaluation/safeguard/runner.py").is_file():
        raise ValueError("Updated shared fhp-evaluation-suite checkout required")
    env = dict(os.environ, PYTHONPATH=str(args.suite_root.resolve()), OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
    subprocess.run([sys.executable, "-m", "fhp_evaluation.safeguard.runner", "--native-root", str(ROOT),
                    "--input-root", str(args.input_root.resolve()), "--output", str(args.output.resolve()),
                    "--mode", args.mode], env=env, cwd=args.suite_root, check=True)


if __name__ == "__main__":
    main()
