#!/usr/bin/env python3
"""Experiment 30 Batch entry point."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp.strategic_features_batch import main
if __name__ == "__main__":
    main(experiment=30)
