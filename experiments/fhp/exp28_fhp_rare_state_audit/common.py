"""Atomic, checksummed audit outputs; no silent resumption of partial experiments."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time


def write_json(path, value):
    """Stdlib-only so submission/source preflight needs no ML environment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3], text=True).strip()


def complete(directory, **metadata):
    root = Path(directory)
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != "SUCCESS.json" and not p.name.endswith(".tmp")}
    result = dict(status="complete", artifacts=files, **metadata)
    write_json(root / "SUCCESS.json", result)
    return result


def verify(directory):
    root = Path(directory)
    row = json.loads((root / "SUCCESS.json").read_text())
    if row.get("status") != "complete" or not row.get("artifacts"):
        raise ValueError("Incomplete audit stage")
    for name, expected in row["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Audit artifact checksum/path mismatch: {name}")
    return row


@contextmanager
def timed(costs, name):
    started = time.perf_counter()
    try:
        yield
    finally:
        costs[name] = costs.get(name, 0.) + time.perf_counter()-started


def fresh(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory
