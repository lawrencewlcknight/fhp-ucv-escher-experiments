import json
import os
from pathlib import Path
import subprocess


def test_generated_batch_script_has_valid_shell_and_diagnostics(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    submit_script = (root / "gcp" / "submit_batch_experiment.sh").read_text(
        encoding="utf-8"
    )
    marker = "python3 <<'PY'\n"
    start = submit_script.index(marker) + len(marker)
    end = submit_script.index("\nPY\n", start)
    generator = submit_script[start:end]
    job_json = tmp_path / "job.json"
    environment = {
        "JOB_JSON": str(job_json),
        "JOB_NAME": "diagnostics-shell-test",
        "EXPERIMENT_COMMAND": (
            "python -m experiments.fhp.exp1_ucv_escher_baseline.run"
        ),
        "MACHINE_TYPE": "n2-standard-8",
        "MAX_RUN_SECONDS": "50400",
        "CPU_MILLI": "8000",
        "MEMORY_MIB": "32000",
        "BOOT_DISK_SIZE_GB": "100",
        "BUCKET": "gs://example-results",
        "SA_EMAIL": "batch@example.invalid",
        "REPO_URL": "https://example.invalid/repository.git",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    namespace = {"__builtins__": __builtins__}
    exec(compile(generator, str(root / "gcp" / "submit_batch_experiment.sh"), "exec"), namespace)
    job = json.loads(job_json.read_text(encoding="utf-8"))
    generated = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]

    result = subprocess.run(
        ["bash", "-n"],
        input=generated,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    assert "--interval-seconds 15" in generated
    assert "--cloud-log-every 4" in generated
    assert "batch_diagnostics.json" in generated
    assert "--experiment-exit-code" in generated
