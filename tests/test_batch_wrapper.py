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


def test_readme_documents_gcp_smoke_and_full_runs_for_all_experiments():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    batch_section = readme.split("## Google Cloud Batch", 1)[1].split(
        "## Verification", 1
    )[0]
    experiments = (
        (
            "exp1-fhp-ucv-baseline",
            "experiments.fhp.exp1_ucv_escher_baseline.run",
        ),
        (
            "exp2-fhp-ucv-sequential",
            "experiments.fhp.exp2_ucv_escher_sequential.run",
        ),
        (
            "exp3-fhp-ucv-parallel",
            "experiments.fhp.exp3_ucv_escher_parallel.run",
        ),
        (
            "exp4-fhp-ucv-cpu-optimized",
            "experiments.fhp.exp4_ucv_escher_cpu_optimized.run",
        ),
    )

    assert batch_section.count("#### GCP Batch smoke test") == 4
    assert batch_section.count("#### GCP Batch full run") == 4
    for job_prefix, module in experiments:
        assert f'JOB_NAME="{job_prefix}-smoke-' in batch_section
        assert f'JOB_NAME="{job_prefix}-full-' in batch_section
        commands = batch_section.split(f"python -m {module}", 2)
        assert len(commands) == 3
        assert "--smoke" in commands[1].split('"', 1)[0]
        assert "--smoke" not in commands[2].split('"', 1)[0]
