from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "run_exp1_grouped_wide.sh",
    "run_exp2_lossless_structured.sh",
    "run_exp3_wider_structured.sh",
)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_cloud_submission_rejects_missing_service_account(tmp_path, launcher):
    fake_gcloud = tmp_path / "gcloud"
    fake_gcloud.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'NOT_FOUND: service account does not exist' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "PROJECT_ID": "test-project",
            "REGION": "europe-west1",
            "BUCKET": "gs://test-bucket",
            "SA_EMAIL": "missing@test-project.iam.gserviceaccount.com",
            "REPO_REF": "deadbeef",
            "RUN_ID": f"test-{launcher.removesuffix('.sh').replace('_', '-')}",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "gcp" / launcher), "run"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "does not exist or is inaccessible" in result.stderr
    assert "NOT_FOUND" in result.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_controller_polling_has_explicit_permission_failure_guard(launcher):
    source = (ROOT / "gcp" / launcher).read_text(encoding="utf-8")
    assert "preflight_remote_controller" in source
    assert "Remote controller cannot inspect Batch jobs" in source
    assert "Unable to inspect Batch job $1; aborting controller" in source
    assert 'submit_job "$1" "$2" || return $?' in source
