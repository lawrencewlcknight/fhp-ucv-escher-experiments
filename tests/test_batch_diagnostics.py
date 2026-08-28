from fhp_escher import batch_diagnostics


def _snapshot(*, current: int, peak: int, limit: int, oom_kill: int):
    return {
        "timestamp_utc": "2026-08-09T00:00:00+00:00",
        "cgroup_memory": {
            "version": 2,
            "current_bytes": current,
            "peak_bytes": peak,
            "limit_bytes": limit,
            "events": {"oom": oom_kill, "oom_kill": oom_kill},
        },
        "system_memory": {"memavailable_bytes": limit - current},
        "largest_processes": [{"rss_bytes": current // 2}],
    }


def test_diagnose_confirms_cgroup_oom(monkeypatch):
    limit = 32_000 * 1024 * 1024
    first = _snapshot(current=limit // 4, peak=limit // 4, limit=limit, oom_kill=0)
    final = _snapshot(current=limit // 8, peak=limit, limit=limit, oom_kill=1)
    monkeypatch.setattr(batch_diagnostics, "take_snapshot", lambda: final)

    result = batch_diagnostics.diagnose(
        snapshots=[first],
        exit_code=137,
        experiment_exit_code=137,
        requested_memory_mib=32_000,
        failure_files=[],
    )

    assert result["diagnosis"] == "confirmed_cgroup_out_of_memory"
    assert result["cgroup_event_delta"]["oom_kill"] == 1
    assert result["cgroup_memory_peak_fraction"] == 1.0
    assert result["signal_number"] == 9


def test_diagnose_uses_failure_file_for_python_exception(monkeypatch, tmp_path):
    limit = 32_000 * 1024 * 1024
    snapshot = _snapshot(
        current=limit // 4,
        peak=limit // 3,
        limit=limit,
        oom_kill=0,
    )
    monkeypatch.setattr(batch_diagnostics, "take_snapshot", lambda: snapshot)
    failure_file = tmp_path / "failure.json"
    failure_file.write_text(
        '{"exception_type": "RuntimeError", "message": "training failed"}',
        encoding="utf-8",
    )

    result = batch_diagnostics.diagnose(
        snapshots=[snapshot],
        exit_code=1,
        experiment_exit_code=1,
        requested_memory_mib=32_000,
        failure_files=[failure_file],
    )

    assert result["diagnosis"] == "python_exception"
    assert result["experiment_failure_files"][0]["payload"]["exception_type"] == (
        "RuntimeError"
    )


def test_diagnose_recognises_reported_allocator_failure(monkeypatch, tmp_path):
    limit = 32_000 * 1024 * 1024
    snapshot = _snapshot(
        current=limit // 2,
        peak=limit * 3 // 4,
        limit=limit,
        oom_kill=0,
    )
    monkeypatch.setattr(batch_diagnostics, "take_snapshot", lambda: snapshot)
    failure_file = tmp_path / "failure.json"
    failure_file.write_text(
        '{"exception_type": "RuntimeError", '
        '"message": "DefaultCPUAllocator: cannot allocate memory"}',
        encoding="utf-8",
    )

    result = batch_diagnostics.diagnose(
        snapshots=[snapshot],
        exit_code=1,
        experiment_exit_code=1,
        requested_memory_mib=32_000,
        failure_files=[failure_file],
    )

    assert result["diagnosis"] == "reported_out_of_memory_exception"


def test_cgroup_cpu_counters_are_exposed_in_resource_heartbeats(monkeypatch, tmp_path):
    (tmp_path / "cpu.stat").write_text(
        "usage_usec 123456\nuser_usec 100000\nsystem_usec 23456\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        batch_diagnostics,
        "_candidate_cgroup_directories",
        lambda: iter((tmp_path,)),
    )

    cpu = batch_diagnostics._cgroup_cpu()
    assert cpu["version"] == 2
    assert cpu["stat"]["usage_usec"] == 123456
    heartbeat = batch_diagnostics._heartbeat(
        {
            "timestamp_utc": "2026-08-09T00:00:00+00:00",
            "load_average": [1.0, 2.0, 3.0],
            "logical_cpu_count": 32,
            "cgroup_memory": {},
            "cgroup_cpu": cpu,
            "system_memory": {},
            "largest_processes": [],
        }
    )["resource_heartbeat"]
    assert heartbeat["logical_cpu_count"] == 32
    assert heartbeat["cgroup_cpu"] == cpu
