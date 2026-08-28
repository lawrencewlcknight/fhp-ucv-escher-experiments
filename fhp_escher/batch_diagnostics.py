"""Independent resource monitoring and failure diagnosis for Batch runs.

The monitor runs outside the training process so its snapshots normally survive
an abrupt SIGKILL of the experiment, including a Linux cgroup OOM kill.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import threading
from typing import Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    value = _read_text(path)
    if value is None or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _key_value_file(path: Path) -> dict[str, int]:
    contents = _read_text(path)
    values: dict[str, int] = {}
    if not contents:
        return values
    for line in contents.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def _candidate_cgroup_directories() -> Iterable[Path]:
    root = Path("/sys/fs/cgroup")
    candidates: list[Path] = []
    membership = _read_text(Path("/proc/self/cgroup")) or ""
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, relative = parts[1], parts[2].lstrip("/")
        if controllers == "":
            candidates.append(root / relative)
        elif "memory" in controllers.split(","):
            candidates.extend((root / "memory" / relative, root / relative))
    candidates.extend((root, root / "memory"))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen and candidate.exists():
            seen.add(candidate)
            yield candidate


def _cgroup_memory() -> dict[str, object]:
    for base in _candidate_cgroup_directories():
        if (base / "memory.current").exists():
            return {
                "version": 2,
                "path": str(base),
                "current_bytes": _read_int(base / "memory.current"),
                "peak_bytes": _read_int(base / "memory.peak"),
                "limit_bytes": _read_int(base / "memory.max"),
                "swap_current_bytes": _read_int(base / "memory.swap.current"),
                "swap_peak_bytes": _read_int(base / "memory.swap.peak"),
                "events": _key_value_file(base / "memory.events"),
                "events_local": _key_value_file(base / "memory.events.local"),
                "pressure": _read_text(base / "memory.pressure"),
            }
        if (base / "memory.usage_in_bytes").exists():
            oom_control = _key_value_file(base / "memory.oom_control")
            fail_count = _read_int(base / "memory.failcnt")
            events = {"failcnt": fail_count or 0, **oom_control}
            return {
                "version": 1,
                "path": str(base),
                "current_bytes": _read_int(base / "memory.usage_in_bytes"),
                "peak_bytes": _read_int(base / "memory.max_usage_in_bytes"),
                "limit_bytes": _read_int(base / "memory.limit_in_bytes"),
                "swap_current_bytes": _read_int(base / "memory.memsw.usage_in_bytes"),
                "swap_peak_bytes": _read_int(base / "memory.memsw.max_usage_in_bytes"),
                "events": events,
                "pressure": None,
            }
    return {"available": False}


def _cgroup_cpu() -> dict[str, object]:
    """Return cumulative cgroup CPU counters for utilisation calculations."""
    for base in _candidate_cgroup_directories():
        cpu_stat = base / "cpu.stat"
        if cpu_stat.exists():
            return {
                "version": 2,
                "path": str(base),
                "stat": _key_value_file(cpu_stat),
            }
        cpuacct_usage = _read_int(base / "cpuacct.usage")
        if cpuacct_usage is not None:
            return {
                "version": 1,
                "path": str(base),
                "usage_nanoseconds": cpuacct_usage,
            }
    return {"available": False}


def _system_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    contents = _read_text(Path("/proc/meminfo")) or ""
    for line in contents.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        parts = remainder.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        values[f"{key.lower()}_bytes"] = value * 1024
    return values


def _process_snapshot(limit: int = 20) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return rows
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        status = _read_text(process_dir / "status")
        if not status:
            continue
        fields: dict[str, str] = {}
        for line in status.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()

        def kib_field(name: str) -> int:
            try:
                return int(fields.get(name, "0 kB").split()[0]) * 1024
            except (ValueError, IndexError):
                return 0

        stat = _read_text(process_dir / "stat") or ""
        stat_tail = stat[stat.rfind(")") + 2 :].split() if ")" in stat else []

        def stat_field(index: int) -> int:
            try:
                return int(stat_tail[index])
            except (ValueError, IndexError):
                return 0

        rows.append(
            {
                "pid": int(process_dir.name),
                "ppid": int(fields.get("PPid", "0")),
                "name": fields.get("Name", "unknown"),
                "rss_bytes": kib_field("VmRSS"),
                "peak_rss_bytes": kib_field("VmHWM"),
                "virtual_bytes": kib_field("VmSize"),
                "threads": int(fields.get("Threads", "0")),
                "user_cpu_ticks": stat_field(11),
                "system_cpu_ticks": stat_field(12),
            }
        )
    rows.sort(key=lambda row: int(row["rss_bytes"]), reverse=True)
    return rows[:limit]


def take_snapshot() -> dict[str, object]:
    disk = shutil.disk_usage("/")
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = []
    return {
        "timestamp_utc": _utc_now(),
        "load_average": load_average,
        "logical_cpu_count": os.cpu_count(),
        "system_memory": _system_memory(),
        "cgroup_memory": _cgroup_memory(),
        "cgroup_cpu": _cgroup_cpu(),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "largest_processes": _process_snapshot(),
    }


def _heartbeat(snapshot: dict[str, object]) -> dict[str, object]:
    cgroup = snapshot.get("cgroup_memory", {})
    cgroup_cpu = snapshot.get("cgroup_cpu", {})
    system = snapshot.get("system_memory", {})
    processes = snapshot.get("largest_processes", [])
    largest_process = processes[0] if isinstance(processes, list) and processes else None
    return {
        "resource_heartbeat": {
            "timestamp_utc": snapshot.get("timestamp_utc"),
            "load_average": snapshot.get("load_average"),
            "logical_cpu_count": snapshot.get("logical_cpu_count"),
            "cgroup_memory": cgroup,
            "cgroup_cpu": cgroup_cpu,
            "system_available_bytes": (
                system.get("memavailable_bytes") if isinstance(system, dict) else None
            ),
            "largest_process": largest_process,
        }
    }


def monitor(output: Path, interval_seconds: float, cloud_log_every: int = 4) -> int:
    if interval_seconds <= 0:
        raise ValueError("Monitor interval must be positive")
    if cloud_log_every <= 0:
        raise ValueError("Cloud-log frequency must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()

    def request_stop(_signum, _frame):
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    snapshot_index = 0
    with open(output, "a", encoding="utf-8", buffering=1) as handle:
        while not stopped.is_set():
            try:
                snapshot = take_snapshot()
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
                handle.flush()
                if snapshot_index % cloud_log_every == 0:
                    print(json.dumps(_heartbeat(snapshot), sort_keys=True), flush=True)
                snapshot_index += 1
            except BaseException as exc:  # Keep monitoring after transient /proc races.
                error = {
                    "timestamp_utc": _utc_now(),
                    "monitor_error": f"{type(exc).__name__}: {exc}",
                }
                handle.write(json.dumps(error, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(error, sort_keys=True), flush=True)
            stopped.wait(interval_seconds)
    return 0


def _load_snapshots(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    snapshots = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "cgroup_memory" in row:
            snapshots.append(row)
    return snapshots


def _event_delta(
    first: dict[str, object], last: dict[str, object]
) -> dict[str, int]:
    first_events = first.get("events", {})
    last_events = last.get("events", {})
    if not isinstance(first_events, dict) or not isinstance(last_events, dict):
        return {}
    return {
        str(key): int(value) - int(first_events.get(key, 0))
        for key, value in last_events.items()
        if isinstance(value, int)
    }


def _optional_exit_code(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def diagnose(
    *,
    snapshots: list[dict[str, object]],
    exit_code: int,
    experiment_exit_code: int | None,
    requested_memory_mib: int,
    failure_files: list[Path],
) -> dict[str, object]:
    final_snapshot = take_snapshot()
    all_snapshots = [*snapshots, final_snapshot]
    first_cgroup = all_snapshots[0].get("cgroup_memory", {})
    last_cgroup = all_snapshots[-1].get("cgroup_memory", {})
    if not isinstance(first_cgroup, dict):
        first_cgroup = {}
    if not isinstance(last_cgroup, dict):
        last_cgroup = {}
    event_delta = _event_delta(first_cgroup, last_cgroup)

    peak_candidates = []
    maximum_current_candidates = []
    minimum_available_candidates = []
    maximum_process_candidates = []
    for snapshot in all_snapshots:
        cgroup = snapshot.get("cgroup_memory", {})
        system = snapshot.get("system_memory", {})
        processes = snapshot.get("largest_processes", [])
        if isinstance(cgroup, dict):
            if isinstance(cgroup.get("peak_bytes"), int):
                peak_candidates.append(cgroup["peak_bytes"])
            if isinstance(cgroup.get("current_bytes"), int):
                maximum_current_candidates.append(cgroup["current_bytes"])
        if isinstance(system, dict) and isinstance(system.get("memavailable_bytes"), int):
            minimum_available_candidates.append(system["memavailable_bytes"])
        if isinstance(processes, list):
            rss_values = [
                process.get("rss_bytes", 0)
                for process in processes
                if isinstance(process, dict)
                and isinstance(process.get("rss_bytes", 0), int)
            ]
            if rss_values:
                maximum_process_candidates.append(max(rss_values))

    requested_bytes = requested_memory_mib * 1024 * 1024
    cgroup_limit = last_cgroup.get("limit_bytes")
    effective_limit = requested_bytes
    if isinstance(cgroup_limit, int) and cgroup_limit > 0:
        # Some cgroup-v1 hosts expose a near-int64 "unlimited" sentinel.
        # The Batch task request remains the useful upper bound in that case.
        effective_limit = min(int(cgroup_limit), requested_bytes)
    peak_bytes = max(peak_candidates, default=None)
    memory_peak_fraction = (
        float(peak_bytes) / float(effective_limit)
        if peak_bytes is not None and effective_limit > 0
        else None
    )

    oom_events = sum(
        max(0, int(event_delta.get(name, 0)))
        for name in ("oom", "oom_kill", "oom_group_kill", "failcnt")
    )
    observed_exit = experiment_exit_code if experiment_exit_code is not None else exit_code
    signal_number = observed_exit - 128 if 128 <= observed_exit <= 255 else None
    failure_payloads = []
    for path in failure_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"unreadable": True}
        failure_payloads.append({"path": str(path), "payload": payload})
    failure_text = json.dumps(failure_payloads, sort_keys=True).lower()
    allocation_failure_markers = (
        "memoryerror",
        "outofmemoryerror",
        "out of memory",
        "cannot allocate memory",
        "can't allocate memory",
        "defaultcpuallocator",
    )
    reported_allocation_failure = any(
        marker in failure_text for marker in allocation_failure_markers
    )

    evidence = []
    if oom_events > 0:
        diagnosis = "confirmed_cgroup_out_of_memory"
        evidence.append("Linux cgroup memory OOM/failure counters increased during the run")
    elif reported_allocation_failure:
        diagnosis = "reported_out_of_memory_exception"
        evidence.append("failure.json contains a Python or tensor allocator memory error")
    elif observed_exit == 137:
        diagnosis = "probable_out_of_memory_or_forced_sigkill"
        evidence.append("Experiment exited 137, indicating SIGKILL")
        if memory_peak_fraction is not None and memory_peak_fraction >= 0.9:
            diagnosis = "probable_out_of_memory"
            evidence.append("Observed cgroup memory peak reached at least 90% of its limit")
    elif observed_exit in (124, 143):
        diagnosis = "terminated_or_batch_time_limit"
        evidence.append(f"Experiment exit code {observed_exit} indicates timeout or SIGTERM")
    elif exit_code == 0:
        diagnosis = "success"
    elif failure_payloads:
        diagnosis = "python_exception"
        evidence.append("The experiment wrote failure.json before exiting")
    else:
        diagnosis = "nonzero_exit_without_specific_cause"

    if memory_peak_fraction is not None and memory_peak_fraction >= 0.9:
        evidence.append("Memory headroom was below 10% at the observed peak")

    return {
        "diagnosis": diagnosis,
        "evidence": evidence,
        "exit_code": exit_code,
        "experiment_exit_code": experiment_exit_code,
        "signal_number": signal_number,
        "requested_memory_mib": requested_memory_mib,
        "effective_memory_limit_bytes": effective_limit,
        "cgroup_memory_peak_bytes": peak_bytes,
        "cgroup_memory_peak_fraction": memory_peak_fraction,
        "maximum_observed_cgroup_current_bytes": max(
            maximum_current_candidates, default=None
        ),
        "maximum_observed_process_rss_bytes": max(
            maximum_process_candidates, default=None
        ),
        "minimum_observed_system_available_bytes": min(
            minimum_available_candidates, default=None
        ),
        "cgroup_event_delta": event_delta,
        "first_cgroup_memory": first_cgroup,
        "final_cgroup_memory": last_cgroup,
        "snapshot_count": len(all_snapshots),
        "first_snapshot_utc": all_snapshots[0].get("timestamp_utc"),
        "final_snapshot": final_snapshot,
        "experiment_failure_files": failure_payloads,
    }


def finalize(args: argparse.Namespace) -> int:
    snapshots_path = Path(args.snapshots)
    failure_root = Path(args.failure_root)
    failure_files = sorted(failure_root.glob("**/failure.json")) if failure_root.exists() else []
    result = diagnose(
        snapshots=_load_snapshots(snapshots_path),
        exit_code=int(args.exit_code),
        experiment_exit_code=_optional_exit_code(args.experiment_exit_code),
        requested_memory_mib=int(args.requested_memory_mib),
        failure_files=failure_files,
    )
    result.update(
        {
            "schema_version": 1,
            "job_name": args.job_name,
            "cleanup_timestamp_utc": _utc_now(),
            "bucket_destination": args.bucket_destination,
            "resource_snapshots": str(snapshots_path),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    status = {
        key: result[key]
        for key in (
            "schema_version",
            "job_name",
            "exit_code",
            "experiment_exit_code",
            "cleanup_timestamp_utc",
            "bucket_destination",
            "diagnosis",
            "evidence",
            "cgroup_memory_peak_bytes",
            "cgroup_memory_peak_fraction",
            "cgroup_event_delta",
        )
    }
    status["diagnostics_file"] = str(output)
    status_output = Path(args.status_output)
    status_output.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--output", type=Path, required=True)
    monitor_parser.add_argument("--interval-seconds", type=float, default=15.0)
    monitor_parser.add_argument("--cloud-log-every", type=int, default=4)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--snapshots", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--status-output", required=True)
    finalize_parser.add_argument("--failure-root", required=True)
    finalize_parser.add_argument("--exit-code", type=int, required=True)
    finalize_parser.add_argument("--experiment-exit-code")
    finalize_parser.add_argument("--requested-memory-mib", type=int, required=True)
    finalize_parser.add_argument("--job-name", required=True)
    finalize_parser.add_argument("--bucket-destination", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.command == "monitor":
        return monitor(args.output, args.interval_seconds, args.cloud_log_every)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
