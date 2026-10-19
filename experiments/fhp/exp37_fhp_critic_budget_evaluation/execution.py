"""Bounded, durable match execution: retain other in-flight successes on error."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
from time import perf_counter

from experiments.fhp.retrospective_exp6_exp7_evaluation import run as shared

protocol = shared.protocol


def timed_worker(task):
    started = perf_counter()
    result = protocol._evaluation_worker(task)
    result["evaluation_task_seconds"] = perf_counter()-started
    return result


def run_tasks(tasks, workers, cache_dir, fingerprint):
    cache_dir.mkdir(parents=True, exist_ok=True)
    results, pending = [], []
    for task in tasks:
        path = cache_dir / f"{task['task_id']}.json"
        if not path.exists():
            pending.append(task)
            continue
        saved = json.loads(path.read_text())
        row = saved["result"]
        if (saved["fingerprint"] != fingerprint or saved["task"] != shared._portable_task(task)
                or saved["result_sha256"] != shared._digest(row)
                or row["task_id"] != task["task_id"] or row["num_deal_pairs"] != task["num_deals"]):
            raise ValueError(f"Incompatible or corrupt cached match: {path}")
        results.append(dict(row, **task))
    print(f"Reusing {len(results)}/{len(tasks)} shards; {len(pending)} pending on {workers} processes", flush=True)

    def collect(task, row):
        if row["num_deal_pairs"] != task["num_deals"] or row["task_id"] != task["task_id"]:
            raise ValueError("Incomplete match result")
        shared._write_json(cache_dir / f"{task['task_id']}.json", dict(fingerprint=fingerprint,
            task=shared._portable_task(task), result=row, result_sha256=shared._digest(row)))
        results.append(row)
        if len(results) % max(1, len(tasks)//40) == 0 or len(results) == len(tasks):
            print(f"Completed {len(results)}/{len(tasks)} shards", flush=True)

    if workers == 1:
        for task in pending:
            collect(task, timed_worker(task))
    elif pending:
        iterator = iter(pending)
        failures = []
        # Only workers tasks are in flight, rather than queueing hours of work.
        # After an error, stop dispatching but save every other in-flight result.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {}

            def fill():
                while len(futures) < workers:
                    task = next(iterator, None)
                    if task is None:
                        return
                    futures[pool.submit(timed_worker, task)] = task

            fill()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    try:
                        collect(task, future.result())
                    except Exception as error:
                        failures.append(dict(task_id=task["task_id"], error=f"{type(error).__name__}: {error}"))
                        print(f"Shard failed: {task['task_id']}; saving remaining in-flight results", flush=True)
                if not failures:
                    fill()
        if failures:
            shared._write_json(cache_dir.parent / "last_task_failure.json", dict(fingerprint=fingerprint, failures=failures))
            raise RuntimeError(f"Evaluation task failed: {failures[0]['task_id']}: {failures[0]['error']}")
    return sorted(results, key=lambda r: r["task_id"])
