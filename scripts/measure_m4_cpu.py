"""Native-object, observer-free local CPU costs; never measures model latency."""

import argparse
import functools
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import deque
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/native"))


class InjectedPrependError(RuntimeError):
    pass


class Meter:
    """Temporarily wrap actual methods, count visits, and restore every binding."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.originals = []
        self.reset()
        self.inject = False

    def reset(self):
        self.ns = dict.fromkeys(("window", "score", "sort", "commit"), 0)
        self.calls = dict.fromkeys(
            ("window", "score", "sort", "pop", "prepend", "guard_clock", "native_schedule"), 0
        )
        self.visits = dict.fromkeys(("candidate", "queue", "hash_lookup", "pop", "prepend"), 0)

    def install(self, obj, name, fn):
        own = name in getattr(obj, "__dict__", {})
        original = getattr(obj, name)
        self.originals.append((obj, name, original, own))
        setattr(obj, name, fn(original))

    def timer(self, phase):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                self.calls[phase] += 1
                start = time.perf_counter_ns()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.ns[phase] += time.perf_counter_ns() - start

            return wrapped

        return factory

    def __enter__(self):
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue

        s = self.scheduler
        for name, phase in (
            ("_candidate_window", "window"),
            ("_score_request", "score"),
            ("_sort_window", "sort"),
        ):
            if hasattr(s, name):
                self.install(s, name, self.timer(phase))

        def count(name):
            def factory(original):
                def wrapped(*args, **kwargs):
                    self.visits[name] += 1
                    return original(*args, **kwargs)

                return wrapped

            return factory

        if hasattr(s, "_fresh_request"):
            self.install(s, "_fresh_request", count("candidate"))
        self.install(s.kv_cache_manager.block_pool, "get_cached_block", count("hash_lookup"))
        guard = getattr(s, "_wait_guard", None)
        if guard is not None:

            def clock_factory(original):
                def clock():
                    self.calls["guard_clock"] += 1
                    return original()

                return clock

            self.install(guard, "_clock", clock_factory)

        def iterator_factory(original):
            def iterator(queue):
                for request in original(queue):
                    if queue is s.waiting:
                        self.visits["queue"] += 1
                    yield request

            return iterator

        self.install(FCFSRequestQueue, "__iter__", iterator_factory)

        def pop_factory(original):
            def pop(queue):
                if queue is not s.waiting:
                    return original(queue)
                self.calls["pop"] += 1
                self.visits["pop"] += 1
                start = time.perf_counter_ns()
                try:
                    return original(queue)
                finally:
                    self.ns["commit"] += time.perf_counter_ns() - start

            return pop

        def prepend_factory(original):
            def prepend(queue, requests):
                if queue is not s.waiting:
                    return original(queue, requests)
                self.calls["prepend"] += 1

                def counted():
                    for request in requests:
                        self.visits["prepend"] += 1
                        yield request

                start = time.perf_counter_ns()
                try:
                    if self.inject:
                        # Fail after one actual deque prepend; production restores it.
                        iterator = counted()
                        deque.appendleft(queue, next(iterator))
                        raise InjectedPrependError("M4 named partial prepend branch")
                    return original(queue, counted())
                finally:
                    self.ns["commit"] += time.perf_counter_ns() - start

            return prepend

        self.install(FCFSRequestQueue, "pop_request", pop_factory)
        self.install(FCFSRequestQueue, "prepend_requests", prepend_factory)
        return self

    def __exit__(self, *_):
        for obj, name, original, own in reversed(self.originals):
            if own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def empty_cost(samples):
    def bare():
        return None

    def wrapped():
        start = time.perf_counter_ns()
        bare()
        return time.perf_counter_ns() - start

    raw, wrapper, clock = [], [], []
    for _ in range(samples):
        start = time.perf_counter_ns()
        bare()
        raw.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        wrapped()
        wrapper.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        clock.append(time.perf_counter_ns() - start)
    return {"bare_call_ns": raw, "timed_wrapper_call_ns": wrapper, "clock_pair_ns": clock}


def measure(config, model):
    from m2a_harness import create_scheduler, request
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import RequestStatus

    plan = config["overhead"]
    native = json.loads((ROOT / "configs/native-m2b.json").read_text())
    native["max_wait_ms"] = plan["guard_max_wait_ms"]
    native["seed"] = 42
    rows = []
    empty = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    for mode in plan["modes"]:
        for window in plan["windows"]:
            for workload in plan["workloads"]:
                for outside in plan["outside_lengths"]:
                    s = create_scheduler(native, mode, model, capacity=64, window=window)
                    if hasattr(s, "_wait_guard") != (mode == "guarded"):
                        raise RuntimeError("Unexpected guard table presence")
                    # Native setup operations populate real native cache metadata.
                    # This is deliberately outside every timed interval.
                    hot = list(range(1000, 1064))
                    warm = request(native, "cpu-warm", hot)
                    blocks = s.kv_cache_manager.allocate_slots(warm, len(hot))
                    if blocks is None:
                        raise RuntimeError("Native warm-cache setup failed")
                    s.kv_cache_manager.cache_blocks(warm, len(hot))
                    s.kv_cache_manager.free(warm)
                    del warm
                    for index in range(window + outside):
                        is_hot = workload == "tied_hot" or (workload == "mixed" and index % 2 == 1)
                        ids = hot if is_hot else list(range(2000 + index * 64, 2064 + index * 64))
                        s.add_request(request(native, f"cpu-{index}", ids))
                    original = list(islice(s.waiting, window))
                    s.set_pause_state(PauseState.PAUSED_ALL)

                    def reset_head(s=s, original=tuple(original)):
                        for _ in original:
                            deque.popleft(s.waiting)
                        deque.extendleft(s.waiting, reversed(original))

                    def operation(s=s, mode=mode):
                        if mode in {"native", "off"}:
                            return s.schedule()
                        s._reorder_waiting()
                        return None

                    # Same actual operation with no wrappers; output consumption excluded.
                    unwrapped = []
                    for _ in range(plan["warmup_samples"] + plan["samples_per_repeat"]):
                        reset_head()
                        start = time.perf_counter_ns()
                        output = operation()
                        unwrapped.append(time.perf_counter_ns() - start)
                        if output is not None:
                            s.update_from_output(output, empty)
                    unwrapped = unwrapped[plan["warmup_samples"] :]
                    with Meter(s) as meter:
                        for repeat in range(plan["repeats"]):
                            for sample in range(plan["samples_per_repeat"]):
                                reset_head()
                                meter.reset()
                                start = time.perf_counter_ns()
                                output = operation()
                                total = time.perf_counter_ns() - start
                                if mode in {"native", "off"}:
                                    meter.calls["native_schedule"] = 1
                                phases = {
                                    "total": total,
                                    "window": meter.ns["window"],
                                    "score": meter.ns["score"],
                                    "sort_exclusive": meter.ns["sort"] - meter.ns["score"],
                                    "commit": meter.ns["commit"],
                                    "restore_residual": 0,
                                }
                                rows.append(
                                    {
                                        "event": "m4_cpu_sample",
                                        "mode": mode,
                                        "workload": workload,
                                        "window": window,
                                        "outside_length": outside,
                                        "repeat": repeat,
                                        "sample": sample,
                                        "variant": "normal",
                                        "duration_ns": phases,
                                        "calls": meter.calls.copy(),
                                        "visits": meter.visits.copy(),
                                        "observer_io": False,
                                        "snapshot_io": False,
                                        "guard_present": hasattr(s, "_wait_guard"),
                                        "guard_max_wait_ms": native["max_wait_ms"],
                                    }
                                )
                                if output is not None:
                                    s.update_from_output(output, empty)
                            if mode in {"prefix", "guarded"} and workload == "mixed":
                                reset_head()
                                meter.reset()
                                meter.inject = True
                                failed = False
                                start = time.perf_counter_ns()
                                try:
                                    s._reorder_waiting()
                                except InjectedPrependError:
                                    failed = True
                                finally:
                                    total = time.perf_counter_ns() - start
                                    meter.inject = False
                                raw_counts = meter.visits.copy()
                                if (
                                    not failed
                                    or list(islice(deque.__iter__(s.waiting), window)) != original
                                ):
                                    raise RuntimeError("Actual production prepend recovery failed")
                                rows.append(
                                    {
                                        "event": "m4_cpu_sample",
                                        "mode": mode,
                                        "workload": workload,
                                        "window": window,
                                        "outside_length": outside,
                                        "repeat": repeat,
                                        "sample": 0,
                                        "variant": "prepend_restore",
                                        "duration_ns": {
                                            "total": total,
                                            "window": meter.ns["window"],
                                            "score": meter.ns["score"],
                                            "sort_exclusive": meter.ns["sort"] - meter.ns["score"],
                                            "commit": meter.ns["commit"],
                                            "restore_residual": total
                                            - meter.ns["window"]
                                            - meter.ns["sort"]
                                            - meter.ns["commit"],
                                        },
                                        "calls": meter.calls.copy(),
                                        "visits": raw_counts,
                                        "observer_io": False,
                                        "snapshot_io": False,
                                        "guard_present": hasattr(s, "_wait_guard"),
                                        "guard_max_wait_ms": native["max_wait_ms"],
                                        "restored": True,
                                    }
                                )
                                # A separate counter-only replay observes the native deque
                                # recovery primitives; profile overhead is never a timed sample.
                                reset_head()
                                meter.inject = True
                                restore_counts = {
                                    "popleft_calls": 0,
                                    "popleft_entries": 0,
                                    "extendleft_calls": 0,
                                    "extendleft_entries": 0,
                                }
                                pending_call = []
                                code = s._reorder_waiting.__func__.__code__

                                def count_restore(
                                    frame,
                                    event,
                                    arg,
                                    code=code,
                                    queue=s.waiting,
                                    counts=restore_counts,
                                    pending=pending_call,
                                ):
                                    if (
                                        frame.f_code is not code
                                        or getattr(arg, "__self__", None) is not queue
                                    ):
                                        return
                                    name = getattr(arg, "__name__", "")
                                    if name not in {"popleft", "extendleft"}:
                                        return
                                    if event == "c_call":
                                        pending.append((name, deque.__len__(queue)))
                                        counts[name + "_calls"] += 1
                                    elif event == "c_return":
                                        actual, before_size = pending.pop()
                                        counts[actual + "_entries"] += abs(
                                            deque.__len__(queue) - before_size
                                        )

                                previous_profile = sys.getprofile()
                                try:
                                    sys.setprofile(count_restore)
                                    s._reorder_waiting()
                                except InjectedPrependError:
                                    pass
                                finally:
                                    sys.setprofile(previous_profile)
                                    meter.inject = False
                                if restore_counts != {
                                    "popleft_calls": 1,
                                    "popleft_entries": 1,
                                    "extendleft_calls": 1,
                                    "extendleft_entries": window,
                                }:
                                    raise RuntimeError(
                                        "Native recovery primitive access count differs"
                                    )
                                rows.append(
                                    {
                                        "event": "m4_cpu_restore_counter",
                                        "mode": mode,
                                        "window": window,
                                        "outside_length": outside,
                                        "workload": workload,
                                        "repeat": repeat,
                                        "counter_only": True,
                                        "timed_sample": False,
                                        "counts": restore_counts,
                                    }
                                )
                    rows.append(
                        {
                            "event": "m4_cpu_unwrapped",
                            "mode": mode,
                            "window": window,
                            "workload": workload,
                            "outside_length": outside,
                            "samples_ns": unwrapped,
                        }
                    )
                    # Native lifecycle cleanup, outside measurements.
                    s.finish_requests(None, RequestStatus.FINISHED_ABORTED)
                    s.set_pause_state(PauseState.UNPAUSED)
                    for _ in range(8):
                        if not s.has_requests():
                            break
                        s.update_from_output(s.schedule(), empty)
                    if (
                        s.requests
                        or s.has_requests()
                        or getattr(s, "_wait_guard", None)
                        and s._wait_guard.records
                    ):
                        raise RuntimeError("CPU fixture failed to drain actual native requests")
                    s.shutdown()
                    del s, original, reset_head, operation
                    gc.collect()
    return rows


def main():
    if not __debug__ or sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Unoptimized fixed Linux native runtime in an allocation required")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    from m4_runtime import native_identity

    runtime = native_identity(config)
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    if os.environ.get("M3_TRACE_DIR"):
        raise RuntimeError("Disable diagnostic JSONL observer during CPU measurements")
    rows = [
        {
            "event": "m4_cpu_environment",
            "machine": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "clock": {"name": "perf_counter_ns", **vars(time.get_clock_info("perf_counter"))},
            "unit": "one waiting-window transaction; native/off one PAUSED_ALL native schedule",
            "project_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "observer_io": False,
            "snapshot_io": False,
            "empty_wrapper_ns": empty_cost(1000),
            "limitations": [
                "wrapper overhead is included, not subtracted",
                "sort_exclusive subtracts nested score timer only",
                "restore_residual includes exception and transaction bookkeeping",
                "native/off paused schedule is not model latency",
                "native KV metadata only; no model or GPU tensor execution",
                "guard threshold one hour for unexpired local algorithm comparison",
            ],
        }
    ]
    rows.extend(measure(config, args.model_config))
    with (args.output / "trace.jsonl").open("x") as stream:
        for seq, row in enumerate(rows, 1):
            stream.write(
                json.dumps(
                    {
                        "schema": 1,
                        "task_id": config["task_id"],
                        "layer": "cpu",
                        "job_id": runtime["job_id"],
                        "project_commit": runtime["project_commit"],
                        "seq": seq,
                        **row,
                    }
                )
                + "\n"
            )
    groups = {}
    for row in rows:
        if row["event"] == "m4_cpu_sample":
            key = ":".join(
                str(row[k]) for k in ("mode", "workload", "window", "outside_length", "variant")
            )
            groups.setdefault(key, []).append(row["duration_ns"]["total"])
    summary = {
        key: {
            "n": len(values),
            "p50_ns": statistics.median(values),
            "p95_ns": sorted(values)[int((len(values) - 1) * 0.95)],
        }
        for key, values in groups.items()
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
