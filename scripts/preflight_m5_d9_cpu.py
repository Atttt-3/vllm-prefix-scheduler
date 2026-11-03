"""Measure D9 CPU observation overhead with the pinned native call path and synthetic tokens."""

import argparse
import ast
import importlib.metadata
import json
import queue
import sys
import time
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

from vllm_prefix_scheduler import m5_d6_observer, m5_d9_observer
from vllm_prefix_scheduler.m5_campaign import digest, read_json, require
from vllm_prefix_scheduler.m5_d3 import snapshot_native_metrics

ROOT = Path(__file__).resolve().parents[1]
CORE = "vllm/v1/engine/core.py"
EXECUTOR = "vllm/v1/executor/uniproc_executor.py"
NS = 1_000_000_000


def native_method(name, path, **namespace):
    tree = ast.parse(path.read_bytes())
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef | ast.ClassDef) and item.name == name
    )
    if node.decorator_list:
        raise ValueError(f"unexpected decorators on pinned native method {name}")
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def native_paths(source_root):
    pins = {
        **read_json(ROOT / "configs/native-m2b.json")["source_sha256"],
        **read_json(ROOT / "configs/m5-performance.json")["extra_upstream_sha256"],
    }
    if source_root is None:
        dist = importlib.metadata.distribution("vllm")
        require(dist.version == "0.32.0", "D9 CPU preflight vLLM version differs")
        paths = {name: Path(dist.locate_file(name)) for name in pins}
    else:
        paths = {name: source_root / name for name in pins if name in (CORE, EXECUTOR)}
    for name, path in paths.items():
        require(digest(path) == pins[name], f"D9 CPU preflight source differs: {name}")
    return paths, {name: pins[name] for name in paths}


def build_fixture(paths):
    fixture_path = ROOT / "tests/test_m5_d6_observer.py"
    names = ("Clock", "Request", "Schedule", "AsyncOutput", "Scheduler", "Executor", "Core")
    fixture = SimpleNamespace(
        **{
            name: native_method(
                name,
                fixture_path,
                deque=deque,
                Future=Future,
                nullcontext=nullcontext,
                SimpleNamespace=SimpleNamespace,
            )
            for name in names
        }
    )

    clock = fixture.Clock()
    step = native_method(
        "step_with_batch_queue", paths[CORE], cast=cast, Future=Future, ModelRunnerOutput=object
    )
    loop = native_method("_process_engine_step", paths[CORE], time=clock)
    poll = native_method("_process_input_queue", paths[CORE])
    native_future = native_method("AsyncOutputFuture", paths[EXECUTOR], Future=Future)

    class Executor(fixture.Executor):
        def sample_tokens(self, grammar, *, non_block):
            assert non_block
            self.events.append("sample")
            ids = self.runner.execute_model_state.input_batch.req_ids
            self.runner.execute_model_state = None
            value = SimpleNamespace(req_ids=ids, sampled_token_ids=[[3] for _ in ids])
            return native_future(fixture.AsyncOutput(self.events, value), single_value=True)

    return fixture, clock, step, loop, poll, native_future, Executor


def measure(native, windows, directory, traced, width, codec):
    fixture, clock, step, loop, poll, native_future, executor_type = native
    t0 = 10 * NS
    absolute = [[t0 + start, t0 + end] for start, end in windows]
    before = time.perf_counter_ns()
    operations, packets = [], []
    runner = SimpleNamespace(execute_model_state=None)
    identities = [f"request-{index:06d}" for index in range(60)]
    cohorts = [identities[index : index + width] for index in range(0, len(identities), width)]
    plans = [fixture.Schedule(dict.fromkeys(cohort, 1)) for cohort in cohorts for _ in range(16)]
    scheduler = fixture.Scheduler(clock, operations, plans)
    core = fixture.Core(scheduler, executor_type(runner, operations))
    core.step_fn = step.__get__(core)
    core._process_engine_step = loop.__get__(core)
    core._process_input_queue = poll.__get__(core)
    core.input_queue = queue.Queue()
    core.has_work = lambda: True
    core.output_queue = SimpleNamespace(put_nowait=packets.append)
    core.post_step = lambda executed: None

    clock.stamp = t0 - 250_000_000
    trace = (
        m5_d9_observer._DualTrace(core, runner, native_future, fixture.AsyncOutput, t0, absolute)
        if traced
        else None
    )
    clock.stamp = t0
    for identity in identities:
        scheduler.add_request(fixture.Request(identity))
    phases = (t0, absolute[0][1], absolute[1][0], absolute[1][1])
    slices = (len(plans) * 2 // 5, len(plans) // 4)
    counts = (*slices, len(plans) - sum(slices), 1)
    for at, count in zip(phases, counts, strict=True):
        clock.stamp = at
        for _ in range(count):
            core._process_input_queue()
            core._process_engine_step()
    require(not core.batch_queue and not scheduler.has_requests(), "native fixture did not drain")
    total_before_export = time.perf_counter_ns()
    output_size = 0
    if trace is not None:
        result = trace.export()
        require(result["status"] == "complete", f"D9 CPU trace incomplete: {result['errors']}")
        if codec is not None:
            result = codec.decode(codec.encode(result))
        path = directory / "cpu-step.json"
        m5_d9_observer.write_d9_trace(path, result)
        output_size = path.stat().st_size
        path.unlink()
        stats = SimpleNamespace(
            queued_ts=10.0,
            scheduled_ts=10.1,
            first_token_ts=10.2,
            last_token_ts=10.3,
            num_generation_tokens=16,
            num_preemptions=0,
        )
        require(
            all(snapshot_native_metrics(stats, SimpleNamespace, t0) for _ in range(60 * 16)),
            "native endpoint snapshot failed",
        )
    after = time.perf_counter_ns()
    contents = [
        (output.request_id, output.new_token_ids)
        for _, packet in packets
        for output in packet.outputs
    ]
    require(
        len(contents) == 960
        and all(
            operations.count(key) == len(plans)
            for key in ("schedule", "execute", "sample", "materialize", "consume")
        ),
        "pinned call or token denominator changed",
    )
    return {
        "native_path_ns": total_before_export - before,
        "export_and_endpoint_copies_ns": after - total_before_export,
        "trace_bytes": output_size,
        "native_batches": len(plans),
    }, (operations, contents)


def run(args):
    require(not sys.flags.optimize and sys.version_info[:3] == (3, 12, 16), "D9 CPU Python differs")
    require(args.repeats > 0, "D9 CPU preflight requires at least one ABBA")
    plan = read_json(args.proposal)
    require(digest(args.input) == plan["target"]["input_sha256"], "D9 CPU input digest differs")
    arrival_table = read_json(args.input)
    last_arrival = max(row["offset_ns"] for row in arrival_table["requests"])
    windows = plan["trace"]["windows_ns"]
    margin_ns = plan["budget"]["trace_export_margin_seconds"] * NS
    # Charge the entire common launch lead and the worst possible quiet tail
    # for both B leaves. A valid drain cannot precede the final planned arrival.
    fixed_bound_ns = 4 * plan["trace"]["arm_lead_ns"] + 2 * max(0, windows[-1][1] - last_arrival)
    paths, pins = native_paths(args.source_root)
    native = build_fixture(paths)
    codec = None
    if args.source_root is None:
        from msgspec import msgpack

        clock = time.get_clock_info("monotonic")
        require(
            sys.platform == "linux"
            and clock.implementation == "clock_gettime(CLOCK_MONOTONIC)"
            and clock.monotonic
            and not clock.adjustable,
            "D9 CPU monotonic clock differs",
        )
        codec = msgpack
    m5_d6_observer.time = m5_d9_observer.time = native[1]
    samples = []
    with TemporaryDirectory(prefix="d9-cpu-", dir=args.scratch) as temporary:
        directory = Path(temporary)
        for width in (4, 1):
            for _ in range(args.repeats):
                rows = [
                    measure(native, windows, directory, traced, width, codec)
                    for traced in (False, True, True, False)
                ]
                require(all(value == rows[0][1] for _, value in rows), "native ABBA results differ")
                legs = [row for row, _ in rows]
                # Use the slower observed B and faster observed A for both B leaves.
                incremental = 2 * max(
                    0,
                    max(
                        row["native_path_ns"] + row["export_and_endpoint_copies_ns"]
                        for row in legs[1:3]
                    )
                    - min(legs[0]["native_path_ns"], legs[3]["native_path_ns"]),
                )
                samples.append({"width": width, "legs": legs, "incremental_ns": incremental})
    observed_ns = max(row["incremental_ns"] for row in samples)
    report = {
        "scope": "pinned EngineCore/UniProc CPU methods, synthetic scheduler/runner; no GPU or model",
        "python": list(sys.version_info[:3]),
        "source_sha256": pins,
        "msgpack_round_trip": codec is not None,
        "plan_sha256": digest(args.proposal),
        "input_sha256": digest(args.input),
        "last_arrival_ns": last_arrival,
        "fixed_lead_and_quiet_bound_ns": fixed_bound_ns,
        "maximum_two_b_incremental_ns": observed_ns,
        "total_added_upper_envelope_ns": fixed_bound_ns + observed_ns,
        "margin_ns": margin_ns,
        "samples": samples,
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    require(report["total_added_upper_envelope_ns"] <= margin_ns, "D9 CPU overhead exceeds budget")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--source-root", type=Path, help="local pinned checkout, for dry-run only")
    run(parser.parse_args())
