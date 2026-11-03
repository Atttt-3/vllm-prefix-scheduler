"""Controls for the independent two-window join and ABBA decisions."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m5_d6_observer import AsyncFuture, AsyncOutput, Clock, Core, Executor, Request, Schedule
from test_m5_d6_observer import Scheduler as FixtureScheduler

from vllm_prefix_scheduler import m5_d6_observer, m5_d9_observer
from vllm_prefix_scheduler.m5_campaign import read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m5_d9_audit import EARLY, LATE, assess_d9, audit_d9_trace, signature  # noqa: E402

T0 = 10_000_000_000
PLAN = read_json(ROOT / "configs/m5-d9.json")
CONFIG = read_json(ROOT / "configs/m5-performance.json")
LIMITS = PLAN["trace"]


def fixture(monkeypatch, request_plan=None, *, crossing=False):
    clock = Clock()
    monkeypatch.setattr(m5_d6_observer, "time", clock)
    monkeypatch.setattr(m5_d9_observer, "time", clock)
    runner = SimpleNamespace(execute_model_state=None)
    events, queue_stamp, first_schedule = [], {}, {}

    class Scheduler(FixtureScheduler):
        def add_request(self, request):
            queue_stamp[request.request_id] = clock.monotonic_ns()
            return super().add_request(request)

        def schedule(self, throttle):
            stamp = clock.monotonic_ns()
            out = super().schedule(throttle)
            for identity in out.num_scheduled_tokens:
                first_schedule.setdefault(identity, stamp)
            return out

    scheduler = Scheduler(clock, events, [])
    core = Core(scheduler, Executor(runner, events))
    clock.stamp = T0 - 2_000
    observer = m5_d9_observer._DualTrace(
        core,
        runner,
        AsyncFuture,
        AsyncOutput,
        T0,
        [[T0 + lo, T0 + hi] for lo, hi in (EARLY, LATE)],
    )
    if request_plan is None:
        request_plan = [
            {"request_id": f"request-{index:06d}", "offset_ns": 125_000_000 * index}
            for index in range(60)
        ]
    requests = []
    for index, entry in enumerate(request_plan):
        identity, offset = entry["request_id"], entry["offset_ns"]
        clock.stamp = max(T0 + offset, clock.stamp + 1)
        submit = clock.stamp
        core.scheduler.add_request(Request(identity))
        scheduler.plans.extend(Schedule({identity: 1}) for _ in range(16))
        packets = []
        while scheduler.has_requests() or core.batch_queue:
            if crossing and core.batch_queue:
                if index in (23, 47) and not scheduler.plans:
                    boundary = EARLY[1] if index == 23 else LATE[0]
                    clock.stamp = max(clock.stamp, T0 + boundary)
                elif index == 59 and len(scheduler.plans) == 1:
                    clock.stamp = max(clock.stamp, T0 + LATE[1])
            core._process_engine_step()
            outputs = core.last_step[0]
            if outputs:
                packets.extend(outputs.values())
        times = [round(packet.timestamp * 10**9) for packet in packets]
        first, last = times[0], times[-1]
        stamp = {
            "queued_ts": queue_stamp[identity] / 10**9,
            "scheduled_ts": first_schedule[identity] / 10**9,
            "first_token_ts": first / 10**9,
            "native_preemptions": 0,
        }
        requests.append(
            {
                "request_id": identity,
                "planned_ns": T0 + offset,
                "submit_ns": submit,
                "first_token_ns": first + 1,
                "last_token_ns": last + 1,
                "end_ns": last + 2,
                "cached_tokens": 0,
                "token_ids": [3] * 16,
                "status": "completed",
                "chunk_tokens": [{"at_ns": t + 1, "token_ids": [3]} for t in times],
                "native_metrics": {
                    "first": {
                        **stamp,
                        "client_received_ns": first + 1,
                        "native_generation_tokens": 1,
                        "last_token_ts": first / 10**9,
                    },
                    "last": {
                        **stamp,
                        "client_received_ns": last + 1,
                        "native_generation_tokens": 16,
                        "last_token_ts": last / 10**9,
                    },
                },
            }
        )

    drain = requests[-1]["end_ns"]
    clock.stamp = max(T0 + LATE[1], drain)
    trace = observer.export()
    assert trace["status"] == "complete", trace["errors"]
    return trace, make_client(trace, requests, drain), trace["export_ns"] + 500


def make_client(trace, requests, drain):
    phases = {
        "load_start_ns": T0 - 7_000,
        "load_end_ns": T0 - 6_000,
        "warmup_start_ns": T0 - 5_000,
        "warmup_end_ns": T0 - 3_000,
        "measure_start_ns": T0,
        "drain_end_ns": drain,
    }

    def probes(start):
        return [
            {"send_ns": start + i * 3, "engine_ns": start + i * 3 + 1, "recv_ns": start + i * 3 + 2}
            for i in range(5)
        ]

    pre, post = probes(phases["load_end_ns"]), probes(trace["export_ns"] + 1)
    phases["arrival_end_ns"] = requests[-1]["planned_ns"]
    deadline = CONFIG["gates"]["deadline_ns"]
    return {
        "tier": "endpoint",
        "deadline_ns": deadline,
        "drain_deadline_ns": phases["arrival_end_ns"] + deadline,
        "d9_capture": "step",
        "d9_arm_return_ns": T0 - 1_000,
        "phases": phases,
        "clock": {
            "host": "cpu-fixture",
            "boot_id": "fixture-boot",
            "time_namespace": "fixture-clock",
            "clock": "CLOCK_MONOTONIC",
            "pid": 10,
        },
        "clock_samples": pre,
        "clock_check": pre[0],
        "native_metrics_schema_version": 1,
        "native_metrics_clock": {
            "implementation": "clock_gettime(CLOCK_MONOTONIC)",
            "monotonic": True,
            "adjustable": False,
            "resolution_ns": 1,
            "post_clock_samples": post,
            "post_clock_check": post[0],
        },
        "requests": requests,
    }


def test_d9_independent_proof_closes_real_token_denominator(monkeypatch):
    trace, client, end = fixture(monkeypatch)
    audited = audit_d9_trace(trace, client, LIMITS, leaf_end_ns=end)
    assert audited["verified"]
    assert audited["native_real_tokens"] == 960
    assert audited["gap_packets"] > 0
    assert audited["observed_through_offset_ns"] >= LATE[1]
    assert client["phases"]["drain_end_ns"] < T0 + LATE[1]


def test_d9_crossing_tokens_keep_absolute_order_and_missing_launch_evidence(monkeypatch):
    trace, client, end = fixture(monkeypatch, crossing=True)
    audited = audit_d9_trace(trace, client, LIMITS, leaf_end_ns=end)
    assert audited["native_real_tokens"] == 960
    assert audited["tail_packets"] == 1
    early, late = audited["windows"]["early"], audited["windows"]["late"]
    early_last = early["links"]["request-000023"][-1]
    assert early_last["packet_ns"] >= T0 + EARLY[1] and early_last["ordinal"] == 16
    assert late["carry_batch_count"] == 1
    assert "runner" not in late["carry"]
    assert late["links"]["request-000047"][-1]["batch_id"] == 0
    assert late["links"]["request-000047"][-1]["ordinal"] == 16


def audit_child(tmp_path, trace, client, end):
    paths = [tmp_path / name for name in ("trace.json", "client.json")]
    for path, obj in zip(paths, (trace, client), strict=True):
        path.write_text(json.dumps(obj))
    return subprocess.run(
        [
            sys.executable,
            "-c",
            """import json
from pathlib import Path
import sys
from m5_d3_audit import native_clock
from m5_d9_audit import audit_d9_trace
from vllm_prefix_scheduler.m5_campaign import read_json, require
trace, client = (read_json(Path(path)) for path in sys.argv[1:3])
limits = read_json(Path(sys.argv[3]))["trace"]
config = read_json(Path(sys.argv[4]))
end = int(sys.argv[5])
engine = {"clock": {**client["clock"], "pid": 11, "time_namespace": sys.argv[6]}}
try:
    clock = native_clock(client, engine, config["gates"], end)
    require(clock["verified"], "D9 test clock is unverified")
    result = audit_d9_trace(trace, client, limits, leaf_end_ns=end)
    require(result["verified"], "D9 test trace is incomplete")
except (ValueError, KeyError) as error:
    sys.exit(str(error))
print(json.dumps({"verified": result["verified"], "native_real_tokens": result["native_real_tokens"]}))
""",
            *(str(path) for path in paths),
            str(ROOT / "configs/m5-d9.json"),
            str(ROOT / "configs/m5-performance.json"),
            str(end),
            "fixture-clock",
        ],
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda trace, client: trace["windows"].reverse(), "idle arm"),
        (lambda trace, client: trace["windows"][1]["preroll_tokens"].pop(), "ordinal"),
        (lambda trace, client: trace["gap_tokens"].pop(), "ordinal inventory"),
        (
            lambda trace, client: trace["drained"].update(native_fifo=1),
            "export observation horizon",
        ),
        (lambda trace, client: trace.update(export_ns=T0 + LATE[1] - 1), "observation horizon"),
        (lambda trace, client: trace.update(overflow=True), "observation horizon"),
        (
            lambda trace, client: client["clock"].update(time_namespace="tampered"),
            "clock is unverified",
        ),
        (
            lambda trace, client: trace["windows"][1]["steps"][1].update(consumed_batch_id=2),
            "FIFO order",
        ),
    ],
)
def test_d9_real_child_accepts_raw_and_rejects_semantic_controls(
    monkeypatch, tmp_path, mutate, reason
):
    trace, client, end = fixture(monkeypatch)
    good = audit_child(tmp_path, trace, client, end)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout) == {"verified": True, "native_real_tokens": 960}
    mutate(trace, client)
    bad = audit_child(tmp_path, trace, client, end)
    assert bad.returncode == 1
    assert reason in bad.stderr


def test_quiet_horizon_does_not_move_terminal_metrics_or_add_samples():
    client = {
        "phases": {"measure_start_ns": T0, "drain_end_ns": T0 + 8_400_000_000},
        "requests": [
            {
                "request_id": identity,
                "chunk_tokens": [
                    {"at_ns": T0 + 7_800_000_000 + index * 35_000_000, "token_ids": [0]}
                    for index in range(18)
                ],
            }
            for identity in ("one", "two")
        ],
    }
    assert not signature(client, LATE)["bins"][-1]["eligible"]
    result = signature(client, LATE, observation_end_ns=T0 + LATE[1])
    assert result["bins"][-1]["complete"]
    assert result["bins"][-1]["eligible"]
    assert result["candidate"] == "detected"
    assert result["trigger_starts_offset_ns"] == [7_500_000_000]
    assert client["phases"]["drain_end_ns"] == T0 + 8_400_000_000


def metrics(ttft, *, rps=8.0, tps=128.0):
    return {
        "strata": {"all": {"ttft_ns": {"p95": ttft}}},
        "throughput": {"last_terminal_requests_per_s": rps, "last_terminal_tokens_per_s": tps},
    }


def test_d9_pair_failure_remains_diagnostic_and_preserves_d8_direction():
    runs = {
        leaf: {
            "endpoint_evidence_complete": True,
            "metrics": metrics(100_000_000, tps=120.0 if leaf == "E3" else 128.0),
        }
        for leaf in ("E1", "E2", "E3", "E4")
    }
    traces = {leaf: {"verified": True} for leaf in ("E2", "E3")}
    result = assess_d9(runs, {}, traces, CONFIG["gates"], heat_verified=True)
    assert result["capture_equivalence"] == "outside_tolerance"
    assert not result["pairs"]["E3/E4"]["within_tolerance"]
    assert not result["pairs"]["E4/E3"]["within_tolerance"]
    assert result["backlog_formation"]["status"] == "unresolved"
    assert result["late_amplification"]["status"] == "unresolved"
    incomplete = assess_d9(runs, {}, {}, CONFIG["gates"], heat_verified=True)
    assert incomplete["capture_equivalence"] == "unverified"
