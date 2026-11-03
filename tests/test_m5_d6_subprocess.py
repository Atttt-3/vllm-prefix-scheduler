"""Exercise the raw D6 checker in a separate Python process."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_m5_d4_launch import authorize, put, sha
from test_m5_d5r_launch import d5r_approval
from test_m5_d6_audit import EXPORT, T0, fixture

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/m5-d6.json"


def clock_brackets(base):
    return [{"send_ns": base + i, "engine_ns": base + i, "recv_ns": base + i} for i in range(5)]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("crossing-window", None),
        ("drop-batch", "batch ID inventory"),
        ("wrong-stage", "FIFO order"),
        ("wrong-active-cap", "admission cap or stop reason differs"),
        ("wrong-domain", "native_clock_domain_or_schema_differs"),
        ("early-export", "recorder health"),
        ("overflow", "trace is incomplete"),
        ("missing-packet", "real token lacks"),
    ],
)
def test_raw_d6_checker_subprocess(tmp_path, case, reason):
    trace, client = fixture()
    # Keep the empty notification in flight across the trace cutoff.
    trace["end_ns"] = client["phases"]["measure_start_ns"] + 3_000_000_000
    end = trace["end_ns"]
    shift = end - T0 - 80
    batch = trace["batches"][-1]
    for key in (
        "future_result_start_ns",
        "future_result_end_ns",
        "consume_start_ns",
        "consume_end_ns",
    ):
        batch[key] += shift
    for rows in ("steps", "loops"):
        for key in ("start_ns", "end_ns"):
            trace[rows][-1][key] += shift
    client["clock"] = {
        "host": "fixture",
        "boot_id": "boot",
        "time_namespace": "ns",
        "clock": "CLOCK_MONOTONIC",
        "pid": 20,
    }
    engine = {"clock": {**client["clock"], "pid": 21}}
    client["phases"].update(load_end_ns=T0 - 60, warmup_start_ns=T0 - 40)
    before = clock_brackets(T0 - 59)
    after = clock_brackets(client["phases"]["drain_end_ns"] + 1)
    client["clock_samples"], client["clock_check"] = before, before[0]
    client["native_metrics_schema_version"] = 1
    client["native_metrics_clock"] = {
        "implementation": "clock_gettime(CLOCK_MONOTONIC)",
        "monotonic": True,
        "adjustable": False,
        "resolution_ns": 1,
        "post_clock_samples": after,
        "post_clock_check": after[0],
    }
    if case == "drop-batch":
        del trace["batches"][1]
    elif case == "wrong-stage":
        trace["steps"][1]["consumed_batch_id"] = 2
    elif case == "wrong-active-cap":
        trace["batches"][0]["effective_active_cap"] = 5
    elif case == "wrong-domain":
        engine["clock"]["time_namespace"] = "other"
    elif case == "early-export":
        trace["export_ns"] = end - 1
    elif case == "overflow":
        trace.update(status="incomplete", overflow=True)
    elif case == "missing-packet":
        trace["batches"][1]["output_counts"] = []
    for name, value in (("client", client), ("engine", engine), ("d6-step", trace)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_m5_d6_trace.py"),
            "--raw",
            str(tmp_path),
            "--config",
            str(ROOT / "configs/m5-performance.json"),
            "--proposal",
            str(PLAN),
            "--leaf-end-ns",
            str(EXPORT + 1),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if reason is None:
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["clock"]["verified"]
        assert report["trace"]["empty_batches"] == 1
        assert report["trace"]["linked_real_tokens"] == 2
    else:
        assert result.returncode == 1, result.stdout
        assert reason in result.stderr


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"roles": ["D6", "C1"]}, "limited to one new diagnostic allocation"),
        ({"max_attempts": 15, "max_requested_minutes": 300}, "limited to one new diagnostic"),
        ({"d6_plan_sha256": "f" * 64}, "detached from the reviewed plan"),
    ],
)
def test_d6_subprocess_authorizes_only_fourteenth_diagnostic(tmp_path, mutation, reason):
    record = d5r_approval()
    del record["d5r_plan_sha256"]
    record.update(
        user_approval_ref="local synthetic D6 authority fixture",
        roles=["D6"],
        max_attempts=14,
        max_requested_minutes=280,
        d6_plan_sha256=sha(PLAN),
    )
    path = tmp_path / "authorization.json"
    approved = authorize(put(path, record))
    assert approved.returncode == 0, approved.stderr
    assert json.loads(approved.stdout) == ["D6"]
    rejected = authorize(put(path, {**record, **mutation}))
    assert rejected.returncode == 1
    assert reason in rejected.stderr
