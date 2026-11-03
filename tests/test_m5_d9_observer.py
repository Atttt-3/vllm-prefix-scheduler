"""Check D9's two native windows, one arm, and unbroken packet ordinals."""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m5_d6_observer import AsyncFuture, AsyncOutput, Clock, Core, Executor, Request, Schedule
from test_m5_d6_observer import Scheduler as FixtureScheduler

from vllm_prefix_scheduler import m5_d6_observer, m5_d9_observer

T0 = 10_000_000_000
WINDOWS = [(T0 + start, T0 + end) for start, end in m5_d9_observer.D9_WINDOWS_NS]
EARLY, LATE = WINDOWS


def test_d9_post_drain_export_keeps_compact_bytes_and_create_once(tmp_path):
    path = tmp_path / "d9-step.json"
    report = {"windows": [{"rows": [{"request_id": "r", "ordinal": 1}]}]}
    m5_d9_observer.write_d9_trace(path, report)
    assert path.read_text() == json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    with pytest.raises(FileExistsError):
        m5_d9_observer.write_d9_trace(path, report)


def harness(monkeypatch, observed=True):
    clock = Clock()
    monkeypatch.setattr(m5_d6_observer, "time", clock)
    monkeypatch.setattr(m5_d9_observer, "time", clock)
    events = []
    runner = SimpleNamespace(execute_model_state=None)
    plans = [{"r": 1}] * 6 + [{}, {"r": 1}]
    scheduler = FixtureScheduler(clock, events, [Schedule(tokens) for tokens in plans])
    core = Core(scheduler, Executor(runner, events))
    trace = (
        m5_d9_observer._DualTrace(core, runner, AsyncFuture, AsyncOutput, T0, WINDOWS)
        if observed
        else None
    )
    return core, trace, clock, events


def drive(core, clock):
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    outcomes = []
    for now in (
        T0 + 100,
        T0 + 200,
        EARLY[1],
        EARLY[1] + 100,
        LATE[0] - 100,
        LATE[0],
        LATE[0] + 100,
        LATE[1],
        LATE[1] + 100,
    ):
        clock.stamp = now
        core._process_input_queue()
        executed = core._process_engine_step()
        packets, inner_executed = core.last_step
        outcomes.append(
            (
                executed,
                inner_executed,
                [
                    (item.request_id, item.new_token_ids)
                    for packet in (packets or {}).values()
                    for item in packet.outputs
                ],
            )
        )
    return outcomes


def test_d9_single_arm_preserves_native_interleaving_and_token_ordinals(monkeypatch):
    core, _, clock, native_calls = harness(monkeypatch, observed=False)
    native_outcomes = drive(core, clock)
    core, trace, clock, observed_calls = harness(monkeypatch)
    assert drive(core, clock) == native_outcomes
    assert observed_calls == native_calls
    assert len(trace.patches) == 8
    report = trace.export()
    assert report["status"] == "complete", report["errors"]
    assert report["drained"] == dict(pending=0, native_fifo=0, sampling=0, has_requests=False)
    early, late = report["windows"]
    assert early["carry_in"] == []
    assert [row["consumed_batch_id"] for row in early["steps"]] == [None, 1, 2]
    assert early["batches"][-1]["consume_end_ns"] >= EARLY[1]
    assert [row["ordinal"] for row in report["gap_tokens"]] == [3, 4]
    assert [row["ordinal"] for row in late["preroll_tokens"]] == [1, 2, 3, 4]
    [carry] = late["carry_in"]
    assert carry["batch_id"] == 0 and "runner" not in carry
    assert carry["output_counts"][0]["ordinal"] == 5
    assert late["boundary"]["state"]["pending_batch_ids"] == [0]
    assert [row["scheduled_batch_id"] for row in late["steps"]] == [1, 2, None]
    assert [row["consumed_batch_id"] for row in late["steps"]] == [0, 1, 2]
    assert late["batches"][0]["output_counts"][0]["ordinal"] == 6
    assert late["batches"][-1]["scheduled"] == []
    assert late["batches"][-1]["consume_end_ns"] >= LATE[1]
    assert [row["ordinal"] for row in report["tail_tokens"]] == [7]
    [crossing] = [row for row in early["steps"] if row["start_ns"] >= EARLY[1]]
    assert crossing["scheduled_batch_id"] is None and crossing["consumed_batch_id"] == 2
    assert core.step_fn == core.step


def test_d9_quiet_horizon_does_not_run_an_extra_step(monkeypatch):
    core, trace, clock, events = harness(monkeypatch)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    for stamp in (T0 + 100, T0 + 200, EARLY[1], LATE[0], LATE[0] + 100, LATE[0] + 200):
        clock.stamp = stamp
        core._process_engine_step()
    while core.scheduler.has_requests() or core.batch_queue:
        core._process_engine_step()
    native_calls = events.copy()
    clock.stamp = LATE[1]
    report = trace.export()
    assert report["status"] == "complete", report["errors"]
    assert events == native_calls


def test_d9_rejects_early_export_and_incomplete_fifo(monkeypatch):
    core, trace, clock, _ = harness(monkeypatch)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    core._process_engine_step()
    report = trace.export()
    assert report["status"] == "incomplete"
    assert report["drained"]["native_fifo"] == report["drained"]["pending"] == 1
    assert any("drained post-window boundary" in row["error"] for row in report["errors"])


def test_d9_fails_closed_if_the_early_fifo_cannot_close_before_late(monkeypatch):
    core, trace, clock, events = harness(monkeypatch)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    core._process_engine_step()
    clock.stamp = LATE[0]
    while core.scheduler.has_requests() or core.batch_queue:
        core._process_engine_step()
    clock.stamp = LATE[1]
    report = trace.export()
    assert report["status"] == "incomplete"
    assert any("early FIFO did not close" in row["error"] for row in report["errors"])
    assert events.count("consume") == 8 and events.count("materialize") == 7


def test_d9_applies_single_event_budget_across_both_windows(monkeypatch):
    monkeypatch.setattr(m5_d9_observer, "D6_EVENT_LIMIT", 13)
    core, trace, clock, events = harness(monkeypatch)
    drive(core, clock)
    report = trace.export()
    assert report["status"] == "incomplete" and report["overflow"]
    assert events.count("consume") == 8
    assert events.count("fifo-result") == events.count("materialize") == 7


def test_d9_requires_both_frozen_windows_and_pre_t0_arm(monkeypatch):
    core, trace, clock, _ = harness(monkeypatch)
    trace.unpatch()
    with pytest.raises(ValueError, match="frozen intervals"):
        m5_d9_observer._DualTrace(
            core, trace.runner, AsyncFuture, AsyncOutput, T0, [EARLY, (LATE[0], LATE[1] - 1)]
        )
    clock.stamp = T0
    with pytest.raises(ValueError, match="frozen future interval"):
        m5_d9_observer._DualTrace(core, trace.runner, AsyncFuture, AsyncOutput, T0, WINDOWS)


def test_d9_replay_arms_only_stepped_leaves_and_waits_past_measured_drain(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    replay = importlib.import_module("replay_m5")
    plan = {
        "role": "D9",
        "runs": ["thermal-pilot", "E1", "E2", "E3", "E4"],
        "capture": ["pilot", "endpoint", "step", "step", "endpoint"],
        "trace": {"windows_ns": m5_d9_observer.D9_WINDOWS_NS},
    }
    assert [leaf for leaf in plan["runs"] if replay._trace_capture(plan, leaf) == "step"] == [
        "E2",
        "E3",
    ]
    assert replay._trace_arm_arguments(plan, T0) == {
        "t0_ns": T0,
        "windows_ns": [list(pair) for pair in WINDOWS],
    }
    clock = Clock()
    clock.stamp = LATE[1] - 230_000_001
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        clock.stamp += round(seconds * 1_000_000_000)

    monkeypatch.setattr(replay, "time", clock)
    monkeypatch.setattr(replay, "asyncio", SimpleNamespace(sleep=sleep))
    asyncio.run(replay._wait_until(LATE[1]))
    assert sleeps == [0.23]
    asyncio.run(replay._wait_until(LATE[1]))
    assert sleeps == [0.23]
