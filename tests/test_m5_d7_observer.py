"""Exercise D7's delayed native FIFO boundary and absolute output ordinals."""

from types import SimpleNamespace

import pytest
from test_m5_d6_observer import (
    AsyncFuture,
    AsyncOutput,
    Clock,
    Core,
    Executor,
    Request,
    Schedule,
    Scheduler,
)

from vllm_prefix_scheduler import m5_d6_observer, m5_d7_observer

T0 = 10_000_000_000
START = T0 + m5_d7_observer.D7_START_OFFSET_NS
END = T0 + m5_d7_observer.D7_END_OFFSET_NS


def harness(monkeypatch, count, *, clock=None):
    clock = Clock() if clock is None else clock
    monkeypatch.setattr(m5_d6_observer, "time", clock)
    monkeypatch.setattr(m5_d7_observer, "time", clock)
    events = []
    runner = SimpleNamespace(execute_model_state=None)
    scheduler = Scheduler(clock, events, [Schedule({"r": 1}) for _ in range(count)])
    core = Core(scheduler, Executor(runner, events))
    return core, runner, clock, events


def arm(core, runner):
    return m5_d7_observer._DelayedTrace(core, runner, AsyncFuture, AsyncOutput, T0, START, END)


def test_d7_carry_and_window_use_existing_fifo_once(monkeypatch):
    core, runner, clock, events = harness(monkeypatch, 5)
    trace = arm(core, runner)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    core._process_engine_step()
    core._process_engine_step()
    assert events.count("materialize") == 1
    assert trace.counts["batches"] == 0
    clock.stamp = START
    core._process_input_queue()
    core._process_engine_step()
    core._process_engine_step()
    clock.stamp = END
    core._process_engine_step()
    core._process_engine_step()
    report = trace.export()
    assert report["status"] == "complete"
    assert report["t0_ns"] == T0
    assert START <= report["boundary"]["at_ns"] < report["loops"][0]["start_ns"]
    assert report["boundary"]["state"]["pending_batch_ids"] == [0]
    assert report["steps"][0]["queue_before"] == 1
    [carry] = report["carry_in"]
    assert carry["batch_id"] == 0 and "runner" not in carry
    assert carry["scheduled"] == [{"request_id": "r", "generation": 0, "tokens": 1}]
    assert carry["future_result_start_ns"] > report["boundary"]["at_ns"]
    assert carry["output_counts"][0]["ordinal"] == 2
    assert [row["ordinal"] for row in report["preroll_tokens"]] == [1]
    assert [row["batch_id"] for row in report["batches"]] == [1, 2]
    assert [row["output_counts"][0]["ordinal"] for row in report["batches"]] == [3, 4]
    assert [row["consumed_batch_id"] for row in report["steps"]] == [0, 1, 2]
    assert report["batches"][-1]["consume_end_ns"] > END
    assert (
        events.count("materialize") == events.count("fifo-result") == events.count("consume") == 5
    )
    assert core.step_fn == core.step


def test_d7_left_straddler_remains_native_preroll_and_idle_boundary_is_empty(monkeypatch):
    core, runner, clock, _ = harness(monkeypatch, 1)
    original_consume = core.scheduler.update_from_output

    def crossing(schedule, output):
        clock.stamp = START + 100
        return original_consume(schedule, output)

    core.scheduler.update_from_output = crossing
    trace = arm(core, runner)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    core._process_engine_step()
    clock.stamp = START - 100
    core._process_engine_step()
    core.scheduler.plans.append(Schedule({"r": 1}))
    core._process_engine_step()
    core._process_engine_step()
    clock.stamp = END
    report = trace.export()
    assert report["status"] == "complete"
    [prefix] = report["preroll_tokens"]
    assert START < prefix["packet_ns"] < report["boundary"]["at_ns"]
    assert report["boundary"]["state"]["pending_batch_ids"] == []
    assert report["carry_in"] == []
    assert report["steps"][0]["queue_before"] == 0
    assert report["batches"][0]["output_counts"][0]["ordinal"] == 2


def test_d7_cannot_arm_after_t0_or_change_window(monkeypatch):
    core, runner, clock, _ = harness(monkeypatch, 0)
    with pytest.raises(ValueError, match="frozen delayed interval"):
        m5_d7_observer._DelayedTrace(core, runner, AsyncFuture, AsyncOutput, T0, START - 1, END)
    clock.stamp = T0
    with pytest.raises(ValueError, match="frozen future interval"):
        arm(core, runner)
    assert core.step_fn == core.step


def test_d7_preroll_cap_fails_closed_without_extra_native_result(monkeypatch):
    monkeypatch.setattr(m5_d7_observer, "D7_PREROLL_TOKEN_LIMIT", 1)
    core, runner, clock, events = harness(monkeypatch, 4)
    trace = arm(core, runner)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    for _ in range(3):
        core._process_engine_step()
    clock.stamp = START
    core._process_engine_step()
    core._process_engine_step()
    clock.stamp = END
    report = trace.export()
    assert report["status"] == "incomplete" and report["overflow"]
    assert len(report["preroll_tokens"]) == 1
    assert (
        events.count("materialize") == events.count("fifo-result") == events.count("consume") == 4
    )
