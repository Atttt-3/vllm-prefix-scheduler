"""D6 controls exercise the independent FIFO, request and packet joins."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from m5_d6_audit import assess_d6, audit_step_trace, packet_path, queue_transitions  # noqa: E402
from m5_d6_signature import HORIZON_NS  # noqa: E402

T0 = 10_000_000_000
END = T0 + HORIZON_NS
EXPORT = END + 250
LIMITS = {
    "window_ns": HORIZON_NS,
    "arm_lead_ns": 250_000_000,
    "event_limit": 32_768,
    "batch_limit": 1024,
}


def state(pending, *, active=True):
    return {
        "running": [["a", 0]] if active else [],
        "waiting": [],
        "kv_holding_waiting": [],
        "pending_batch_ids": pending,
    }


def batch(number, schedule, consume, schedule_pending, consume_pending, *, token=None):
    request = {"request_id": "a", "generation": 0}
    positive = token is not None
    return {
        "batch_id": number,
        "effective_active_cap": 4,
        "admission_stop_reason": "unknown",
        "schedule_start_ns": T0 + schedule,
        "schedule_end_ns": T0 + schedule + 1,
        "schedule_before": state(schedule_pending),
        "schedule_after": state(schedule_pending),
        "scheduled": [
            {
                **request,
                "tokens": 1,
                "start": number - 1,
                "end": number,
                "prompt_length": 641,
                "preemptions": 0,
            }
        ]
        if positive
        else [],
        "execute_start_ns": T0 + schedule + 2,
        "execute_end_ns": T0 + schedule + 3,
        "sample_start_ns": T0 + schedule + 4 if positive else None,
        "sample_end_ns": T0 + schedule + 5 if positive else None,
        "future_result_start_ns": T0 + consume,
        "materialize_start_ns": T0 + consume + 1 if positive else None,
        "materialize_end_ns": T0 + consume + 2 if positive else None,
        "future_result_end_ns": T0 + consume + 3,
        "consume_start_ns": T0 + consume + 4,
        "consume_end_ns": T0 + consume + 6,
        "consume_before": state(consume_pending),
        "consume_after": state(consume_pending),
        "runner": {
            "effective_tokens": 1,
            "padded_tokens": 1,
            "rows": [{**request, "query_start": 0, "query_end": 1, "prefill_length": 641}],
        }
        if positive
        else None,
        "materialized_counts": [{**request, "generated_tokens": 1}] if positive else [],
        "output_counts": [
            {**request, "generated_tokens": 1, "token_ids": [token], "packet_ns": T0 + consume + 5}
        ]
        if positive
        else [],
    }


def fixture():
    steps = [
        {
            "step_id": index,
            "start_ns": T0 + lo,
            "end_ns": T0 + hi,
            "queue_before": before,
            "queue_after": after,
            "scheduled_batch_id": dispatch,
            "consumed_batch_id": consume,
            "model_executed": dispatch in {1, 2},
        }
        for index, (lo, hi, before, after, dispatch, consume) in enumerate(
            (
                (10, 20, 0, 1, 1, None),
                (22, 49, 1, 1, 2, 1),
                (52, 79, 1, 1, 3, 2),
                (82, 99, 1, 0, None, 3),
            ),
            1,
        )
    ]
    trace = {
        "schema_version": 1,
        "status": "complete",
        "overflow": False,
        "errors": [],
        "start_ns": T0,
        "end_ns": END,
        "armed_ns": T0 - 8,
        "export_ns": EXPORT,
        "input_polls": [{"start_ns": T0, "end_ns": T0 + 7}],
        "enqueues": [{"request_id": "a", "generation": 0, "start_ns": T0 + 2, "end_ns": T0 + 5}],
        "steps": steps,
        "loops": [
            {
                "step_id": row["step_id"],
                "start_ns": row["start_ns"] - 1,
                "end_ns": row["end_ns"] + 1,
                "model_executed": row["model_executed"],
            }
            for row in steps
        ],
        "batches": [
            batch(1, 12, 38, [], [2], token=17),
            batch(2, 24, 68, [1], [3], token=19),
            batch(3, 54, 88, [2], [], token=None),
        ],
    }
    client = {
        "d6_capture": "step",
        "d6_arm_return_ns": T0 - 3,
        "phases": {"measure_start_ns": T0, "warmup_end_ns": T0 - 30, "drain_end_ns": END + 100},
        "requests": [
            {
                "request_id": "a",
                "submit_ns": T0,
                "status": "completed",
                "chunk_tokens": [
                    {"at_ns": T0 + 50, "token_ids": [17]},
                    {"at_ns": T0 + 80, "token_ids": [19]},
                ],
                "native_metrics": {
                    "first": {
                        "queued_ts": (T0 + 3) / 10**9,
                        "scheduled_ts": (T0 + 12) / 10**9,
                        "first_token_ts": (T0 + 43) / 10**9,
                    },
                    "last": {"last_token_ts": (T0 + 73) / 10**9},
                },
            }
        ],
    }
    return trace, client


def audit(trace, client):
    return audit_step_trace(trace, client, LIMITS, leaf_end_ns=EXPORT + 1)


def test_real_token_ordinals_bind_to_native_fifo_packet_construction():
    trace, client = fixture()
    report = audit(trace, client)
    assert report["verified"] is True
    assert (report["linked_real_tokens"], report["positive_batches"], report["empty_batches"]) == (
        2,
        2,
        1,
    )
    earlier, later = report["links"]["a"]
    assert (earlier["ordinal"], later["ordinal"]) == (1, 2)
    assert earlier["packet_ns"] == T0 + 43
    path = packet_path(report, earlier, later)
    assert sum(path.values()) == 30
    # The successor's native scheduler/execute interval preceded the previous
    # packet because vLLM kept work in flight. It cannot count twice.
    assert path["scheduler_call"] == path["executor_call"] == 0
    assert path["sample_to_fifo_result"] == 25
    assert path["packet_to_client_envelope_change"] == 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("steps", 1, "consumed_batch_id"), 2, "native FIFO order"),
        (("loops", 1, "step_id"), 3, "outer-loop chronology"),
        (("input_polls", 0, "end_ns"), T0 + 10, "polling overlaps"),
        (("batches", 1, "schedule_before", "pending_batch_ids"), [], "scheduler snapshot"),
        (("batches", 0, "consume_after", "pending_batch_ids"), [], "consumption snapshot"),
        (("batches", 0, "scheduled", 0, "generation"), 1, "scheduler request generation"),
        (("batches", 0, "runner", "rows", 0, "query_end"), 2, "query span"),
        (("batches", 0, "materialized_counts", 0, "generated_tokens"), 0, "materialized token"),
        (("batches", 1, "output_counts", 0, "token_ids"), [20], "client receipt"),
        (("batches", 1, "output_counts", 0, "packet_ns"), T0 + 72, "packet timestamp"),
        (("batches", 1, "output_counts"), [], "real token lacks"),
        (("overflow",), True, "recorder health"),
    ],
)
def test_complete_trace_rejects_semantic_mutations(path, value, message):
    trace, client = fixture()
    branch = trace
    for key in path[:-1]:
        branch = branch[key]
    branch[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        audit(trace, client)


def test_stop_and_overflow_are_evidence_for_unresolved_capture_only():
    _, client = fixture()
    report = audit({"schema_version": 1, "status": "incomplete", "overflow": True}, client)
    assert report["verified"] is False
    assert report["overflow"] is True
    not_started = {"schema_version": 1, "status": "not-started"}
    with pytest.raises(ValueError, match="not-started trace has"):
        audit(not_started, client)
    rejected = copy.deepcopy(client)
    del rejected["phases"]["measure_start_ns"]
    del rejected["d6_arm_return_ns"]
    assert audit(not_started, rejected)["verified"] is False


def test_pinned_shaped_cpu_producer_export_passes_independent_join(monkeypatch):
    from types import SimpleNamespace

    from test_m5_d6_observer import (
        AsyncFuture,
        AsyncOutput,
        Clock,
        Core,
        Executor,
        Request,
        Schedule,
    )
    from test_m5_d6_observer import Scheduler as FakeScheduler

    from vllm_prefix_scheduler import m5_d6_observer as observer

    clock = Clock()
    clock.stamp = T0 - 20
    monkeypatch.setattr(observer, "time", clock)
    events = []
    runner = SimpleNamespace(execute_model_state=None)
    scheduler = FakeScheduler(
        clock, events, [Schedule({"a": 3}), Schedule({"a": 2}), Schedule({"a": 1})]
    )
    core = Core(scheduler, Executor(runner, events))
    recorder = observer._Trace(core, runner, AsyncFuture, AsyncOutput, T0, END)
    clock.stamp = T0
    core.scheduler.add_request(Request("a"))
    core._process_engine_step()
    core._process_engine_step()
    clock.stamp = END
    core._process_engine_step()
    core._process_engine_step()
    last = round(core.last_step[0][0].timestamp * 10**9)
    raw = recorder.export()
    first, second = (batch["output_counts"][0]["packet_ns"] for batch in raw["batches"])
    client = {
        "d6_capture": "step",
        "d6_arm_return_ns": T0 - 1,
        "phases": {
            "warmup_end_ns": T0 - 25,
            "measure_start_ns": T0,
            "drain_end_ns": last + 1,
        },
        "requests": [
            {
                "request_id": "a",
                "submit_ns": T0,
                "status": "completed",
                "chunk_tokens": [
                    {"at_ns": time + 1, "token_ids": [3]} for time in (first, second, last)
                ],
            }
        ],
    }
    result = audit_step_trace(raw, client, LIMITS, leaf_end_ns=clock.stamp + 1)
    assert result["verified"] is True
    assert result["linked_real_tokens"] == 2
    assert result["links"]["a"][1]["packet_ns"] > END


def classify_inputs():
    traces = {}
    for leaf in ("E2", "E3"):
        trace, client = fixture()
        traces[leaf] = audit(trace, client)
    delta = 40_000_000
    slow = traces["E3"]
    for key in ("materialize_end_ns", "future_result_end_ns", "consume_start_ns", "consume_end_ns"):
        slow["batches"]["2"][key] += delta
    for key in ("packet_ns", "client_ns"):
        slow["links"]["a"][1][key] += delta

    for leaf, trace in traces.items():
        source = trace["batches"]["2"]
        third = copy.deepcopy(source)
        third["batch_id"] = 4
        for key, value in source.items():
            if key.endswith("_ns"):
                third[key] = value + 30 + (delta if leaf == "E3" else 0)
        trace["batches"]["4"] = third
        trace["fifo"]["4"] = {
            **trace["fifo"]["2"],
            "schedule_step_id": 4,
            "consume_step_id": 5,
        }
        previous = trace["links"]["a"][1]
        trace["links"]["a"].append(
            {
                **previous,
                "ordinal": 3,
                "batch_id": 4,
                "packet_ns": previous["packet_ns"] + 30 + (delta if leaf == "E3" else 0),
                "client_ns": previous["client_ns"] + 30 + (delta if leaf == "E3" else 0),
            }
        )
        if leaf == "E3":
            for key in (
                "schedule_start_ns",
                "schedule_end_ns",
                "execute_start_ns",
                "execute_end_ns",
                "sample_start_ns",
                "sample_end_ns",
                "future_result_start_ns",
                "materialize_start_ns",
            ):
                third[key] -= delta

    def gap_client(interval):
        return {
            "phases": {"measure_start_ns": T0},
            "requests": [
                {
                    "request_id": name,
                    "chunk_tokens": [
                        {"at_ns": T0 + offset + ordinal * interval, "token_ids": [ordinal]}
                        for ordinal in range(55)
                    ],
                }
                for name, offset in (("a", 50), ("b", 100))
            ],
        }

    fast, slower = gap_client(20_000_000), gap_client(delta)
    clients = {"E1": slower, "E2": fast, "E3": slower}
    gates = {
        "overhead_ttft_ns": 2_000_000,
        "overhead_ttft_rel": 0.05,
        "overhead_throughput_rel": 0.05,
    }
    runs = {
        name: {
            "endpoint_evidence_complete": True,
            "native_timing": {
                "status": "complete",
                "per_id": {"a": {"cached_tokens": 0, "native_preemptions": 0}},
            },
            "metrics": {
                "strata": {"all": {"ttft_ns": {"p95": 20_000_000}}},
                "throughput": {
                    "last_terminal_requests_per_s": 8,
                    "last_terminal_tokens_per_s": 128,
                },
            },
        }
        for name in clients
    }
    return runs, clients, traces, gates


def test_only_one_matched_host_envelope_can_localize_candidate():
    result = assess_d6(*classify_inputs(), heat_verified=True)
    assert result["localization"] == "native_materialization_host_envelope"
    assert result["candidate"]["generated_token_ordinal"] == 2
    assert (
        result["candidate"]["slow_minus_fast_ns"]["native_materialization_host_envelope"]
        == 40_000_000
    )
    assert "GPU" not in result["localization"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("no_heat", "thermal_request_or_trace_evidence_incomplete"),
        ("no_endpoint_reproduction", "endpoint_signature_not_reproduced"),
        ("both_slow", "stepped_fast_and_slow_contrast_missing"),
        ("different_composition", "host_envelopes_do_not_isolate_one_repeated_boundary"),
        ("multiple_boundaries", "host_envelopes_do_not_isolate_one_repeated_boundary"),
    ],
)
def test_localization_stays_unresolved_when_independent_comparison_fails(mutation, reason):
    runs, clients, traces, gates = classify_inputs()
    if mutation == "no_endpoint_reproduction":
        clients["E1"] = clients["E2"]
    elif mutation == "both_slow":
        clients["E2"] = clients["E3"]
    elif mutation == "different_composition":
        traces["E3"]["batches"]["2"]["runner"]["padded_tokens"] += 1
    elif mutation == "multiple_boundaries":
        traces["E3"]["links"]["a"][1]["client_ns"] += 40_000_000
        traces["E3"]["links"]["a"][2]["client_ns"] += 80_000_000
    result = assess_d6(runs, clients, traces, gates, heat_verified=mutation != "no_heat")
    assert result["localization"] == "unresolved"
    assert result["reason"] == reason


def test_native_full_slots_release_then_admit_the_waiting_request():
    trace, client = fixture()
    report = audit(trace, client)
    old = ["a", "b", "c", "d"]
    ready = ["a", "b", "c", "x"]

    def occupancy(active, waiting):
        return {
            "running": [[identity, index] for index, identity in enumerate(active)],
            "waiting": [[identity, index + 4] for index, identity in enumerate(waiting)],
            "kv_holding_waiting": [],
            "pending_batch_ids": [],
        }

    first, second = (report["batches"][key] for key in ("1", "2"))
    first.update(
        schedule_before=occupancy(old[:3], ["d", "x"]),
        schedule_after=occupancy(old, ["x"]),
        consume_before=occupancy(old, ["x", "y"]),
        consume_after=occupancy(old[:3], ["x", "y"]),
    )
    second.update(
        schedule_start_ns=T0 + 50,
        schedule_end_ns=T0 + 51,
        schedule_before=occupancy(old[:3], ["x", "y"]),
        schedule_after=occupancy(ready, ["y"]),
        consume_before=occupancy(ready, ["y"]),
        consume_after=occupancy(ready, ["y"]),
        scheduled=[{"request_id": "x"}],
    )
    report["enqueue_end_ns"]["x"] = T0 + 9
    queue = queue_transitions(report, 0, 100)
    assert queue["first_sustained_full_wait"]["waiting"] == 1
    assert queue["peak_full_waiting"]["waiting"] == 2
    [handoff] = queue["release_to_admission"]
    assert handoff["released_requests"] == ["d"]
    assert handoff["request_id"] == "x"
    assert handoff["release_end_to_admission_start_ns"] == 6
    assert handoff["observed_wait_until_admission_ns"] == 41
    assert queue["first_observed_slot_holds"]["d"] == {
        "admission_batch_id": 1,
        "admission_step_id": 1,
        "release_batch_id": 1,
        "release_step_id": 2,
        "duration_lower_ns": 29,
        "duration_upper_ns": 32,
    }

    # An intervening unlinked active admission invalidates the claimed join.
    second["scheduled"] = [{"request_id": "y"}]
    assert not queue_transitions(report, 0, 100)["release_to_admission"]
