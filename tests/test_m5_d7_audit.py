"""Controls for late D7 native joins, request denominators, and ABBA scope."""

import sys
from pathlib import Path

import pytest
from test_m5_d6_observer import Clock, Request
from test_m5_d7_observer import END, START, T0, arm, harness

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from m5_d7_audit import (  # noqa: E402
    assess_d7,
    audit_d7_trace,
    late_signature,
    paired_endpoints,
    waiting_summary,
)

LIMITS = {
    "start_offset_ns": START - T0,
    "window_ns": END - START,
    "arm_lead_ns": 250_000_000,
    "batch_limit": 1_024,
    "event_limit": 32_768,
}
GATES = {"overhead_ttft_ns": 2_000_000, "overhead_ttft_rel": 0.05, "overhead_throughput_rel": 0.05}


def producer_fixture(monkeypatch):
    clock = Clock()
    clock.stamp = T0 - 30
    core, runner, clock, _ = harness(monkeypatch, 5, clock=clock)
    recorder = arm(core, runner)
    clock.stamp = T0
    core.scheduler.add_request(Request("r"))
    core._process_engine_step()
    core._process_engine_step()
    clock.stamp = START
    core._process_input_queue()
    core._process_engine_step()
    core._process_engine_step()
    clock.stamp = END
    core._process_engine_step()
    core._process_engine_step()
    last = round(core.last_step[0][0].timestamp * 10**9)
    raw = recorder.export()
    before = raw["preroll_tokens"][0]["packet_ns"]
    stamps = [
        before,
        raw["carry_in"][0]["output_counts"][0]["packet_ns"],
        *(batch["output_counts"][0]["packet_ns"] for batch in raw["batches"]),
        last,
    ]
    enqueue = raw["enqueues"][0]
    client = {
        "d7_capture": "step",
        "d7_arm_return_ns": T0 - 5,
        "phases": {
            "warmup_end_ns": T0 - 40,
            "measure_start_ns": T0,
            "drain_end_ns": last + 1,
        },
        "requests": [
            {
                "request_id": "r",
                "submit_ns": T0,
                "status": "completed",
                "chunk_tokens": [{"at_ns": stamp + 1, "token_ids": [3]} for stamp in stamps],
                "native_metrics": {
                    "first": {
                        "queued_ts": enqueue["start_ns"] / 10**9,
                        "scheduled_ts": enqueue["end_ns"] / 10**9,
                        "first_token_ts": before / 10**9,
                    },
                    "last": {"last_token_ts": last / 10**9},
                },
            }
        ],
    }
    return raw, client, clock.stamp + 1


def test_live_shaped_carry_and_prefix_preserve_absolute_native_ordinals(monkeypatch):
    raw, client, end = producer_fixture(monkeypatch)
    report = audit_d7_trace(raw, client, LIMITS, leaf_end_ns=end)
    assert report["verified"] is True
    assert report["carry_batch_count"] == report["preroll_tokens"] == 1
    assert report["linked_late_real_tokens"] == 2
    assert [row["batch_id"] for row in report["links"]["r"]] == [None, 0, 1, 2]
    assert [row["ordinal"] for row in report["links"]["r"]] == [1, 2, 3, 4]
    assert report["links"]["r"][-1]["packet_ns"] > END


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("preroll_tokens", 0, "ordinal"), 2, "native packet ordinal"),
        (("preroll_tokens", 0, "packet_ns"), END, "pre-roll packet chronology"),
        (("preroll_tokens", 0, "token_id"), 99, "real client token"),
        (("carry_in", 0, "runner"), {}, "unavailable upstream execution"),
        (("boundary", "state", "pending_batch_ids"), [], "native FIFO"),
        (("steps", 0, "consumed_batch_id"), 1, "FIFO order"),
        (("batches", 0, "output_counts", 0, "ordinal"), 4, "native packet ordinal"),
        (("start_ns",), START + 1, "observed window"),
    ],
)
def test_late_trace_semantic_corruption_is_rejected(monkeypatch, path, value, error):
    raw, client, end = producer_fixture(monkeypatch)
    branch = raw
    for key in path[:-1]:
        branch = branch[key]
    branch[path[-1]] = value
    with pytest.raises(ValueError, match=error):
        audit_d7_trace(raw, client, LIMITS, leaf_end_ns=end)


def synthetic_client(*, delay=20_000_000, drain=8_000_000_000):
    return {
        "phases": {"measure_start_ns": T0, "drain_end_ns": T0 + drain},
        "requests": [
            {
                "request_id": identity,
                "chunk_tokens": [
                    {"at_ns": T0 + offset + delay * index, "token_ids": [1]}
                    for index in range(35)
                    if offset + delay * index <= drain
                ],
            }
            for identity, offset in (("a", START - T0 - delay), ("b", START - T0 - delay + 1))
        ],
    }


def test_late_gap_signature_uses_later_token_and_fixed_complete_bins():
    sig = late_signature(synthetic_client(delay=35_000_000))
    assert sig["candidate"] == "detected"
    assert sig["first_pair_start_offset_ns"] == 3_500_000_000
    assert sig["first_pair_decision_offset_ns"] == 4_500_000_000
    assert sig["bins"][0]["median_gap_ns"] == 35_000_000
    assert late_signature(synthetic_client())["candidate"] == "not_detected"
    assert (
        late_signature(synthetic_client(delay=35_000_000, drain=4_300_000_000))["candidate"]
        == "unverified"
    )


def test_noticeable_and_sustained_wait_are_different_native_anchors():
    client = {"phases": {"measure_start_ns": T0}}
    timing = {
        "status": "complete",
        "clock": {"verified": True},
        "per_id": {
            identity: {
                "submit_ns": T0 + start,
                "planned_ns": T0 + start,
                "first_components_ns": {
                    "submit_to_native_queued": 0,
                    "native_queued_to_first_scheduled": delay,
                },
            }
            for identity, start, delay in (
                ("a", 0, 50_000_000),
                ("b", 100_000_000, 250_000_000),
                ("c", 350_000_000, 250_000_000),
            )
        },
    }
    result = waiting_summary(client, timing)
    assert result["first_wait_ge_50ms"]["request_id"] == "a"
    assert result["first_continuous_wait_ge_500ms"] == {
        "onset_offset_ns": 100_000_000,
        "verified_at_offset_ns": 600_000_000,
        "end_offset_ns": 600_000_000,
    }
    assert result["wait_area_request_ns"] == 550_000_000


def metrics(p95):
    return {
        "strata": {"all": {"ttft_ns": {"p95": p95}}},
        "throughput": {"last_terminal_requests_per_s": 8.0, "last_terminal_tokens_per_s": 128.0},
    }


def test_abba_comparisons_preserve_all_orientations_and_do_not_promote_missing_runs():
    runs = {
        "E1": {"endpoint_evidence_complete": True, "metrics": metrics(100_000_000)},
        "E2": {"endpoint_evidence_complete": True, "metrics": metrics(103_000_000)},
        "E3": {"endpoint_evidence_complete": True, "metrics": metrics(110_000_000)},
        "E4": {"endpoint_evidence_complete": True, "metrics": metrics(111_000_000)},
    }
    pairs = paired_endpoints(runs, GATES)
    assert pairs["E1/E2"]["within_tolerance"]
    assert pairs["E4/E3"]["ttft_delta_ns"] == -1_000_000
    assert not pairs["E1/E3"]["within_tolerance"]
    partial = assess_d7(runs, {}, {}, GATES, heat_verified=False)
    assert partial["capture_equivalence"] == "unverified"
    assert set(partial["primary_capture_pairs"]) == {"E1/E2", "E4/E3"}
    assert partial["localization"] == "unresolved"
