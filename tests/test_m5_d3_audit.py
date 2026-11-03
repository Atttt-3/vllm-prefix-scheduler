"""Local controls for D3 raw timing reconstruction and verdict separation."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MS = 1_000_000
SECOND = 1_000_000_000
GATES = {"max_clock_rtt_ns": 5 * MS, "overhead_throughput_rel": 0.05}


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("m5_d3_audit", ROOT / "scripts/m5_d3_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ledger(queue_ms=4):
    origin = 80_000 * SECOND
    warmup = origin - SECOND
    load = warmup - SECOND
    drained = origin + 8 * SECOND
    initial = [
        {
            "send_ns": load + i * MS,
            "engine_ns": load + i * MS + 20,
            "recv_ns": load + i * MS + 40 + i,
        }
        for i in range(5)
    ]
    post = [
        {
            "send_ns": drained + i * MS,
            "engine_ns": drained + i * MS + 20,
            "recv_ns": drained + i * MS + 40 + i,
        }
        for i in range(5)
    ]
    identity = {
        "host": "fixture",
        "boot_id": "boot",
        "time_namespace": "time",
        "clock": "CLOCK_MONOTONIC",
    }
    client = {
        "clock": {**identity, "pid": 100},
        "clock_check": initial[0],
        "clock_samples": initial,
        "native_metrics_schema_version": 1,
        "native_metrics_clock": {
            "implementation": "clock_gettime(CLOCK_MONOTONIC)",
            "monotonic": True,
            "adjustable": False,
            "resolution_ns": 1,
            "post_clock_check": post[0],
            "post_clock_samples": post,
        },
        "phases": {"load_end_ns": load, "warmup_start_ns": warmup, "drain_end_ns": drained},
        "requests": [],
    }
    inputs = {"requests": []}
    for i in range(60):
        request_id = f"request-{i:06d}"
        plan = origin + i * 100 * MS
        submit = plan + MS
        queued = submit + 2 * MS
        scheduled = queued + queue_ms * MS
        native_first = scheduled + (30 + i) * MS
        native_second = native_first + MS
        native_last = native_first + 3 * MS
        client_first = native_second + MS
        client_last = native_last + 2 * MS
        terminal = client_last + MS
        stamps = {
            key: ts / SECOND
            for key, ts in zip(
                ("queued_ts", "scheduled_ts", "first_token_ts"),
                (queued, scheduled, native_first),
                strict=True,
            )
        }
        first = {
            **stamps,
            "client_received_ns": client_first,
            "last_token_ts": native_second / SECOND,
            "native_generation_tokens": 2,
            "native_preemptions": 0,
        }
        last = {
            **stamps,
            "client_received_ns": client_last,
            "last_token_ts": native_last / SECOND,
            "native_generation_tokens": 4,
            "native_preemptions": 1,
        }
        client["requests"].append(
            {
                "request_id": request_id,
                "status": "completed",
                "planned_ns": plan,
                "submit_ns": submit,
                "first_token_ns": client_first,
                "last_token_ns": client_last,
                "end_ns": terminal,
                "cached_tokens": 0,
                "token_ids": [1, 2, 3, 4],
                "chunk_tokens": [
                    {"at_ns": client_first, "token_ids": [1, 2]},
                    {"at_ns": client_last, "token_ids": [3, 4]},
                ],
                "native_metrics": {"first": first, "last": last},
            }
        )
        inputs["requests"].append({"request_id": request_id})
    return client, {"clock": {**identity, "pid": 200}}, inputs


def test_d3_decomposition_uses_true_first_stamp_of_a_coalesced_chunk(audit):
    before = audit.timing_leaf(*ledger(), GATES)
    after = audit.timing_leaf(*ledger(queue_ms=7), GATES)
    assert before["status"] == after["status"] == "complete"
    per_id = before["per_id"]["request-000048"]
    assert sum(per_id["first_components_ns"].values()) == per_id["ttft_ns"]
    assert per_id["first_components_ns"]["native_first_token_to_client_first_token"] == 2 * MS
    assert sum(per_id["end_components_ns"].values()) == per_id["end_ns"] - per_id["planned_ns"]
    assert per_id["native_preemptions"] == 1
    pair = audit.timing_pair(before, after)
    assert (
        pair["p95_anchors"]["reference_request_id"]
        == pair["p95_anchors"]["observed_request_id"]
        == "request-000056"
    )
    assert pair["p95_anchors"]["ttft_p95_difference_ns"] == 3 * MS
    assert (
        pair["p95_anchors"]["same_id_additive_contrast"]["first_component_deltas_ns"][
            "native_queued_to_first_scheduled"
        ]
        == 3 * MS
    )
    assert pair["archived_request_000048"]["ttft_delta_ns"] == 3 * MS
    assert pair["last_15_planned"]["ids"][0] == "request-000045"
    assert pair["last_15_planned"]["ttft_delta_ns"]["n"] == 15
    assert pair["last_terminal"]["last_plan_to_terminal_delta_ns"] == sum(
        pair["last_terminal"]["component_deltas_ns"].values()
    )


@pytest.mark.parametrize("change", ("missing", "chunk_count", "backwards", "clock", "late_bracket"))
def test_d3_missing_native_timing_retains_endpoint_rows(audit, change):
    client, engine, inputs = ledger()
    row = client["requests"][48]
    native = row["native_metrics"]
    if change == "missing":
        native["last"] = None
    elif change == "chunk_count":
        native["first"]["native_generation_tokens"] = 3
    elif change == "backwards":
        native["last"]["native_preemptions"] = 0
        native["first"]["native_preemptions"] = 1
    elif change == "clock":
        engine["clock"]["boot_id"] = "other"
    else:
        client["native_metrics_clock"]["post_clock_check"] = client["native_metrics_clock"][
            "post_clock_samples"
        ][1]
    result = audit.timing_leaf(client, engine, inputs, GATES)
    per_id = result["per_id"]["request-000048"]
    assert result["planned_count"] == 60
    assert per_id["status"] == "completed"
    assert per_id["ttft_ns"] == row["first_token_ns"] - row["planned_ns"]
    assert per_id["issues"] and per_id["end_components_ns"] is None
    assert result["status"] == ("unverified" if change in ("clock", "late_bracket") else "partial")


def test_d3_unmatched_p95_order_statistics_never_add(audit):
    before = audit.timing_leaf(*ledger(), GATES)
    after = deepcopy(before)
    row = after["per_id"]["request-000056"]
    row["ttft_ns"] -= 20 * MS
    row["first_components_ns"]["native_first_scheduled_to_first_token"] -= 20 * MS
    pair = audit.timing_pair(before, after)
    assert pair["p95_anchors"]["reference_request_id"] != pair["p95_anchors"]["observed_request_id"]
    assert pair["p95_anchors"]["same_id_additive_contrast"] is None
    assert pair["archived_request_000048"]["ttft_delta_ns"] == 0


def test_d3_gate_breach_does_not_imply_archived_direction(audit):
    runs = {
        name: {"pre_t0_verified": True, "prelude": {"verified": True}}
        for name in ("E1", "E2", "E3", "E4")
    }
    matched = {
        "within_tolerance": True,
        "ttft_within_tolerance": True,
        "throughput_within_tolerance": True,
        "ttft_delta_ns": 0,
        "relative_throughput_delta": {"requests": 0, "tokens": 0},
    }
    pairs = [
        {
            "reference": a,
            "observed": b,
            "thermal": {"comparable": True},
            "performance": dict(matched),
        }
        for a, b in (("E1", "E2"), ("E3", "E4"), ("E1", "E4"))
    ]
    bad = {
        **matched,
        "within_tolerance": False,
        "ttft_within_tolerance": False,
        "throughput_within_tolerance": False,
        "ttft_delta_ns": -80 * MS,
        "relative_throughput_delta": {"requests": 0.06, "tokens": 0.06},
    }
    pairs[1]["performance"] = {**bad, "ttft_delta_ns": 80 * MS}
    result = audit.assess_d3(runs, pairs, GATES)
    assert result["same_tier_fluctuation"] == "observed"
    assert result["archived_e3_e4_pattern"] == "not_observed"
    assert not result["native_timing_complete"] and not result["qualification_started"]
    pairs[1]["performance"] = bad
    assert audit.assess_d3(runs, pairs, GATES)["archived_e3_e4_pattern"] == "observed"
    pairs[1]["thermal"]["comparable"] = False
    assert audit.assess_d3(runs, pairs, GATES)["archived_e3_e4_pattern"] == "unverified"


def test_d3_timing_source_binding(audit):
    expected = dict.fromkeys(audit.D3_NATIVE_SOURCES, "a" * 64)
    plan = {"native_timing": {"schema_version": 1, "source_sha256": expected}}
    source = {
        "upstream_sha256": dict(expected),
        "upstream_modules": {name: {"sha256": sha} for name, sha in expected.items()},
    }
    audit.verify_native_sources(plan, source)
    source["upstream_modules"][next(iter(expected))]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="timing source differs"):
        audit.verify_native_sources(plan, source)
