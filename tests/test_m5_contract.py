"""Local contract fixtures only. No engine, GPU or formal conclusions."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_contract import (
    TASK_ID,
    add_interval,
    audit_matrix,
    audit_run,
    comparable,
    paired_summary,
    pressure_metrics,
    quantile,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "check_m5_trace.py"
DEADLINE = 1000


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def setup_capture(tmp_path):
    config = {
        "task_id": TASK_ID,
        "request_count": 500,
        "block_size": 2,
        "window": 16,
        "max_wait_ms": 500,
        "tokenizer_revision": "revision",
        "sampling": {"ignore_eos": True},
        "scenarios": {"mixed": {"prefix_tokens": 2, "output_tokens": 2}},
        "gates": {
            "deadline_ns": DEADLINE,
            "max_clock_rtt_ns": 10,
            "submit_p99_ns": 5,
            "submit_max_ns": 10,
        },
        "matrix": [
            {
                "id": "mixed-01",
                "scenario": "mixed",
                "seed": 10,
                "order": ["native", "off", "guarded", "prefix"],
            },
        ],
    }
    cfg_path = put(tmp_path / "config.json", config)
    inputs = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "scenario": "mixed",
        "seed": 10,
        "block_size": 2,
        "rate_rps": 1_000_000_000,
        "interval_ms": None,
        "sampling": {"ignore_eos": True, "max_tokens": 2},
        "tokenizer_revision": "revision",
        "warmup": [],
        "requests": [
            {
                "request_id": str(i),
                "class": "hot" if i == 0 else "cold",
                "family": "F" if i == 0 else None,
                "cache_salt": str(i),
                "offset_ns": i,
                "prompt_token_ids": [i, 0, 2],
            }
            for i in range(4)
        ],
    }
    input_path = put(tmp_path / "inputs.json", inputs)
    clock = {
        "host": "test",
        "pid": 2,
        "boot_id": "boot",
        "time_namespace": "time",
        "clock": "CLOCK_MONOTONIC",
    }
    client = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "phase": "calibration",
        "scenario": "mixed",
        "seed": 10,
        "mode": "prefix",
        "pair_id": "mixed-01",
        "tier": "light",
        "config_sha256": sha256(cfg_path),
        "input_sha256": sha256(input_path),
        "deadline_ns": DEADLINE,
        "drain_deadline_ns": 100 + 3 + DEADLINE,
        "clock": {**clock, "pid": 1},
        "clock_check": {"send_ns": 2, "engine_ns": 3, "recv_ns": 4},
        "phases": {
            "load_start_ns": 1,
            "load_end_ns": 5,
            "warmup_start_ns": 6,
            "warmup_end_ns": 10,
            "measure_start_ns": 100,
            "arrival_end_ns": 103,
            "drain_end_ns": 1103,
        },
        "requests": [],
        "primer": {"request_id": "runtime-primer", "status": "completed"},
        "warmup": [],
    }
    for i, (status, terminal, chunks) in enumerate(
        (
            ("completed", 145, [(140, [31]), (144, [32])]),
            ("timeout", 1101, [(160, [33])]),
            ("failed", 200, []),
            ("unfinished", None, []),
        )
    ):
        client["requests"].append(
            {
                "request_id": str(i),
                "planned_ns": 100 + i,
                "submit_ns": 100 + i,
                "status": status,
                "end_ns": terminal,
                "first_token_ns": chunks[0][0] if chunks else None,
                "last_token_ns": chunks[-1][0] if chunks else None,
                "token_ids": [t for _, ids in chunks for t in ids],
                "cached_tokens": 0 if chunks else None,
                "chunk_tokens": [{"at_ns": at, "token_ids": tokens} for at, tokens in chunks],
            }
        )
    engine = {
        "schema_version": 1,
        "clock": clock,
        "profile": {"num_gpu_blocks": 129},
        "errors": [],
        "enqueues": [
            {"request_id": str(i), "generation": i, "enqueued_ns": 105 + i} for i in range(4)
        ],
        "batches": [],
        "decisions": [],
        "evictions": [],
        "preemptions": [],
        "allocations": [],
        "boundaries": [],
        "overhead": [],
    }
    for batch_id, (identity, start, end, at) in enumerate(
        (
            ("0", 0, 3, 120),
            ("0", 1, 4, 141),
            ("1", 0, 3, 150),
        )
    ):
        base = {"request_id": identity, "generation": int(identity), "start": start, "end": end}
        engine["batches"].append(
            {
                "batch_id": batch_id,
                "scheduled_ns": at,
                "schedule_ns": 2,
                "scheduled": [{**base, "prompt_length": 3, "preemptions": int(start > 0)}],
                "runner": {
                    "effective_tokens": end - start,
                    "padded_tokens": end - start + 1,
                    "rows": [
                        {**base, "position_source": "runner_positions_d2h", "prefill_length": 3}
                    ],
                },
                "outputs": [
                    {
                        "request_id": identity,
                        "generation": int(identity),
                        "token_ids": [[31], [32], [33]][batch_id],
                    },
                ],
                "materialized": True,
                "consumed": True,
            }
        )
        engine["decisions"].append(
            {
                "batch_id": batch_id,
                "decision_ns": at - 2,
                "candidates": ["1", "0"] if batch_id == 0 else [],
                "ordered": ["0", "1"] if batch_id == 0 else [],
                "protected": [],
            }
        )
    for row in engine["enqueues"]:
        i = int(row["request_id"])
        row["prompt_sha256"] = hashlib.sha256(f"[{i},0,2]".encode()).hexdigest()
        row["block_hashes"] = [f"block{i}"]
    engine["enqueues"].insert(
        0,
        {
            "request_id": "runtime-primer",
            "generation": 100,
            "enqueued_ns": 6,
            "prompt_sha256": engine["enqueues"][0]["prompt_sha256"],
            "block_hashes": ["block0"],
        },
    )
    warmup_interval = {
        "request_id": "runtime-primer",
        "generation": 100,
        "start": 0,
        "end": 3,
    }
    engine["batches"].insert(
        0,
        {
            "batch_id": -1,
            "stage": "warmup",
            "scheduled_ns": 7,
            "schedule_ns": 1,
            "scheduled": [{**warmup_interval, "prompt_length": 3, "preemptions": 0}],
            "runner": {
                "effective_tokens": 3,
                "padded_tokens": 3,
                "rows": [
                    {
                        **warmup_interval,
                        "position_source": "runner_positions_d2h",
                        "prefill_length": 3,
                    }
                ],
            },
            "outputs": [],
            "materialized": True,
            "consumed": True,
        },
    )
    engine["decisions"].insert(
        0,
        {
            "batch_id": -1,
            "decision_ns": 6,
            "candidates": ["runtime-primer"],
            "ordered": ["runtime-primer"],
            "protected": [],
            "window_kind": "production",
        },
    )
    for i, (batch, decision) in enumerate(
        zip(engine["batches"], engine["decisions"], strict=True), 1
    ):
        batch["batch_id"] = decision["batch_id"] = i
        batch["stage"] = "warmup" if i == 1 else "measure"
        decision["window_kind"] = "production"
    resource_keys = (
        "requests",
        "waiting",
        "running",
        "holding",
        "deferred",
        "finished_req_ids",
        "reset_preempted_req_ids",
        "inflight_prefills",
        "guard_records",
        "observation_errors",
        "pending_batches",
    )
    count_keys = (
        "deferred_frees",
        "request_blocks",
        "non_null_ref_sum",
        "inflight_tokens",
        "stale_tokens",
        "output_placeholders",
        "m5_pending",
    )
    engine["boundaries"] = [
        {
            "label": label,
            "at_ns": at,
            "runner_request_ids": [],
            "runner_pending": False,
            **{k: [] for k in resource_keys},
            **dict.fromkeys(count_keys, 0),
            "guard_fault_code": None,
            "free_count": 128,
            "free_unique_count": 128,
            "free_traversed_count": 128,
            "total_non_null_blocks": 128,
            "cached_block_count": 0,
        }
        for label, at in (
            ("pre-reset", 8),
            ("cache-reset", 9),
            ("measure-start", 10),
            ("measure-drained", 1104),
        )
    ]
    put(tmp_path / "client.json", client)
    put(tmp_path / "engine.json", engine)
    return cfg_path, input_path, client, engine


def call_run(tmp_path, cfg, inputs):
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "run",
            "--phase",
            "calibration",
            "--pair",
            "mixed-01",
            "--mode",
            "prefix",
            "--config",
            str(cfg),
            "--inputs",
            str(inputs),
            "--run",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def setup_capture_v2(tmp_path, *, tier="light"):
    cfg, inputs, client, engine = setup_capture(tmp_path)
    config = json.loads(cfg.read_text())
    config["measurement_schema_version"] = 2
    config["scenarios"]["all-cold"] = {
        "prefix_tokens": 2,
        "suffix_tokens": 1,
        "output_tokens": 2,
        "families": 0,
        "hot_percent": 0,
        "prime_families": 0,
    }
    put(cfg, config)
    workload = json.loads(inputs.read_text())
    workload["scenario"] = "all-cold"

    def renumber(rid):
        return rid if rid == "runtime-primer" else f"request-{int(rid):06}"

    for row in workload["requests"]:
        row.update(request_id=renumber(row["request_id"]), **{"class": "cold", "family": None})
    put(inputs, workload)
    client.update(
        schema_version=2,
        scenario="all-cold",
        input_sha256=sha256(inputs),
        config_sha256=sha256(cfg),
        tier=tier,
    )
    # Last terminal is 1101 ns while actual gather finishes at 1102 ns.
    # The fixed comparison horizon stays 1103 ns.
    client["phases"]["drain_end_ns"] = 1102
    for row in (*client["requests"], *engine["enqueues"]):
        row["request_id"] = renumber(row["request_id"])
    engine["schema_version"] = 2
    for batch in engine["batches"]:
        batch["has_candidates"] = False
        if tier == "profile":
            batch["schedule_envelope_ns"] = batch["schedule_ns"] + 2
            engine["overhead"].append(
                {
                    "batch_id": batch["batch_id"],
                    "stage": batch["stage"],
                    **dict.fromkeys(
                        ("window_ns", "score_ns", "sort_inclusive_ns", "commit_ns", "reorder_ns"),
                        0,
                    ),
                }
            )
        for row in (*batch["scheduled"], *batch["runner"]["rows"], *batch["outputs"]):
            row["request_id"] = renumber(row["request_id"])
        scheduled = batch["scheduled"][0]
        engine["allocations"].append(
            {
                "allocation_id": batch["batch_id"],
                "batch_id": batch["batch_id"],
                "stage": batch["stage"],
                "request_id": scheduled["request_id"],
                "generation": scheduled["generation"],
                "free_before": 128,
                "free_after": 127,
                "success": True,
                "hit_tokens": 0,
                "at_ns": batch["scheduled_ns"],
                "eviction_count": int(batch["batch_id"] == 2),
            }
        )
    engine["decisions"] = [row for row in engine["decisions"] if row["candidates"]]
    for row in engine["decisions"]:
        batch = engine["batches"][row["batch_id"] - 1]
        batch["has_candidates"] = True
        row["stage"] = batch["stage"]
        for name in ("candidates", "ordered"):
            row[name] = [renumber(rid) for rid in row[name]]
    engine["evictions"].append(
        {
            "allocation_id": 2,
            "batch_id": 2,
            "request_id": "request-000000",
            "block_id": 3,
            "stage": "measure",
            "at_ns": 119,
            "reusable": True,
            "cache_key": "a" * 64,
            "removed_cache_keys": ["a" * 64],
        }
    )
    put(tmp_path / "client.json", client)
    put(tmp_path / "engine.json", engine)
    return cfg, inputs, client, engine


def test_v2_subprocess_validates_all_cold_and_compares_fixed_window(tmp_path):
    cfg, inputs, client, _ = setup_capture_v2(tmp_path)
    result = call_run(tmp_path, cfg, inputs)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["measurement_schema_version"] == 2
    metrics = report["metrics"]
    assert metrics["all_cold_zero_hit"] is True
    assert metrics["pressure"]["cached_evictions"] == 1
    assert metrics["schedule_envelope_ns"]["n"] == 0
    assert metrics["schedule_timing_scope"] == {"schedule_ns": "inherited_schedule_call"}
    assert metrics["throughput"]["window_ns"] == 1002
    assert metrics["throughput"]["fixed_window_ns"] == 1003
    assert metrics["throughput"]["last_terminal_tokens_per_s"] == 3e9 / 1001
    values = comparable(report)
    assert values["token_throughput"] == 3e9 / 1003
    assert values["request_throughput"] == 1e9 / 1003
    assert values["schedule_p95_ns"] == 2

    client["phases"]["drain_end_ns"] += 1
    put(tmp_path / "client.json", client)
    later = call_run(tmp_path, cfg, inputs)
    assert later.returncode == 0, later.stderr
    later_report = json.loads(later.stdout)
    assert (
        later_report["metrics"]["throughput"]["tokens_per_s"]
        != metrics["throughput"]["tokens_per_s"]
    )
    assert comparable(later_report) == values


def test_v2_subprocess_profile_separates_timing_scopes(tmp_path):
    cfg, inputs, _, _ = setup_capture_v2(tmp_path, tier="profile")
    result = call_run(tmp_path, cfg, inputs)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)["metrics"]
    assert metrics["schedule_ns"]["p95"] == 2
    assert metrics["schedule_envelope_ns"]["p95"] == 4
    assert (
        metrics["schedule_timing_scope"]["schedule_envelope_ns"] == "observer_inclusive_scheduler"
    )


def test_v2_subprocess_endpoint_marks_internal_evidence_unobserved(tmp_path):
    cfg, inputs, client, engine = setup_capture_v2(tmp_path, tier="endpoint")
    for name in ("enqueues", "batches", "decisions", "overhead", "evictions", "allocations"):
        engine[name].clear()
    put(tmp_path / "client.json", client)
    put(tmp_path / "engine.json", engine)
    result = call_run(tmp_path, cfg, inputs)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)["metrics"]
    assert metrics["work"] is None
    assert metrics["pressure"]["cached_evictions"] is None
    assert metrics["all_cold_zero_hit"] is None
    assert metrics["strata"]["cold"]["window_hot_overtakes"]["n"] == 0
    assert metrics["schedule_timing_scope"] is None


def test_v2_reusable_eviction_matches_secondary_native_cache_key_once(tmp_path):
    cfg, inputs, client, engine = setup_capture_v2(tmp_path)
    config = json.loads(cfg.read_text())
    plan = {row["request_id"]: row for row in json.loads(inputs.read_text())["requests"]}
    enqueues = {row["request_id"]: row for row in engine["enqueues"]}
    requests = {row["request_id"]: row for row in client["requests"]}
    first = {"request-000000": (120, 2), "request-000001": (150, 4)}
    plan["request-000001"]["class"] = "hot"
    enqueues["request-000001"]["block_hashes"] = ["b" * 64]
    engine["evictions"][0]["removed_cache_keys"].append("b" * 64)
    pressure, _ = pressure_metrics(engine, config, plan, enqueues, first, requests, 1104, 1)
    assert pressure["reusable_evictions"] == pressure["cached_evictions"] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_decision", "candidate decision missing"),
        ("bad_marker", "candidate decision marker"),
        ("decision_stage", "does not match the marked batch"),
        ("missing_timing", "observer-inclusive schedule timing missing"),
        ("unexpected_timing", "outside profile"),
        ("missing_profile_batch", "stage timing batch missing"),
        ("missing_eviction", "eviction count differs"),
        ("duplicate_eviction", "duplicate physical eviction"),
        ("wrong_eviction_id", "does not match its successful allocation"),
        ("eviction_capacity", "does not match its successful allocation or physical capacity"),
        ("eviction_alias", "native removed cache keys missing or invalid"),
        ("eviction_reorder", "eviction journal or allocation timestamp order differs"),
        ("stale_eviction", "eviction journal or allocation timestamp order differs"),
        ("missing_cache", "missing client cache observation"),
        ("wrong_native_hit", "lacks zero-hit native allocation"),
        ("wrong_native_start", "lacks zero-hit native allocation and runner evidence"),
        ("wrong_input_length", "frozen prompt length differs"),
        ("wrong_input_class", "hot/cold class differs"),
        ("old_schema", "task or trace schema differs"),
    ],
)
def test_v2_semantic_fail_closed_controls(tmp_path, mutation, reason):
    tier = "profile" if mutation in {"missing_timing", "missing_profile_batch"} else "light"
    cfg, inputs, client, engine = setup_capture_v2(tmp_path, tier=tier)
    if mutation == "missing_decision":
        engine["decisions"].pop()
    elif mutation == "bad_marker":
        engine["batches"][1]["has_candidates"] = 1
    elif mutation == "decision_stage":
        engine["decisions"][1]["stage"] = "warmup"
    elif mutation == "missing_timing":
        engine["batches"][1].pop("schedule_envelope_ns")
    elif mutation == "unexpected_timing":
        engine["batches"][1]["schedule_envelope_ns"] = 4
    elif mutation == "missing_profile_batch":
        engine["overhead"].pop()
    elif mutation == "missing_eviction":
        engine["evictions"].clear()
    elif mutation == "duplicate_eviction":
        engine["evictions"].append(engine["evictions"][0].copy())
        engine["allocations"][1]["eviction_count"] += 1
    elif mutation == "wrong_eviction_id":
        engine["evictions"][0]["allocation_id"] = 3
    elif mutation == "eviction_capacity":
        engine["evictions"][0]["block_id"] = engine["profile"]["num_gpu_blocks"]
    elif mutation == "eviction_alias":
        engine["evictions"][0]["removed_cache_keys"] = ["a" * 64, "a" * 64]
    elif mutation == "eviction_reorder":
        later = {**engine["evictions"][0], "block_id": 4, "at_ns": 120}
        engine["evictions"].insert(0, later)
        engine["allocations"][1]["eviction_count"] = 2
    elif mutation == "stale_eviction":
        engine["evictions"][0].update(allocation_id=3, batch_id=3)
        engine["allocations"][1]["eviction_count"] = 0
        engine["allocations"][2]["eviction_count"] = 1
    elif mutation == "missing_cache":
        # The timed-out row still supplies one zero, enough for the old aggregate gate.
        client["requests"][0]["cached_tokens"] = None
    elif mutation == "wrong_native_hit":
        engine["allocations"][1]["hit_tokens"] = 2
    elif mutation == "wrong_native_start":
        batch = engine["batches"][1]
        batch["scheduled"][0].update(start=1, end=4)
        batch["runner"]["rows"][0].update(start=1, end=4)
    elif mutation in {"wrong_input_length", "wrong_input_class"}:
        workload = json.loads(inputs.read_text())
        row = workload["requests"][0]
        if mutation == "wrong_input_length":
            row["prompt_token_ids"].append(3)
        else:
            row.update(**{"class": "hot", "family": "F"})
        put(inputs, workload)
        client["input_sha256"] = sha256(inputs)
    elif mutation == "old_schema":
        engine["schema_version"] = client["schema_version"] = 1
    put(tmp_path / "client.json", client)
    put(tmp_path / "engine.json", engine)
    result = call_run(tmp_path, cfg, inputs)
    assert result.returncode != 0 and reason in result.stderr, result.stderr
    assert result.stdout == ""


def test_real_auditor_subprocess_reconstructs_requests_batches_and_denominators(tmp_path):
    cfg, inputs, _, _ = setup_capture(tmp_path)
    result = call_run(tmp_path, cfg, inputs)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "valid"
    metrics = report["metrics"]
    all_rows = metrics["strata"]["all"]
    assert all_rows["counts"] == {
        "planned": 4,
        "submitted": 4,
        "completed": 1,
        "failed": 1,
        "cancelled": 0,
        "timeout": 1,
        "unfinished": 1,
        "timely_first_token": 2,
    }
    assert all_rows["completion_rate"] == 0.25
    assert all_rows["timely_first_token_rate"] == 0.5
    assert all_rows["ttft_ns"]["p95"] == 40
    assert all_rows["ttft_ns"]["n"] == 1
    assert all_rows["tpot_ns"]["p95"] == 4
    assert metrics["strata"]["cold"]["ttft_ns"]["p95"] is None
    assert metrics["work"] == {
        "prefill_tokens": 8,
        "decode_tokens": 1,
        "recomputed_prefill_tokens": 2,
        "effective_tokens": 9,
        "padded_tokens": 12,
    }
    assert metrics["throughput"]["generated_tokens"] == 3
    assert metrics["max_outstanding"] == 4


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("wrong_mode", "mode label"),
        ("missing_request", "denominator"),
        ("duplicate_request", "duplicate/unknown"),
        ("missing_cached_tokens", "required field"),
        ("wrong_tokens", "times or IDs"),
        ("wrong_clock", "clock domain"),
        ("late_submit", "submission lag"),
        ("missing_runner", "successful runner"),
        ("fake_positions", "runner positions"),
        ("no_materialization", "materialization proof"),
        ("wrong_source", "source hashes"),
        ("wrong_native_prompt", "actual native prompt"),
        ("wrong_native_output", "consumed native outputs"),
        ("startup_residue", "native scheduler residual"),
        ("pending_batch_residue", "native scheduler residual"),
        ("missing_boundary", "drain boundary missing"),
    ],
)
def test_real_auditor_subprocess_fails_closed(tmp_path, mutation, reason):
    cfg, inputs, client, engine = setup_capture(tmp_path)
    if mutation == "wrong_mode":
        client["mode"] = "off"
    elif mutation == "missing_request":
        client["requests"].pop()
    elif mutation == "duplicate_request":
        client["requests"][3] = client["requests"][2].copy()
    elif mutation == "missing_cached_tokens":
        client["requests"][0].pop("cached_tokens")
    elif mutation == "wrong_tokens":
        client["requests"][0]["chunk_tokens"][0]["token_ids"] = [99]
    elif mutation == "wrong_clock":
        engine["clock"]["time_namespace"] = "other"
    elif mutation == "late_submit":
        client["requests"][3]["submit_ns"] += 30
        engine["enqueues"][4]["enqueued_ns"] += 30
    elif mutation == "missing_runner":
        engine["batches"][0].pop("runner")
    elif mutation == "fake_positions":
        engine["batches"][0]["runner"]["rows"][0]["start"] = 1
    elif mutation == "no_materialization":
        engine["batches"][0]["materialized"] = False
    elif mutation == "wrong_source":
        client["input_sha256"] = "0" * 64
    elif mutation == "wrong_native_prompt":
        engine["enqueues"][1]["prompt_sha256"] = "0" * 64
    elif mutation == "wrong_native_output":
        engine["batches"][1]["outputs"][0]["token_ids"] = [99]
    elif mutation == "startup_residue":
        engine["boundaries"][0]["non_null_ref_sum"] = 1
    elif mutation == "pending_batch_residue":
        engine["boundaries"][0]["pending_batches"] = [1]
    elif mutation == "missing_boundary":
        engine["boundaries"].pop()
    put(tmp_path / "client.json", client)
    put(tmp_path / "engine.json", engine)
    result = call_run(tmp_path, cfg, inputs)
    assert result.returncode != 0 and reason in result.stderr
    assert result.stdout == ""


def test_matrix_subprocess_rejects_missing_group(tmp_path):
    _, inputs, _, _ = setup_capture(tmp_path)
    cfg = ROOT / "configs" / "m5-performance.json"
    index = put(
        tmp_path / "index.json",
        {
            "runs": [
                {
                    "pair_id": "mixed-01",
                    "mode": "native",
                    "run": str(tmp_path),
                    "inputs": str(inputs),
                }
            ],
        },
    )
    result = subprocess.run(
        [sys.executable, str(CLI), "matrix", "--config", str(cfg), "--index", str(index)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0 and "formal matrix missing" in result.stderr


def test_interval_overlap_uses_union_and_quantiles_are_exact():
    previous = []
    assert add_interval(previous, 2, 5) == 0
    assert add_interval(previous, 7, 10) == 0
    assert add_interval(previous, 1, 8) == 4
    assert previous == [(1, 10)]
    assert add_interval(previous, 4, 9) == 5
    assert quantile([], 95) is None
    assert quantile(list(range(1, 21)), 95) == 19
    assert paired_summary([{"metric": -2}, {"metric": 0}, {"metric": None}, {"metric": 3}]) == {
        "metric": {
            "n": 3,
            "median": 0,
            "min": -2,
            "max": 3,
            "positive": 1,
            "negative": 1,
            "equal": 1,
        },
    }


def test_calibration_and_formal_are_separate(tmp_path):
    cfg, inputs, _, _ = setup_capture(tmp_path)
    audit_run(cfg, inputs, tmp_path, phase="calibration")
    with pytest.raises(ValueError, match="run phase differs"):
        audit_run(cfg, inputs, tmp_path, phase="formal")
    with pytest.raises(ValueError, match="formal matrix missing"):
        audit_matrix(ROOT / "configs" / "m5-performance.json", {"runs": []})
