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
    paired_summary,
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
