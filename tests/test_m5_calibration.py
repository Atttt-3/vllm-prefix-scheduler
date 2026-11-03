"""Local subprocess controls for the calibration plan and decision gates."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plan_m5_calibration.py"
CONFIG = ROOT / "configs/m5-performance.json"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data))
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(action, stage, output, *options):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            action,
            "--config",
            str(CONFIG),
            "--stage",
            stage,
            "--output",
            str(output),
            *map(str, options),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )


def rates_file(root):
    return write(
        root / "rates.json",
        {
            "stage": "C2",
            "status": "qualified",
            "selected_rates": {"burst": 3, "mixed": 4},
            "config_sha256": digest(CONFIG),
        },
    )


def stage_plan(root, stage, *options):
    out = root / f"plan-{stage}.json"
    result = cli("plan", stage, out, *options)
    assert result.returncode == 0, result.stderr
    return read(out)


def evidence_index(root, plan):
    rows = []
    for spec in plan["runs"]:
        leaf = root / spec["id"]
        leaf.mkdir()
        inputs = write(
            leaf / "inputs.json",
            {
                **{
                    key: spec[key]
                    for key in ("scenario", "seed", "rate_rps", "interval_ms", "mechanism")
                },
                "requests": [{}] * spec["count"],
            },
        )
        audit = write(
            leaf / "audit.json",
            {
                **{key: spec[key] for key in ("scenario", "seed", "mode", "tier")},
                "pair_id": spec["id"],
                "phase": "diagnostic" if spec["scenario"] == "mechanism" else "calibration",
                "status": "valid",
                "config_sha256": digest(CONFIG),
                "input_sha256": digest(inputs),
                "metrics": {
                    "duration_ns": {
                        "load": 1,
                        "warmup": 1,
                        "measure_through_drain": 1,
                        "max_submit_to_end": 161_000_000,
                    },
                    "strata": {
                        "all": {
                            "counts": {"planned": spec["count"], "completed": spec["count"]},
                            "ttft_ns": {"p95": 10_000_000},
                            "completion_rate": 1.0,
                            "timely_first_token_rate": 1.0,
                        }
                    },
                    "pressure": {"competing_windows": 22, "reusable_evictions": 6},
                    "throughput": {
                        "last_terminal_requests_per_s": 4.0,
                        "last_terminal_tokens_per_s": 64.0,
                    },
                    "stage_costs_ns": {
                        key: {"n": 5}
                        for key in ("schedule", "window", "score", "sort_exclusive", "queue_commit")
                    },
                    "max_outstanding": 1,
                    "submit_lag_ns": {"n": spec["count"], "p99": 1000, "max": 1000},
                },
            },
        )
        rows.append(
            {
                "id": spec["id"],
                "inputs": str(inputs),
                "run": str(leaf),
                "audit": str(audit),
                "elapsed_seconds": 42,
            }
        )
    return write(root / "index.json", {"schema_version": 1, "stage": plan["stage"], "runs": rows})


def test_calibration_inventory_and_budget(tmp_path):
    rates = rates_file(tmp_path)
    expected = {"C1": 8, "C2": 10, "C3": 9, "C4": 4, "C5": 4, "C6": 3}
    for stage, count in expected.items():
        plan = stage_plan(tmp_path, stage, *(["--rates", rates] if stage >= "C3" else []))
        assert len(plan["runs"]) == count
        assert plan["allocation_minutes"] == 20
        assert plan["estimated_mode_work_seconds"] + plan["allocation_reserve_seconds"] <= 1050
        assert all(51_000 <= row["seed"] < 52_000 for row in plan["runs"])
    output = tmp_path / "serial-followup.json"
    result = cli("plan", "C6", output, "--rates", rates, "--serial-interval-ms", 230)
    assert result.returncode == 0, result.stderr
    assert [(row["id"], row["count"], row["interval_ms"]) for row in read(output)["runs"]] == [
        ("serial-full", 500, 230)
    ]


def test_native_calibration_selects_lowest_eligible_rate(tmp_path):
    index = evidence_index(tmp_path, stage_plan(tmp_path, "C2"))
    no_contention = tmp_path / "mixed-3rps/audit.json"
    data = read(no_contention)
    data["metrics"]["pressure"]["competing_windows"] = 0
    write(no_contention, data)
    missed_deadline = tmp_path / "mixed-4rps/audit.json"
    data = read(missed_deadline)
    data["metrics"]["strata"]["all"]["completion_rate"] = 0.98
    write(missed_deadline, data)
    missed_submit = tmp_path / "burst-3rps/audit.json"
    data = read(missed_submit)
    data["metrics"]["submit_lag_ns"]["max"] = 25_000_001
    write(missed_submit, data)

    output = tmp_path / "selection.json"
    result = cli("evaluate", "C2", output, "--index", index)
    assert result.returncode == 0, result.stderr
    assert read(output)["selected_rates"] == {"burst": 4, "mixed": 5}
    assert len(read(output)["pilots"]) == 10


@pytest.mark.parametrize("change", ["missing-run", "input-rate", "input-hash", "run-id"])
def test_invalid_rate_evidence_fails_closed(tmp_path, change):
    index = evidence_index(tmp_path, stage_plan(tmp_path, "C2"))
    original = read(index)
    row = original["runs"][0]
    audit = Path(row["audit"])
    inputs = Path(row["inputs"])
    if change == "missing-run":
        original["runs"].pop()
        write(index, original)
    elif change == "input-rate":
        write(inputs, {**read(inputs), "rate_rps": 8})
        write(audit, {**read(audit), "input_sha256": digest(inputs)})
    elif change == "input-hash":
        write(audit, {**read(audit), "input_sha256": "f" * 64})
    else:
        write(audit, {**read(audit), "pair_id": "other-pilot"})
    output = tmp_path / "blocked.json"
    result = cli("evaluate", "C2", output, "--index", index)
    assert result.returncode == 2
    assert "evidence rejected" in result.stderr
    assert not output.exists()


def test_overhead_gate_uses_last_terminal_throughput(tmp_path):
    rates = rates_file(tmp_path)
    index = evidence_index(tmp_path, stage_plan(tmp_path, "C3", "--rates", rates))
    valid = tmp_path / "qualified.json"
    result = cli("evaluate", "C3", valid, "--rates", rates, "--index", index)
    assert result.returncode == 0, result.stderr
    assert all(row["qualified"] for row in read(valid)["comparisons"])

    changed = tmp_path / "burst-2-light/audit.json"
    data = read(changed)
    data["metrics"]["throughput"]["last_terminal_tokens_per_s"] = 57.0
    write(changed, data)
    blocked = tmp_path / "blocked.json"
    result = cli("evaluate", "C3", blocked, "--rates", rates, "--index", index)
    assert result.returncode == 2
    assert read(blocked)["status"] == "blocked"
    assert "paired tolerance" in read(blocked)["reasons"][0]


def diagnostic_evidence(root):
    plan = stage_plan(root, "C1")
    index = evidence_index(root, plan)
    for spec, row in zip(plan["runs"], read(index)["runs"], strict=True):
        profile, mode = spec["mechanism"], spec["mode"]
        native = mode in {"native", "off"}
        victim = native and profile == "pressure"
        hit = 0 if victim else 1024
        prefix = list(range(64))
        inputs_path, audit_path, run = (Path(row[key]) for key in ("inputs", "audit", "run"))
        inputs = read(inputs_path)
        lengths = {"prime-A": 1025, "B": 1537, "C": 1025}
        requests = [
            {"request_id": rid, "prompt_token_ids": [1] * length} for rid, length in lengths.items()
        ]
        write(inputs_path, {**inputs, "warmup": requests[:1], "requests": requests[1:]})
        order = ["B", "C"] if native else ["C", "B"]
        batches = ["prime-A", *order]
        client = write(
            run / "client.json",
            {
                "warmup": [{"request_id": "prime-A", "status": "completed", "token_ids": [1]}],
                "requests": [
                    {"request_id": "B", "status": "completed", "cached_tokens": 0},
                    {"request_id": "C", "status": "completed", "cached_tokens": hit},
                ],
            },
        )
        engine = write(
            run / "engine.json",
            {
                "profile": {"num_gpu_blocks": 129 if profile == "pressure" else 257},
                "probes": [
                    {"request_id": "B", "at_ns": 2, "hit_tokens": 0, "block_ids": []},
                    {"request_id": "C", "at_ns": 3, "hit_tokens": 1024, "block_ids": prefix},
                ],
                "decisions": [
                    {
                        "batch_id": 2,
                        "decision_ns": 4,
                        "candidates": ["B", "C"],
                        "ordered": order,
                        "protected": [],
                    }
                ],
                "batches": [
                    {
                        "batch_id": i,
                        "materialized": True,
                        "consumed": True,
                        "runner": {
                            "rows": [
                                {
                                    "request_id": rid,
                                    "generation": 1,
                                    "position_source": "runner_positions_d2h",
                                    "start": hit if rid == "C" else 0,
                                    "end": lengths[rid],
                                }
                            ]
                        },
                    }
                    for i, rid in enumerate(batches, 1)
                ],
                "allocations": [
                    {
                        "batch_id": batches.index(rid) + 1,
                        "request_id": rid,
                        "generation": 1,
                        "hit_tokens": hit if rid == "C" else 0,
                        "success": True,
                    }
                    for rid in ("B", "C")
                ],
                "evictions": [{"request_id": "B", "batch_id": 2, "block_id": 0}] if victim else [],
            },
        )
        write(
            audit_path,
            {
                **read(audit_path),
                "input_sha256": digest(inputs_path),
                "client_sha256": digest(client),
                "engine_sha256": digest(engine),
            },
        )
    return index


def test_mechanism_requires_actual_bound_causal_chain(tmp_path):
    index = diagnostic_evidence(tmp_path)
    accepted = tmp_path / "mechanism.json"
    result = cli("evaluate", "C1", accepted, "--index", index)
    assert result.returncode == 0, result.stderr
    pressure = {row["id"]: row for row in read(accepted)["mechanism"]}
    assert pressure["mechanism-pressure-native"]["C_prefill_tokens"] == 1025
    assert pressure["mechanism-pressure-prefix"]["C_prefill_tokens"] == 1

    target = tmp_path / "mechanism-pressure-native"
    engine, audit = (target / f"{name}.json" for name in ("engine", "audit"))
    write(engine, {**read(engine), "evictions": []})
    write(audit, {**read(audit), "engine_sha256": digest(engine)})
    result = cli("evaluate", "C1", tmp_path / "rejected.json", "--index", index)
    assert result.returncode == 2
    assert "expected C-prefix eviction" in result.stderr


def test_control_calibration_rederives_serial_spacing_and_budgets(tmp_path):
    rates = rates_file(tmp_path)
    plan = stage_plan(tmp_path, "C6", "--rates", rates)
    followup = tmp_path / "serial-followup.json"
    result = cli("plan", "C6", followup, "--rates", rates, "--serial-interval-ms", 230)
    assert result.returncode == 0, result.stderr
    plan["runs"].extend(read(followup)["runs"])
    index = evidence_index(tmp_path, plan)
    for row in read(index)["runs"]:
        if row["id"] not in {"ample-full", "all-cold-full"}:
            continue
        audit_path = Path(row["audit"])
        audit = read(audit_path)
        audit["metrics"]["strata"]["all"]["cached_tokens"] = {"n": 500, "max": 0}
        if row["id"] == "ample-full":
            inputs_path = Path(row["inputs"])
            inputs = {
                **read(inputs_path),
                "sampling": {"max_tokens": 16},
                "warmup": [],
                "requests": [{"prompt_token_ids": [1] * 129}] * 500,
            }
            write(inputs_path, inputs)
            audit["input_sha256"] = digest(inputs_path)
            audit["metrics"]["pressure"]["reusable_evictions"] = 0
            audit["metrics"]["kv_blocks"] = 5553
        write(audit_path, audit)
    dress_paths = [
        write(
            tmp_path / f"{stage}.json",
            {
                "stage": stage,
                "status": "qualified",
                "config_sha256": digest(CONFIG),
                "mode_wall_seconds": {"native": 100, "off": 105, "prefix": 114, "guarded": 112},
            },
        )
        for stage in ("C4", "C5")
    ]
    args = [
        "--rates",
        rates,
        "--index",
        index,
        "--dress-report",
        dress_paths[0],
        "--dress-report",
        dress_paths[1],
    ]
    accepted = tmp_path / "controls.json"
    result = cli("evaluate", "C6", accepted, *args)
    assert result.returncode == 0, result.stderr
    report = read(accepted)
    assert report["selected_rates"] == {"ample": 4, "all-cold": 4}
    assert report["serial_interval_ms"] == 230
    assert report["control_projection_factor"] == 1.14
    assert report["ample_capacity_upper_bound_blocks"] == 5001

    pilot = tmp_path / "serial-pilot/audit.json"
    data = read(pilot)
    data["metrics"]["max_outstanding"] = 2
    write(pilot, data)
    result = cli("evaluate", "C6", tmp_path / "rejected.json", *args)
    assert result.returncode == 2
    assert "serial pilot incomplete or overlapping" in result.stderr
