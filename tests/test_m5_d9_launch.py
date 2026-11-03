"""Real child-process controls for the D9 frozen stage and exclusive authority."""

import json
import shutil

import pytest
from test_m5_d4_launch import COMMIT, CONFIG, ROOT, approval, authorize, child, put, read, sha

PLAN = ROOT / "configs/m5-d9.json"
PARENT = ROOT / "configs/m5-d8.json"
D2R110 = ROOT / "configs/m5-d2r110.json"
FROZEN_PLAN = """
import json
from pathlib import Path
from vllm_prefix_scheduler.m5_d9 import d9_plan
try:
    print(json.dumps(d9_plan(*map(Path, sys.argv[1:]))))
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
"""


def d9_approval():
    record = approval()
    del record["d4_plan_sha256"]
    return {
        **record,
        "user_approval_ref": "local synthetic D9 authority fixture",
        "roles": ["D9"],
        "max_attempts": 17,
        "max_requested_minutes": 340,
        "d9_plan_sha256": sha(PLAN),
    }


def test_d9_frozen_plan_inherits_original_abba_condition(tmp_path):
    result = child(FROZEN_PLAN, PLAN, PARENT, D2R110)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["runs"] == ["thermal-pilot", "E1", "E2", "E3", "E4"]
    assert plan["capture"] == ["pilot", "endpoint", "step", "step", "endpoint"]
    assert plan["trace"]["windows_ns"] == [[0, 3_000_000_000], [6_000_000_000, 8_500_000_000]]
    assert plan["target"]["arrival_rate_rps"] == 8
    assert plan["target"]["count_per_engine"] == 60
    assert plan["profile"] == {
        "max_num_seqs": 4,
        "max_num_active_seqs": None,
        "num_gpu_blocks_override": None,
        "num_gpu_blocks": 5553,
        "max_num_batched_tokens": 2048,
    }
    assert plan["retained_C3"] == read(PARENT)["retained_C3"]
    assert plan["can_qualify_C3"] is False
    parent = read(PARENT)
    parent["conditioning"]["per_measured_engine"]["preheat_temp_median_from_target_c"] = 3
    rejected = child(FROZEN_PLAN, PLAN, put(tmp_path / PARENT.name, parent), D2R110)
    assert rejected.returncode == 1 and "inherited D8 plan differs" in rejected.stderr


@pytest.mark.parametrize(
    "overrides",
    (
        {"roles": ["D9", "C1"]},
        {"max_attempts": 16, "max_requested_minutes": 320},
        {"max_attempts": 18, "max_requested_minutes": 360},
        {"d9_plan_sha256": "f" * 64},
    ),
)
def test_d9_one_slot_authority(tmp_path, overrides):
    path = tmp_path / "authorization.json"
    allowed = authorize(put(path, d9_approval()))
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout) == ["D9"]
    rejected = authorize(put(path, {**d9_approval(), **overrides}))
    assert rejected.returncode == 1 and "D9" in rejected.stderr


def test_d9_rejects_rehashed_plan_and_disallows_qualification_substitution(tmp_path):
    config_path = tmp_path / CONFIG.name
    shutil.copyfile(CONFIG, config_path)
    plan = read(PLAN)
    plan["trace"]["windows_ns"][1][0] = 5_000_000_000
    proposal = put(tmp_path / PLAN.name, plan)
    authorization_path = tmp_path / "authorization.json"
    auth = {**d9_approval(), "d9_plan_sha256": sha(proposal)}
    rejected = authorize(put(authorization_path, auth), config_path)
    assert (
        rejected.returncode == 1 and "D9 authorization differs from the freeze" in rejected.stderr
    )
    rejected = child(FROZEN_PLAN, proposal, PARENT, D2R110)
    assert rejected.returncode == 1 and "D9 reviewed plan differs" in rejected.stderr

    config = read(config_path)
    config["calibration"]["qualification_predecessor_roles"] = ["D9"]
    put(config_path, config)
    del auth["d9_plan_sha256"]
    auth.update(
        config_sha256=sha(config_path),
        roles=["C1", "C2", "C3-burst", "C3-mixed", "C4", "C5", "C6"],
        max_attempts=24,
        max_requested_minutes=480,
        qualification_batch={
            "schema_version": 2,
            "prior_allocation_count": 17,
            "prior_history_sha256": "a" * 64,
            "new_max_attempts": 7,
            "predecessor": {
                "protocol": "same-condition-ee-v1",
                "role": "D9",
                "attempt": "d9-01",
                "execution_commit": COMMIT,
                "proposal_file": "m5-d9.json",
                "proposal_sha256": sha(PLAN),
                "review_directory": "d9-review",
                "analysis_sha256": "b" * 64,
            },
        },
    )
    rejected = authorize(put(authorization_path, auth), config_path)
    assert rejected.returncode == 1 and "approved same-condition predecessor" in rejected.stderr


def test_d9_previous_submission_intent_consumes_sole_attempt(tmp_path):
    code = """
from pathlib import Path
from submit_m5 import d9_no_prior_intent
try:
    d9_no_prior_intent(Path(sys.argv[1]))
except ValueError as error:
    sys.exit(str(error))
"""
    attempt = tmp_path / "d9-01"
    put(attempt / "planned.json", {"resource_schema_version": 2, "role": "D9"})
    assert child(code, tmp_path).returncode == 0
    put(attempt / "submission-intent.json", {"argv": ["sbatch"]})
    result = child(code, tmp_path)
    assert result.returncode == 1 and "has no retry allowance" in result.stderr


def test_d9_controller_requires_linux_cpu_receipt_and_margin(tmp_path):
    code = """
from pathlib import Path
from submit_m5 import d9_submission_preflight
from vllm_prefix_scheduler.m5_campaign import digest, read_json
review, proposal, table = map(Path, sys.argv[1:4])
plan = read_json(proposal)
plan["target"]["input_sha256"] = digest(table)
try:
    print(d9_submission_preflight(
        review, sys.argv[4], {"proposed-d9.json": proposal, "input.json": table}, [], plan
    ))
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
"""
    pins = {
        **read(CONFIG.with_name("native-m2b.json"))["source_sha256"],
        **read(CONFIG)["extra_upstream_sha256"],
    }
    last_arrival = 7_375_000_000
    fixed = 3_250_000_000
    table = put(tmp_path / "input.json", {"requests": [{"offset_ns": last_arrival}]})

    def samples(incremental):
        native = {"native_path_ns": 5_000_000, "export_and_endpoint_copies_ns": 0}
        observed = {**native, "export_and_endpoint_copies_ns": incremental // 2}
        return [
            {
                "width": width,
                "legs": [native, observed, observed, native],
                "incremental_ns": incremental,
            }
            for width in (1, 4)
        ]

    benchmark = {
        "python": [3, 12, 16],
        "source_sha256": pins,
        "msgpack_round_trip": True,
        "plan_sha256": sha(PLAN),
        "input_sha256": sha(table),
        "last_arrival_ns": last_arrival,
        "fixed_lead_and_quiet_bound_ns": fixed,
        "maximum_two_b_incremental_ns": 350_000_000,
        "margin_ns": 5_000_000_000,
        "total_added_upper_envelope_ns": 3_600_000_000,
        "samples": samples(350_000_000),
    }
    benchmark_path = put(tmp_path / "linux-cpu-benchmark.json", benchmark)
    report = {
        "execution_commit": COMMIT,
        "gpu_allocations": 0,
        "verified_inputs_sha256": {"proposed-d9.json": sha(PLAN), "input.json": sha(table)},
        "source": {
            "verified_count": len(pins),
            "python": [3, 12, 16],
            "vllm": "0.32.0",
            "clock": "clock_gettime(CLOCK_MONOTONIC)",
        },
        "runs": [],
        "dual_window_extra_seconds": 3.6,
        "fixed_lead_and_quiet_bound_ns": fixed,
        "cpu_benchmark_sha256": sha(benchmark_path),
    }
    report_path = put(tmp_path / "login-stage-preflight.json", report)
    result = child(code, tmp_path, PLAN, table, COMMIT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == sha(report_path)
    benchmark.update(
        total_added_upper_envelope_ns=5_000_000_002,
        maximum_two_b_incremental_ns=1_750_000_002,
        samples=samples(1_750_000_002),
    )
    put(benchmark_path, benchmark)
    report.update(dual_window_extra_seconds=5.000000002, cpu_benchmark_sha256=sha(benchmark_path))
    put(report_path, report)
    rejected = child(code, tmp_path, PLAN, table, COMMIT)
    assert rejected.returncode == 1 and "five-second work margin" in rejected.stderr

    benchmark["total_added_upper_envelope_ns"] = 3_600_000_000
    put(benchmark_path, benchmark)
    report.update(dual_window_extra_seconds=3.6, cpu_benchmark_sha256=sha(benchmark_path))
    put(report_path, report)
    rejected = child(code, tmp_path, PLAN, table, COMMIT)
    assert rejected.returncode == 1 and "five-second work margin" in rejected.stderr
