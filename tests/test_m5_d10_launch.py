"""Real subprocess controls for D10's exclusive allocation and frozen intervention."""

import json

import pytest
from test_m5_d4_launch import COMMIT, CONFIG, ROOT, approval, authorize, child, put, read, sha

PLAN = ROOT / "configs/m5-d10.json"
PARENT = ROOT / "configs/m5-d8.json"
D2R110 = ROOT / "configs/m5-d2r110.json"
READ_PLAN = """
import json
from pathlib import Path
from vllm_prefix_scheduler.m5_d10 import d10_plan, d10_specs
try:
    plan = d10_plan(*map(Path, sys.argv[1:]))
    print(json.dumps(d10_specs([dict(id=identity) for identity in plan["runs"]], plan)))
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
"""


def d10_approval():
    record = approval()
    del record["d4_plan_sha256"]
    return {
        **record,
        "roles": ["D10"],
        "max_attempts": 18,
        "max_requested_minutes": 360,
        "d10_plan_sha256": sha(PLAN),
    }


def test_d10_baab_inherits_exact_baseline(tmp_path):
    result = child(READ_PLAN, PLAN, PARENT, D2R110)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"id": "thermal-pilot", "d10_placement": "inherited"},
        {"id": "E1", "d10_placement": "split"},
        {"id": "E2", "d10_placement": "inherited"},
        {"id": "E3", "d10_placement": "inherited"},
        {"id": "E4", "d10_placement": "split"},
    ]
    plan = read(PLAN)
    plan["placement_order"][0] = "inherited"
    changed_plan = put(tmp_path / PLAN.name, plan)
    rejected = child(READ_PLAN, changed_plan, PARENT, D2R110)
    assert rejected.returncode == 1 and "reviewed plan differs" in rejected.stderr
    changed = read(PARENT)
    changed["stability_gate"]["ttft_p95_relative_margin"] = 0.06
    rejected = child(READ_PLAN, PLAN, put(tmp_path / PARENT.name, changed), D2R110)
    assert rejected.returncode == 1 and "inherited D8" in rejected.stderr


@pytest.mark.parametrize(
    "overrides",
    (
        {"roles": ["D10", "C1"]},
        {"max_attempts": 17, "max_requested_minutes": 340},
        {"max_attempts": 19, "max_requested_minutes": 380},
        {"d10_plan_sha256": "f" * 64},
    ),
)
def test_d10_single_allocation_authorization(tmp_path, overrides):
    path = tmp_path / "auth.json"
    result = authorize(put(path, d10_approval()))
    assert result.returncode == 0 and json.loads(result.stdout) == ["D10"]
    result = authorize(put(path, {**d10_approval(), **overrides}))
    assert result.returncode == 1 and "D10" in result.stderr


def test_d10_intent_latches_retry_but_not_d9_history(tmp_path):
    code = """
from pathlib import Path
from submit_m5 import d9_no_prior_intent
try:
    d9_no_prior_intent(Path(sys.argv[1]), "D10")
except ValueError as error:
    sys.exit(str(error))
"""
    prior = tmp_path / "d9-01"
    current = tmp_path / "d10-01"
    put(prior / "planned.json", {"resource_schema_version": 2, "role": "D9"})
    put(prior / "submission-intent.json", {"argv": ["sbatch"]})
    put(current / "planned.json", {"resource_schema_version": 2, "role": "D10"})
    assert child(code, tmp_path).returncode == 0
    put(current / "submission-intent.json", {"argv": ["sbatch"]})
    result = child(code, tmp_path)
    assert result.returncode == 1 and "no retry allowance" in result.stderr


def test_d10_cpu_receipt_checks_frozen_inputs_and_placement(tmp_path):
    code = """
from pathlib import Path
from submit_m5 import d10_submission_preflight
review, proposal = map(Path, sys.argv[1:3])
try:
    print(d10_submission_preflight(review, sys.argv[3], {"proposed-d10.json": proposal}, []))
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
"""
    pins = {
        **read(CONFIG.with_name("native-m2b.json"))["source_sha256"],
        **read(CONFIG)["extra_upstream_sha256"],
    }
    report = {
        "execution_commit": COMMIT,
        "gpu_allocations": 0,
        "verified_inputs_sha256": {"proposed-d10.json": sha(PLAN)},
        "source": {
            "verified_count": len(pins),
            "python": [3, 12, 16],
            "vllm": "0.32.0",
            "clock": "clock_gettime(CLOCK_MONOTONIC)",
        },
        "runs": [],
        "placement_cpu_verified": True,
    }
    path = put(tmp_path / "login-stage-preflight.json", report)
    result = child(code, tmp_path, PLAN, COMMIT)
    assert result.returncode == 0 and result.stdout.strip() == sha(path)
    report["placement_cpu_verified"] = False
    put(path, report)
    rejected = child(code, tmp_path, PLAN, COMMIT)
    assert rejected.returncode == 1 and "placement proof" in rejected.stderr
