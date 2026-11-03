"""Subprocess controls for D7's frozen late window and one-slot authority."""

import json

import pytest
from test_m5_d4_launch import CONFIG, ROOT, authorize, child, put, read, sha
from test_m5_d5_launch import D3_PLAN, d5_approval
from test_m5_d5r_launch import D5_PLAN

D5R_PLAN = ROOT / "configs/m5-d5r.json"
D6_PLAN = ROOT / "configs/m5-d6.json"
PLAN = ROOT / "configs/m5-d7.json"


def d7_approval():
    record = d5_approval()
    del record["d5_plan_sha256"]
    record.update(
        user_approval_ref="local synthetic D7 authority fixture",
        roles=["D7"],
        max_attempts=15,
        max_requested_minutes=300,
        d7_plan_sha256=sha(PLAN),
    )
    return record


def frozen_plan(plan, inherited_d6=D6_PLAN, inherited_d5r=D5R_PLAN):
    return child(
        """import json
from pathlib import Path
from vllm_prefix_scheduler.m5_d7 import d7_plan
try:
    print(json.dumps(d7_plan(*map(Path, sys.argv[1:]))))
except ValueError as error:
    sys.exit(str(error))
""",
        plan,
        inherited_d6,
        inherited_d5r,
        D5_PLAN,
        D3_PLAN,
    )


def test_d7_child_freezes_late_window_abba_and_inherited_thermal_gates(tmp_path):
    result = frozen_plan(PLAN)
    assert result.returncode == 0, result.stderr
    plan, d5r, d5, d3 = json.loads(result.stdout)
    assert (
        plan == read(PLAN) and d5r == read(D5R_PLAN) and d5 == read(D5_PLAN) and d3 == read(D3_PLAN)
    )
    assert plan["capture"] == ["pilot", "endpoint", "step", "step", "endpoint"]
    trace, budget = plan["trace"], plan["budget"]
    assert (trace["start_offset_ns"], trace["start_offset_ns"] + trace["window_ns"]) == (
        3_500_000_000,
        7_500_000_000,
    )
    assert (budget["work_estimate_s"], budget["work_deadline_s"], budget["retries"]) == (
        1040,
        1050,
        0,
    )
    assert plan["can_qualify_C3"] is False

    changed_d6 = read(D6_PLAN)
    changed_d6["trace"]["window_ns"] = 3_500_000_000
    rejected_d6 = frozen_plan(PLAN, inherited_d6=put(tmp_path / D6_PLAN.name, changed_d6))
    assert rejected_d6.returncode == 1 and "D6" in rejected_d6.stderr

    changed_d5r = read(D5R_PLAN)
    changed_d5r["pilot_target_median_c"] = [62, 66]
    rejected_d5r = frozen_plan(PLAN, inherited_d5r=put(tmp_path / D5R_PLAN.name, changed_d5r))
    assert rejected_d5r.returncode == 1 and "D5-R" in rejected_d5r.stderr


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ({"roles": ["D7", "C1"]}, "D7 is limited to one new diagnostic allocation"),
        ({"roles": ["D6", "D7"]}, "D6 is limited to one new diagnostic allocation"),
        (
            {"max_attempts": 16, "max_requested_minutes": 320},
            "D7 is limited to one new diagnostic allocation",
        ),
        ({"d7_plan_sha256": "f" * 64}, "D7 authorization is detached from the reviewed plan"),
    ),
)
def test_d7_authority_child_allows_one_diagnostic_only(tmp_path, mutation, reason):
    record = d7_approval()
    path = tmp_path / "authorization.json"
    approved = authorize(put(path, record))
    assert approved.returncode == 0, approved.stderr
    assert json.loads(approved.stdout) == ["D7"]

    rejected = authorize(put(path, {**record, **mutation}))
    assert rejected.returncode == 1 and rejected.stderr.strip() == reason


@pytest.mark.parametrize(
    ("section", "changed"),
    (
        (None, {"can_qualify_C3": True}),
        (None, {"D6_analysis_sha256": "a" * 64}),
        (None, {"late_timeline_sha256": "b" * 64}),
        (None, {"capture": ["pilot", "step", "endpoint", "endpoint", "step"]}),
        ("trace", {"start_offset_ns": 4_500_000_000, "window_ns": 3_000_000_000}),
        ("waiting", {"noticeable_ns": 30_000_000}),
        ("budget", {"retries": 1}),
    ),
)
def test_d7_reviewed_plan_rejects_changes_even_with_rebound_auth(tmp_path, section, changed):
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    plan = read(PLAN)
    (plan if section is None else plan[section]).update(changed)
    proposal = put(tmp_path / PLAN.name, plan)
    record = d7_approval()
    record["d7_plan_sha256"] = sha(proposal)
    receipt = put(tmp_path / "authorization.json", record)

    approved = authorize(receipt, config)
    assert approved.returncode == 0, approved.stderr
    rejected = frozen_plan(proposal)
    assert rejected.returncode == 1 and rejected.stderr.strip() == "D7 reviewed plan differs"


def test_d7_authority_cannot_unlock_c3_even_if_config_names_it(tmp_path):
    config = read(CONFIG)
    config["calibration"]["qualification_predecessor_roles"] = ["D7"]
    config_path = put(tmp_path / CONFIG.name, config)
    record = d7_approval()
    del record["d7_plan_sha256"]
    record.update(
        config_sha256=sha(config_path),
        max_attempts=22,
        max_requested_minutes=440,
        roles=["C1", "C2", "C3-burst", "C3-mixed", "C4", "C5", "C6"],
        qualification_batch={
            "schema_version": 2,
            "prior_allocation_count": 15,
            "prior_history_sha256": "a" * 64,
            "new_max_attempts": 7,
            "predecessor": {
                "protocol": "same-condition-ee-v1",
                "role": "D7",
                "attempt": "d7-01",
                "execution_commit": record["execution_commit"],
                "proposal_file": PLAN.name,
                "proposal_sha256": sha(PLAN),
                "review_directory": "d7-review",
                "analysis_sha256": "b" * 64,
            },
        },
    )
    rejected = authorize(put(tmp_path / "qualification.json", record), config_path)
    assert rejected.returncode == 1 and "approved same-condition predecessor" in rejected.stderr
