"""Real subprocess controls for D5-R plan, authorization, and ancestry."""

import json

import pytest
from test_m5_d4_launch import CONFIG, ROOT, authorize, child, put, read, sha
from test_m5_d5_launch import D3_PLAN, d5_approval

D5_PLAN = ROOT / "configs/m5-d5.json"
PLAN = ROOT / "configs/m5-d5r.json"


def d5r_approval():
    record = d5_approval()
    del record["d5_plan_sha256"]
    record.update(
        user_approval_ref="local synthetic D5-R authority fixture",
        roles=["D5-R"],
        max_attempts=13,
        max_requested_minutes=260,
        d5r_plan_sha256=sha(PLAN),
    )
    return record


def frozen_plan(plan, inherited_d5=D5_PLAN, inherited_d3=D3_PLAN):
    return child(
        """import json
from pathlib import Path
from vllm_prefix_scheduler.m5_d5r import d5r_plan
try:
    print(json.dumps(d5r_plan(*map(Path, sys.argv[1:]))))
except ValueError as error:
    sys.exit(str(error))
""",
        plan,
        inherited_d5,
        inherited_d3,
    )


def test_d5r_plan_child_binds_exact_d5_and_d3_protocols(tmp_path):
    result = frozen_plan(PLAN)
    assert result.returncode == 0, result.stderr
    candidate, d5, d3 = json.loads(result.stdout)
    assert candidate == read(PLAN)
    assert candidate["pilot_target_median_c"] == [64, 66]
    assert d5 == read(D5_PLAN) and d3 == read(D3_PLAN)

    changed_d5 = read(D5_PLAN)
    changed_d5["target"]["measure_order_max_num_active_seqs"] = [5, 4, 4, 5]
    rejected_d5 = frozen_plan(PLAN, put(tmp_path / D5_PLAN.name, changed_d5))
    assert rejected_d5.returncode == 1 and "D5" in rejected_d5.stderr

    changed_d3 = read(D3_PLAN)
    changed_d3["conditioning"]["per_measured_engine_cap_s"] = 115
    rejected_d3 = frozen_plan(PLAN, inherited_d3=put(tmp_path / D3_PLAN.name, changed_d3))
    assert rejected_d3.returncode == 1 and "thermal plan" in rejected_d3.stderr


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ({"roles": ["D5-R", "C1"]}, "D5-R is limited to one new diagnostic allocation"),
        ({"roles": ["D5", "D5-R"]}, "D5 is limited to one new diagnostic allocation"),
        (
            {"max_attempts": 14, "max_requested_minutes": 280},
            "D5-R is limited to one new diagnostic allocation",
        ),
        ({"d5r_plan_sha256": "f" * 64}, "D5-R authorization is detached from the reviewed plan"),
    ),
)
def test_d5r_authority_child_allows_one_diagnostic_only(tmp_path, mutation, reason):
    record = d5r_approval()
    path = tmp_path / "authorization.json"
    approved = authorize(put(path, record))
    assert approved.returncode == 0, approved.stderr
    assert json.loads(approved.stdout) == ["D5-R"]

    rejected = authorize(put(path, {**record, **mutation}))
    assert rejected.returncode == 1 and rejected.stderr.strip() == reason


@pytest.mark.parametrize(
    "changed",
    (
        {"pilot_target_median_c": [62, 66]},
        {"pilot_target_median_c": [64, 67]},
        {"D5_analysis_sha256": "a" * 64},
        {"D5_plan_sha256": "b" * 64},
        {"can_qualify_C3": True},
    ),
)
def test_d5r_plan_child_rejects_tampering_even_with_rebound_authority(tmp_path, changed):
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    proposal = put(tmp_path / PLAN.name, {**read(PLAN), **changed})
    record = d5r_approval()
    record["d5r_plan_sha256"] = sha(proposal)
    receipt = put(tmp_path / "authorization.json", record)

    # The reviewed plan pin is separate from the approval's digest of the local file.
    approved = authorize(receipt, config)
    assert approved.returncode == 0, approved.stderr
    rejected = frozen_plan(proposal)
    assert rejected.returncode == 1 and "reviewed plan differs" in rejected.stderr


def test_d5r_authority_cannot_unlock_c3_even_if_config_names_it(tmp_path):
    config = read(CONFIG)
    config["calibration"]["qualification_predecessor_roles"] = ["D5-R"]
    config_path = put(tmp_path / CONFIG.name, config)
    record = d5r_approval()
    del record["d5r_plan_sha256"]
    record.update(
        config_sha256=sha(config_path),
        max_attempts=20,
        max_requested_minutes=400,
        roles=["C1", "C2", "C3-burst", "C3-mixed", "C4", "C5", "C6"],
        qualification_batch={
            "schema_version": 2,
            "prior_allocation_count": 13,
            "prior_history_sha256": "a" * 64,
            "new_max_attempts": 7,
            "predecessor": {
                "protocol": "same-condition-ee-v1",
                "role": "D5-R",
                "attempt": "d5-r-01",
                "execution_commit": record["execution_commit"],
                "proposal_file": PLAN.name,
                "proposal_sha256": sha(PLAN),
                "review_directory": "d5-r-review",
                "analysis_sha256": "b" * 64,
            },
        },
    )
    rejected = authorize(put(tmp_path / "qualification.json", record), config_path)
    assert rejected.returncode == 1 and "approved same-condition predecessor" in rejected.stderr
