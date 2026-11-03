"""Subprocess controls for the frozen D5 plan and allocation authority."""

import json

import pytest
from test_m5_d4_launch import CONFIG, ROOT, approval, authorize, child, put, read, sha

PLAN = ROOT / "configs/m5-d5.json"
D3_PLAN = ROOT / "configs/m5-d3.json"


def d5_approval():
    record = approval()
    del record["d4_plan_sha256"]
    record.update(
        user_approval_ref="local synthetic D5 authority fixture",
        roles=["D5"],
        max_attempts=12,
        max_requested_minutes=240,
        d5_plan_sha256=sha(PLAN),
    )
    return record


def frozen_plan(plan, inherited=D3_PLAN):
    return child(
        """import json
from pathlib import Path
from vllm_prefix_scheduler.m5_d5 import d5_plan
try:
    proposal, inherited = d5_plan(*map(Path, sys.argv[1:]))
except ValueError as error:
    sys.exit(str(error))
print(json.dumps([proposal, inherited]))
""",
        plan,
        inherited,
    )


def test_d5_plan_child_binds_exact_plan_and_inherited_thermal_protocol(tmp_path):
    result = frozen_plan(PLAN)
    assert result.returncode == 0, result.stderr
    proposal, inherited = json.loads(result.stdout)
    assert proposal == read(PLAN)
    assert inherited == read(D3_PLAN)

    changed = read(D3_PLAN)
    changed["conditioning"]["per_measured_engine_cap_s"] = 95
    result = frozen_plan(PLAN, put(tmp_path / D3_PLAN.name, changed))
    assert result.returncode == 1 and "D5 inherited thermal plan differs" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        {"roles": ["D5", "C1"]},
        {"max_attempts": 13, "max_requested_minutes": 260},
        {"d5_plan_sha256": "f" * 64},
    ),
)
def test_d5_authority_child_allows_one_diagnostic_only(tmp_path, mutation):
    record = d5_approval()
    path = tmp_path / "authorization.json"
    approved = authorize(put(path, record))
    assert approved.returncode == 0, approved.stderr
    assert json.loads(approved.stdout) == ["D5"]

    denied = authorize(put(path, {**record, **mutation}))
    assert denied.returncode == 1 and "D5" in denied.stderr


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        ("[4, 5, 5, 4]", "[5, 4, 4, 5]"),
        ("5553", "5552"),
        ("2048", "2050"),
        ('"can_qualify_C3": false', '"can_qualify_C3": true'),
    ),
)
def test_d5_plan_child_rejects_changes_even_with_rebound_authority(tmp_path, original, replacement):
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    changed = PLAN.read_text()
    assert original in changed
    proposal = tmp_path / PLAN.name
    proposal.write_text(changed.replace(original, replacement))
    record = d5_approval()
    record["d5_plan_sha256"] = sha(proposal)
    receipt = put(tmp_path / "authorization.json", record)

    # The approval hash can be rebound to the on-disk proposal. The reviewed-plan pin is separate.
    allowed = authorize(receipt, config)
    assert allowed.returncode == 0, allowed.stderr
    result = frozen_plan(proposal)
    assert result.returncode == 1 and "D5 reviewed plan differs" in result.stderr


def test_d5_authority_cannot_unlock_c3_even_if_config_names_it(tmp_path):
    config = read(CONFIG)
    config["calibration"]["qualification_predecessor_roles"] = ["D5"]
    config_path = put(tmp_path / CONFIG.name, config)
    record = d5_approval()
    del record["d5_plan_sha256"]
    record.update(
        config_sha256=sha(config_path),
        max_attempts=19,
        max_requested_minutes=380,
        roles=["C1", "C2", "C3-burst", "C3-mixed", "C4", "C5", "C6"],
        qualification_batch={
            "schema_version": 2,
            "prior_allocation_count": 12,
            "prior_history_sha256": "a" * 64,
            "new_max_attempts": 7,
            "predecessor": {
                "protocol": "same-condition-ee-v1",
                "role": "D5",
                "attempt": "d5-01",
                "execution_commit": record["execution_commit"],
                "proposal_file": PLAN.name,
                "proposal_sha256": sha(PLAN),
                "review_directory": "d5-review",
                "analysis_sha256": "b" * 64,
            },
        },
    )
    result = authorize(put(tmp_path / "qualification.json", record), config_path)
    assert result.returncode == 1 and "approved same-condition predecessor" in result.stderr
