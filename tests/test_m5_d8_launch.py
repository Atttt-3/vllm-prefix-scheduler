"""Offline child-process controls for D8's original condition and exclusive slot."""

import json

import pytest
from test_m5_d4_launch import CONFIG, ROOT, approval, authorize, child, put, read, sha

PLAN = ROOT / "configs/m5-d8.json"
PARENT = ROOT / "configs/m5-d2r110.json"


def d8_approval():
    record = approval()
    del record["d4_plan_sha256"]
    record.update(
        user_approval_ref="local synthetic D8 authority fixture",
        roles=["D8"],
        max_attempts=16,
        max_requested_minutes=320,
        d8_plan_sha256=sha(PLAN),
    )
    return record


def frozen_plan(plan, inherited=PARENT):
    return child(
        """import json
from pathlib import Path
from vllm_prefix_scheduler.m5_d8 import d8_plan
try:
    print(json.dumps(d8_plan(*map(Path, sys.argv[1:]))))
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
""",
        plan,
        inherited,
    )


def test_d8_binds_original_condition_and_parent_bytes(tmp_path):
    result = frozen_plan(PLAN)
    assert result.returncode == 0, result.stderr
    plan, parent = json.loads(result.stdout), read(PARENT)
    assert plan == read(PLAN)
    for key in ("target", "conditioning", "stability_gate", "budget", "retained_C3"):
        assert plan[key] == parent[key]
    assert plan["target"]["fresh_engines"] == ["thermal-pilot", "E1", "E2", "E3", "E4"]
    assert plan["target"]["tier"] == "endpoint"
    assert plan["budget"]["qualification_allocations"] == 0

    parent["target"]["arrival_rate_rps"] = 6
    rejected = frozen_plan(PLAN, put(tmp_path / PARENT.name, parent))
    assert rejected.returncode == 1
    assert rejected.stderr.strip() == "D8 inherited D2-R110 plan differs"


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"roles": ["D8", "C1"]}, "D8 is limited to one new diagnostic allocation"),
        ({"roles": ["D7", "D8"]}, "D7 is limited to one new diagnostic allocation"),
        (
            {"max_attempts": 15, "max_requested_minutes": 300},
            "D8 is limited to one new diagnostic allocation",
        ),
        (
            {"max_attempts": 17, "max_requested_minutes": 340},
            "D8 is limited to one new diagnostic allocation",
        ),
        ({"d8_plan_sha256": "f" * 64}, "D8 authorization is detached from the reviewed plan"),
    ),
)
def test_d8_authority_is_exclusive(tmp_path, changes, reason):
    record = d8_approval()
    path = tmp_path / "authorization.json"
    accepted = authorize(put(path, record))
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout) == ["D8"]
    rejected = authorize(put(path, {**record, **changes}))
    assert rejected.returncode == 1 and rejected.stderr.strip() == reason


def test_rehashed_d8_plan_cannot_lower_stability_threshold(tmp_path):
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    plan = read(PLAN)
    plan["stability_gate"]["ttft_p95_relative_margin"] = 0.10
    proposal = put(tmp_path / PLAN.name, plan)
    authorization = {**d8_approval(), "d8_plan_sha256": sha(proposal)}
    accepted_digest = authorize(put(tmp_path / "authorization.json", authorization), config)
    assert accepted_digest.returncode == 0, accepted_digest.stderr
    rejected = frozen_plan(proposal)
    assert rejected.returncode == 1 and rejected.stderr.strip() == "D8 reviewed plan differs"


@pytest.mark.parametrize(
    ("bypass_private_d7", "reason"),
    ((False, "D8 retained D7 analysis differs"), (True, "D2 archived input differs")),
)
def test_d8_does_not_accept_substituted_prior_or_input(tmp_path, bypass_private_d7, reason):
    wrong_prior = put(tmp_path / "previous-d7-analysis.json", {"role": "D7"})
    wrong_table = put(tmp_path / "input.json", {"rate_rps": 6})
    result = child(
        """from pathlib import Path
from vllm_prefix_scheduler import m5_d8
from vllm_prefix_scheduler.m5_campaign import read_json
config, plan, table, original, inherited, prior = map(Path, sys.argv[1:7])
if sys.argv[7] == "True":
    # Portable fixture reaches the exact input hash guard without private D7 bytes.
    m5_d8.d8_predecessor = lambda *_: None
try:
    m5_d8.d8_specs(read_json(config), plan, table, original, inherited, prior)
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
""",
        CONFIG,
        PLAN,
        wrong_table,
        tmp_path / "original-c3-result.json",
        PARENT,
        wrong_prior,
        bypass_private_d7,
    )
    assert result.returncode == 1 and result.stderr.strip() == reason
