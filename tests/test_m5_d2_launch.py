"""Limit D2 and D2-R to their separately authorized allocation slots."""

import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler.m5_campaign import (
    CALIBRATION,
    D2_ROLE,
    D2R_PLAN_FILENAME,
    D2R_PREVIOUS_ANALYSIS,
    D2R_ROLE,
    TASK,
    bind_d2r_inputs,
    check_authorization,
    digest,
)
from vllm_prefix_scheduler.m5_d2 import D2R_PROPOSAL_NAME

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/m5-performance.json"
COMMIT = "b" * 40


def test_d2_has_one_new_exclusive_slot():
    approval = {
        "schema_version": 1,
        "task_id": TASK,
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": digest(CONFIG),
        "user_approval_ref": "explicit one-slot D2 approval",
        "roles": [D2_ROLE],
        "max_attempts": 7,
        "max_requested_minutes": 140,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
    }
    assert check_authorization(approval, CONFIG, COMMIT, "calibration") == [D2_ROLE]
    assert D2_ROLE not in CALIBRATION
    for invalid in (
        {"roles": ["C4", D2_ROLE]},
        {"max_attempts": 8, "max_requested_minutes": 160},
        {"max_attempts": 6, "max_requested_minutes": 120},
    ):
        with pytest.raises(ValueError, match="one new diagnostic allocation"):
            check_authorization({**approval, **invalid}, CONFIG, COMMIT, "calibration")


def d2r_approval(config=CONFIG):
    return {
        "schema_version": 1,
        "task_id": TASK,
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": digest(config),
        "user_approval_ref": "fixture: explicit one-slot D2-R approval",
        "roles": [D2R_ROLE],
        "max_attempts": 8,
        "max_requested_minutes": 160,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
        "d2r_plan_sha256": digest(config.with_name(D2R_PLAN_FILENAME)),
    }


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"roles": ["C4", D2R_ROLE]}, "one new diagnostic allocation"),
        ({"max_attempts": 7, "max_requested_minutes": 140}, "one new diagnostic allocation"),
        ({"max_attempts": 9, "max_requested_minutes": 180}, "one new diagnostic allocation"),
        ({"d2r_plan_sha256": "f" * 64}, "detached from the reviewed plan"),
        ({"d2r_plan_sha256": None}, "detached from the reviewed plan"),
        (
            {"roles": [D2_ROLE], "max_attempts": 7, "max_requested_minutes": 140},
            "outside its exclusive authorization",
        ),
    ],
)
def test_d2r_requires_exact_exclusive_authorization(change, reason):
    approval = d2r_approval()
    assert check_authorization(approval, CONFIG, COMMIT, "calibration") == [D2R_ROLE]
    assert D2R_ROLE not in CALIBRATION
    with pytest.raises(ValueError, match=reason):
        check_authorization({**approval, **change}, CONFIG, COMMIT, "calibration")


def test_d2r_staged_plan_and_predecessor_are_independently_bound(tmp_path):
    proposal = tmp_path / D2R_PROPOSAL_NAME
    previous = tmp_path / D2R_PREVIOUS_ANALYSIS
    prior = {"status": "diagnostic-incomplete", "proposal_sha256": "a" * 64}
    previous.write_text(json.dumps(prior))
    previous_sha = digest(previous)
    plan = {"retained_D2": {**prior, "analysis_sha256": previous_sha}}
    proposal.write_text(json.dumps(plan))
    approval = {"roles": [D2R_ROLE], "d2r_plan_sha256": digest(proposal)}
    assert bind_d2r_inputs(approval, proposal, previous) == previous_sha

    proposal.write_text(json.dumps(plan, indent=2))
    with pytest.raises(ValueError, match="staged plan differs"):
        bind_d2r_inputs(approval, proposal, previous)
    approval["d2r_plan_sha256"] = digest(proposal)
    previous.write_text(json.dumps({**prior, "status": "diagnostic-complete"}))
    with pytest.raises(ValueError, match="retained D2 analysis differs"):
        bind_d2r_inputs(approval, proposal, previous)
    plan["retained_D2"]["analysis_sha256"] = digest(previous)
    proposal.write_text(json.dumps(plan))
    approval["d2r_plan_sha256"] = digest(proposal)
    with pytest.raises(ValueError, match="not the retained incomplete D2"):
        bind_d2r_inputs(approval, proposal, previous)


def test_d2r_resource_history_requires_retained_seventh_slot(tmp_path, monkeypatch):
    from vllm_prefix_scheduler import m5_resources

    start = datetime(2026, 1, 1)
    previous = []
    for i in range(8):
        folder = tmp_path / f"a{i}"
        folder.mkdir()
        (folder / "planned.json").write_text("{}")
        previous.append(
            {
                "job_id": str(i + 1),
                "phase": "calibration",
                "role": "C3" if i < 6 else D2_ROLE if i == 6 else D2R_ROLE,
                "complete": True,
                "max_attempts": max(6, i + 1),
                "slurm_start": start + timedelta(minutes=20 * i),
                "slurm_end": start + timedelta(minutes=20 * i + 1),
            }
        )
    monkeypatch.setattr(m5_resources, "audit_attempt", lambda path: previous[int(path.name[1:])])
    assert m5_resources.audit_history(tmp_path)[-1]["role"] == D2R_ROLE
    previous[-2]["complete"] = False
    with pytest.raises(ValueError, match="approved eighth allocation"):
        m5_resources.audit_history(tmp_path)


def test_d2r_controller_stages_exact_plan_and_retained_inputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    controller = importlib.import_module("submit_m5")
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    proposal = config.with_name(D2R_PLAN_FILENAME)
    proposal.write_bytes(CONFIG.with_name(D2R_PLAN_FILENAME).read_bytes())
    auth_path = tmp_path / "approval.json"
    auth = d2r_approval(config)
    auth_path.write_text(json.dumps(auth))
    history = tmp_path / "attempts"
    old_c3 = history / "c3-01/node"
    table = old_c3 / "mixed-2-endpoint/input.json"
    table.parent.mkdir(parents=True)
    table.write_text('{"fixture":"frozen mixed table"}')
    result = old_c3 / "calibration-result.json"
    result.write_text('{"fixture":"blocked C3"}')
    retained = tmp_path / "d2-independent-review/analysis.json"
    retained.parent.mkdir()
    retained.write_text('{"fixture":"incomplete D2"}')
    predecessor = {"phase": "calibration", "role": D2_ROLE, "complete": True}
    monkeypatch.setattr(controller, "CONFIG", config)
    monkeypatch.setattr(controller, "HISTORY", history)
    monkeypatch.setattr(controller, "audit_history", lambda _: [predecessor] * 7)
    monkeypatch.setattr(controller, "snapshot_previous", lambda *args: {})
    monkeypatch.setattr(
        controller,
        "audit_attempt",
        lambda path, require_completed: (
            predecessor
            if path == history / "d2-01" and require_completed
            else pytest.fail("wrong retained D2 attempt")
        ),
    )
    monkeypatch.setattr(
        controller,
        "subprocess",
        SimpleNamespace(check_output=lambda argv, **_: COMMIT if "rev-parse" in argv else ""),
    )
    monkeypatch.setattr(
        controller, "connection", lambda: ("offline", {"m5_checkout": Path("/scratch/m5")}, "q")
    )
    checked = []
    monkeypatch.setattr(controller, "bind_d2r_inputs", lambda *args: checked.append(args))
    monkeypatch.setattr(controller, "d2r_specs", lambda *args: checked.append(args))
    uploads = {}
    monkeypatch.setattr(
        controller, "upload", lambda ssh, path, text: uploads.update({path.name: text})
    )

    def remote(ssh, argv, **kwargs):
        if argv[0] == "mkdir":
            raise ValueError("offline stop before controller or submission")
        return SimpleNamespace(returncode=0, stdout=COMMIT if "rev-parse" in argv else "")

    monkeypatch.setattr(controller, "remote", remote)
    with pytest.raises(ValueError, match="offline stop"):
        controller.submit(
            SimpleNamespace(
                authorization=auth_path,
                phase="calibration",
                role=D2R_ROLE,
                attempt="d2-r-01",
                freeze=None,
            )
        )
    assert checked[0] == (auth, proposal, retained)
    assert checked[1][1:] == (proposal, table, result)
    assert uploads == {
        "authorization.json": auth_path.read_text(),
        D2R_PROPOSAL_NAME: proposal.read_text(),
        "input.json": table.read_text(),
        "original-c3-result.json": result.read_text(),
        D2R_PREVIOUS_ANALYSIS: retained.read_text(),
    }
