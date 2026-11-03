"""Limit D2 diagnostic roles to their separately authorized allocation slots."""

import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler.m5_campaign import (
    CALIBRATION,
    D2_ROLE,
    D2R110_PLAN_FILENAME,
    D2R110_PREVIOUS_ANALYSIS,
    D2R110_ROLE,
    D2R_PLAN_FILENAME,
    D2R_PREVIOUS_ANALYSIS,
    D2R_ROLE,
    TASK,
    bind_d2r110_inputs,
    bind_d2r_inputs,
    check_authorization,
    digest,
    qualification_roles,
)
from vllm_prefix_scheduler.m5_d2 import D2R110_PROPOSAL_NAME, D2R_PROPOSAL_NAME

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


def d2r110_approval(config=CONFIG):
    approval = d2r_approval()
    del approval["d2r_plan_sha256"]
    return {
        **approval,
        "config_sha256": digest(config),
        "roles": [D2R110_ROLE],
        "max_attempts": 9,
        "max_requested_minutes": 180,
        "d2r110_plan_sha256": digest(config.with_name(D2R110_PLAN_FILENAME)),
    }


def test_d2r110_requires_exact_exclusive_ninth_slot():
    approval = d2r110_approval()
    assert check_authorization(approval, CONFIG, COMMIT, "calibration") == [D2R110_ROLE]
    assert D2R110_ROLE not in CALIBRATION
    for changed, reason in (
        ({"roles": [D2R110_ROLE, "C1"]}, "one new diagnostic allocation"),
        ({"max_attempts": 8, "max_requested_minutes": 160}, "one new diagnostic allocation"),
        ({"max_attempts": 10, "max_requested_minutes": 200}, "one new diagnostic allocation"),
        ({"d2r110_plan_sha256": "f" * 64}, "detached from the reviewed plan"),
        ({"roles": ["C1"]}, "outside its exclusive authorization"),
    ):
        with pytest.raises(ValueError, match=reason):
            check_authorization({**approval, **changed}, CONFIG, COMMIT, "calibration")


def test_split_c3_requires_a_separate_exact_seven_slot_authorization(tmp_path):
    config = tmp_path / CONFIG.name
    prospective = json.loads(CONFIG.read_text())
    prospective["calibration"].pop("qualification_authorization_schema_version", None)
    prospective["calibration"].pop("qualification_predecessor_roles", None)
    prospective["calibration"].pop("split_c3", None)
    assert qualification_roles(prospective) == CALIBRATION
    prospective["calibration"]["split_c3"] = True
    config.write_text(json.dumps(prospective))
    roles = ["C1", "C2", "C3-burst", "C3-mixed", "C4", "C5", "C6"]
    assert qualification_roles(prospective) == tuple(roles)
    approval = d2r110_approval()
    del approval["d2r110_plan_sha256"]
    approval.update(
        config_sha256=digest(config),
        roles=roles,
        max_attempts=16,
        max_requested_minutes=320,
        d2r110_analysis_sha256="a" * 64,
    )
    assert check_authorization(approval, config, COMMIT, "calibration") == roles
    for changed in (
        {"roles": roles[1:]},
        {"max_attempts": 15, "max_requested_minutes": 300},
        {"max_attempts": 17, "max_requested_minutes": 340},
        {"d2r110_analysis_sha256": None},
    ):
        with pytest.raises(ValueError, match="exclusive seven-slot"):
            check_authorization({**approval, **changed}, config, COMMIT, "calibration")


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


def test_d2r110_staging_requires_both_retained_analyses(tmp_path, monkeypatch):
    from vllm_prefix_scheduler import m5_campaign

    proposal = tmp_path / D2R110_PROPOSAL_NAME
    previous = tmp_path / D2R_PREVIOUS_ANALYSIS
    previous_d2r = tmp_path / D2R110_PREVIOUS_ANALYSIS
    prior = {"status": "diagnostic-incomplete", "proposal_sha256": "a" * 64}
    prior_d2r = {"role": D2R_ROLE, **prior, "proposal_sha256": "b" * 64}
    previous.write_text(json.dumps(prior))
    previous_d2r.write_text(json.dumps(prior_d2r))
    previous_sha, previous_d2r_sha = digest(previous), digest(previous_d2r)
    monkeypatch.setattr(m5_campaign, "D2R_OFFICIAL_ANALYSIS_SHA256", previous_d2r_sha)
    plan = {
        "retained_D2": {**prior, "analysis_sha256": previous_sha},
        "retained_D2R": {**prior_d2r, "analysis_sha256": previous_d2r_sha},
    }
    proposal.write_text(json.dumps(plan))
    approval = {"roles": [D2R110_ROLE], "d2r110_plan_sha256": digest(proposal)}
    assert bind_d2r110_inputs(approval, proposal, previous, previous_d2r) == (
        previous_sha,
        previous_d2r_sha,
    )
    previous_d2r.write_text(json.dumps(prior_d2r, indent=2))
    with pytest.raises(ValueError, match="retained D2-R analysis differs"):
        bind_d2r110_inputs(approval, proposal, previous, previous_d2r)
    plan["retained_D2R"]["analysis_sha256"] = digest(previous_d2r)
    proposal.write_text(json.dumps(plan))
    approval["d2r110_plan_sha256"] = digest(proposal)
    with pytest.raises(ValueError, match="retained D2-R analysis differs"):
        bind_d2r110_inputs(approval, proposal, previous, previous_d2r)
    previous.write_text(json.dumps(prior, indent=2))
    with pytest.raises(ValueError, match="retained D2 analysis differs"):
        bind_d2r110_inputs(approval, proposal, previous, previous_d2r)


@pytest.mark.parametrize(
    ("terminal_role", "prior_role", "word"),
    ((D2R_ROLE, D2_ROLE, "eighth"), (D2R110_ROLE, D2R_ROLE, "ninth")),
)
def test_d2_resource_history_requires_retained_previous_slot(
    tmp_path, monkeypatch, terminal_role, prior_role, word
):
    from vllm_prefix_scheduler import m5_resources

    start = datetime(2026, 1, 1)
    previous = []
    tail = [D2_ROLE, D2R_ROLE]
    if terminal_role == D2R110_ROLE:
        tail.append(D2R110_ROLE)
    for i, role in enumerate(["C3"] * 6 + tail):
        folder = tmp_path / f"a{i}"
        folder.mkdir()
        (folder / "planned.json").write_text("{}")
        previous.append(
            {
                "job_id": str(i + 1),
                "phase": "calibration",
                "role": role,
                "complete": True,
                "max_attempts": max(6, i + 1),
                "slurm_start": start + timedelta(minutes=20 * i),
                "slurm_end": start + timedelta(minutes=20 * i + 1),
            }
        )
    monkeypatch.setattr(m5_resources, "audit_attempt", lambda path: previous[int(path.name[1:])])
    reviewed = m5_resources.audit_history(tmp_path)
    assert [row["role"] for row in reviewed[-2:]] == [prior_role, terminal_role]
    previous[-2]["complete"] = False
    with pytest.raises(ValueError, match=f"approved {word} allocation"):
        m5_resources.audit_history(tmp_path)


@pytest.mark.parametrize("role", (D2R_ROLE, D2R110_ROLE))
def test_d2_controller_stages_exact_plan_and_retained_inputs(tmp_path, monkeypatch, role):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    controller = importlib.import_module("submit_m5")
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    retry = role == D2R110_ROLE
    plan_name = D2R110_PLAN_FILENAME if retry else D2R_PLAN_FILENAME
    proposal = config.with_name(plan_name)
    proposal.write_bytes(CONFIG.with_name(plan_name).read_bytes())
    auth_path = tmp_path / "approval.json"
    auth = d2r110_approval(config) if retry else d2r_approval(config)
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
    retained_d2r = tmp_path / "d2-r-review/analysis.json"
    retained_d2r.parent.mkdir()
    retained_d2r.write_text('{"fixture":"incomplete D2-R"}')
    predecessor = {"phase": "calibration", "role": D2R_ROLE if retry else D2_ROLE, "complete": True}
    prior_attempt = "d2-r-01" if retry else "d2-01"
    monkeypatch.setattr(controller, "CONFIG", config)
    monkeypatch.setattr(controller, "HISTORY", history)
    monkeypatch.setattr(controller, "audit_history", lambda _: [predecessor] * (8 if retry else 7))
    monkeypatch.setattr(controller, "snapshot_previous", lambda *args: {})
    monkeypatch.setattr(
        controller,
        "audit_attempt",
        lambda path, require_completed: (
            predecessor
            if path == history / prior_attempt and require_completed
            else pytest.fail("wrong retained D2 predecessor attempt")
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
    monkeypatch.setattr(
        controller,
        "bind_d2r110_inputs" if retry else "bind_d2r_inputs",
        lambda *args: checked.append(args),
    )
    monkeypatch.setattr(
        controller, "d2r110_specs" if retry else "d2r_specs", lambda *args: checked.append(args)
    )
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
                role=role,
                attempt="d2-r110-01" if retry else "d2-r-01",
                freeze=None,
            )
        )
    expected_retained = (retained, retained_d2r) if retry else (retained,)
    assert checked[0] == (auth, proposal, *expected_retained)
    assert checked[1][1:] == (proposal, table, result)
    assert uploads == {
        "authorization.json": auth_path.read_text(),
        (D2R110_PROPOSAL_NAME if retry else D2R_PROPOSAL_NAME): proposal.read_text(),
        "input.json": table.read_text(),
        "original-c3-result.json": result.read_text(),
        D2R_PREVIOUS_ANALYSIS: retained.read_text(),
        **({D2R110_PREVIOUS_ANALYSIS: retained_d2r.read_text()} if retry else {}),
    }
