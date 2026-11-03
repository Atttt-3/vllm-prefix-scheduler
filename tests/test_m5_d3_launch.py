"""Check that D3 can use only its one new allocation and retained diagnostic inputs."""

import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m5_campaign, m5_resources
from vllm_prefix_scheduler.m5_campaign import (
    D2_ROLE,
    D2R110_PREVIOUS_ANALYSIS,
    D2R110_ROLE,
    D2R_PREVIOUS_ANALYSIS,
    D2R_ROLE,
    D3_PLAN_FILENAME,
    D3_PREVIOUS_ANALYSIS,
    D3_ROLE,
    TASK,
    bind_d3_inputs,
    check_authorization,
    digest,
)
from vllm_prefix_scheduler.m5_d2 import D3_PROPOSAL_NAME, d3_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/m5-performance.json"
COMMIT = "d" * 40


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def approval(config):
    return {
        "schema_version": 1,
        "task_id": TASK,
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": digest(config),
        "user_approval_ref": "explicit D3 budget fixture",
        "roles": [D3_ROLE],
        "max_attempts": 10,
        "max_requested_minutes": 200,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
        "d3_plan_sha256": digest(config.with_name(D3_PLAN_FILENAME)),
    }


def test_d3_authorization_and_plan_are_exclusive():
    record = approval(CONFIG)
    assert check_authorization(record, CONFIG, COMMIT, "calibration") == [D3_ROLE]
    assert D3_ROLE not in m5_campaign.qualification_roles(json.loads(CONFIG.read_text()))
    for changed, error in (
        ({"roles": [D3_ROLE, "C1"]}, "one new diagnostic allocation"),
        ({"max_attempts": 9, "max_requested_minutes": 180}, "one new diagnostic allocation"),
        ({"max_attempts": 11, "max_requested_minutes": 220}, "one new diagnostic allocation"),
        ({"d3_plan_sha256": "0" * 64}, "detached from the reviewed plan"),
        ({"roles": ["C1"]}, "outside its exclusive authorization"),
    ):
        with pytest.raises(ValueError, match=error):
            check_authorization({**record, **changed}, CONFIG, COMMIT, "calibration")
    plan = json.loads(CONFIG.with_name(D3_PLAN_FILENAME).read_text())
    assert d3_plan(plan)["budget"]["estimated_work_envelope_seconds"] == 1035
    plan["conditioning"]["per_measured_engine_cap_s"] = 111
    with pytest.raises(ValueError, match="measured cap"):
        d3_plan(plan)


def test_d3_binds_verified_but_unstable_predecessor(tmp_path, monkeypatch):
    prior = {"status": "diagnostic-incomplete", "proposal_sha256": "1" * 64}
    retry = {**prior, "role": D2R_ROLE}
    latest = {
        **prior,
        "status": "diagnostic-complete",
        "role": D2R110_ROLE,
        "decision": {
            "thermal_control": "verified",
            "endpoint_stability_under_control": "unstable",
            "qualification_started": False,
        },
    }
    previous = put(tmp_path / D2R_PREVIOUS_ANALYSIS, prior)
    previous_retry = put(tmp_path / D2R110_PREVIOUS_ANALYSIS, retry)
    previous_latest = put(tmp_path / D3_PREVIOUS_ANALYSIS, latest)
    latest_sha = digest(previous_latest)
    monkeypatch.setattr(m5_campaign, "D2R_OFFICIAL_ANALYSIS_SHA256", digest(previous_retry))
    monkeypatch.setattr(m5_campaign, "D2R110_OFFICIAL_ANALYSIS_SHA256", latest_sha)
    records = {
        "retained_D2": {**prior, "analysis_sha256": digest(previous)},
        "retained_D2R": {**retry, "analysis_sha256": digest(previous_retry)},
        "retained_D2R110": {**latest, "analysis_sha256": latest_sha},
    }
    plan = put(tmp_path / D3_PROPOSAL_NAME, records)
    auth = {"roles": [D3_ROLE], "d3_plan_sha256": digest(plan)}
    assert bind_d3_inputs(auth, plan, previous, previous_retry, previous_latest) == (
        digest(previous),
        digest(previous_retry),
        latest_sha,
    )
    latest["decision"]["endpoint_stability_under_control"] = "stable"
    put(previous_latest, latest)
    with pytest.raises(ValueError, match="retained D2-R110 analysis differs"):
        bind_d3_inputs(auth, plan, previous, previous_retry, previous_latest)
    new_sha = digest(previous_latest)
    monkeypatch.setattr(m5_campaign, "D2R110_OFFICIAL_ANALYSIS_SHA256", new_sha)
    records["retained_D2R110"]["analysis_sha256"] = new_sha
    put(plan, records)
    auth["d3_plan_sha256"] = digest(plan)
    with pytest.raises(ValueError, match="endpoint-unstable"):
        bind_d3_inputs(auth, plan, previous, previous_retry, previous_latest)


def test_d3_resource_history_requires_released_ninth_slot(tmp_path, monkeypatch):
    start = datetime(2026, 1, 1)
    rows = []
    for i, role in enumerate(["C3"] * 6 + [D2_ROLE, D2R_ROLE, D2R110_ROLE, D3_ROLE]):
        put(tmp_path / str(i) / "planned.json", {})
        rows.append(
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
    monkeypatch.setattr(m5_resources, "audit_attempt", lambda path: rows[int(path.name)])
    assert m5_resources.audit_history(tmp_path)[-1]["role"] == D3_ROLE
    rows[-2]["complete"] = False
    with pytest.raises(ValueError, match="approved tenth allocation"):
        m5_resources.audit_history(tmp_path)


def test_d3_controller_stages_plan_and_three_analyses_before_any_submission(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    controller = importlib.import_module("submit_m5")
    config = tmp_path / CONFIG.name
    config.write_bytes(CONFIG.read_bytes())
    plan = config.with_name(D3_PLAN_FILENAME)
    plan.write_bytes(CONFIG.with_name(D3_PLAN_FILENAME).read_bytes())
    auth = put(tmp_path / "approval.json", approval(config))
    history = tmp_path / "attempts"
    old = history / "c3-01/node"
    table = put(old / "mixed-2-endpoint/input.json", {"fixture": "table"})
    result = put(old / "calibration-result.json", {"fixture": "C3"})
    prior = {"phase": "calibration", "role": D2R110_ROLE, "complete": True}
    prior_attempt = history / "d2-r110-01"
    put(prior_attempt / "planned.json", {"execution_commit": COMMIT})
    analyses = {
        D2R_PREVIOUS_ANALYSIS: put(tmp_path / "d2-independent-review/analysis.json", {}),
        D2R110_PREVIOUS_ANALYSIS: put(tmp_path / "d2-r-review/analysis.json", {}),
        D3_PREVIOUS_ANALYSIS: put(
            tmp_path / "d2-r110-review/analysis.json",
            {"resource_release": prior, "execution_commit": COMMIT},
        ),
    }
    monkeypatch.setattr(controller, "CONFIG", config)
    monkeypatch.setattr(controller, "HISTORY", history)
    monkeypatch.setattr(controller, "audit_history", lambda _: [prior] * 9)
    monkeypatch.setattr(controller, "snapshot_previous", lambda *args: {})

    def release(path, *, require_completed):
        assert path == prior_attempt and require_completed
        return prior

    monkeypatch.setattr(controller, "audit_attempt", release)
    monkeypatch.setattr(
        controller,
        "subprocess",
        SimpleNamespace(check_output=lambda argv, **_: COMMIT if "rev-parse" in argv else ""),
    )
    monkeypatch.setattr(
        controller, "connection", lambda: ("offline", {"m5_checkout": Path("/scratch/m5")}, "q")
    )
    checked, uploads = [], {}
    monkeypatch.setattr(controller, "bind_d3_inputs", lambda *args: checked.append(args))
    monkeypatch.setattr(controller, "d3_specs", lambda *args: checked.append(args))
    monkeypatch.setattr(
        controller, "upload", lambda ssh, path, text: uploads.update({path.name: text})
    )

    def remote(ssh, argv, **kwargs):
        if argv[0] == "mkdir":
            raise ValueError("offline stop before submission")
        return SimpleNamespace(returncode=0, stdout=COMMIT if "rev-parse" in argv else "")

    monkeypatch.setattr(controller, "remote", remote)
    args = SimpleNamespace(
        authorization=auth, phase="calibration", role=D3_ROLE, attempt="d3-01", freeze=None
    )
    with pytest.raises(ValueError, match="offline stop"):
        controller.submit(args)
    assert checked[0] == (json.loads(auth.read_text()), plan, *analyses.values())
    assert checked[1][1:] == (plan, table, result)
    assert uploads == {
        "authorization.json": auth.read_text(),
        D3_PROPOSAL_NAME: plan.read_text(),
        "input.json": table.read_text(),
        "original-c3-result.json": result.read_text(),
        **{name: path.read_text() for name, path in analyses.items()},
    }
    put(analyses[D3_PREVIOUS_ANALYSIS], {"resource_release": {}, "execution_commit": COMMIT})
    uploads.clear()
    with pytest.raises(ValueError, match="detached from the prior release"):
        controller.submit(args)
    assert not uploads
