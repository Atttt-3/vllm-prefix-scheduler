"""Offline controls for versioned qualification scope and the released evidence gate."""

import importlib
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_campaign import TASK, check_authorization, digest, qualification_roles
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
PRIOR_COMMIT = "b" * 40
PLAN_SHA = "e" * 64
ROLE = "D3-successor"


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


@pytest.fixture
def bundle_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("m5_c3_bundle")


def prospective_config(tmp_path):
    config = json.loads((ROOT / "configs/m5-performance.json").read_text())
    config["calibration"].update(
        qualification_authorization_schema_version=2, qualification_predecessor_roles=[ROLE]
    )
    return config, put(tmp_path / "m5-performance.json", config)


def approval(config_path, config, prior, *, analysis_sha="d" * 64, history_sha="c" * 64):
    return {
        "schema_version": 1,
        "task_id": TASK,
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": digest(config_path),
        "user_approval_ref": "synthetic approval only",
        "roles": list(qualification_roles(config)),
        "max_attempts": prior + 7,
        "max_requested_minutes": (prior + 7) * 20,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
        "qualification_batch": {
            "schema_version": 2,
            "prior_allocation_count": prior,
            "prior_history_sha256": history_sha,
            "new_max_attempts": 7,
            "predecessor": {
                "protocol": "same-condition-ee-v1",
                "role": ROLE,
                "attempt": "predecessor-01",
                "execution_commit": PRIOR_COMMIT,
                "proposal_file": "m5-successor.json",
                "proposal_sha256": PLAN_SHA,
                "review_directory": "successor-review",
                "analysis_sha256": analysis_sha,
            },
        },
    }


@pytest.mark.parametrize("prior", (10, 11))
def test_qualification_uses_explicit_cumulative_budget(tmp_path, prior):
    config, path = prospective_config(tmp_path)
    auth = approval(path, config, prior)
    assert check_authorization(auth, path, COMMIT, "calibration") == auth["roles"]
    for bad in (
        {**auth, "max_attempts": 16, "max_requested_minutes": 320},
        {**auth, "max_attempts": prior + 8, "max_requested_minutes": (prior + 8) * 20},
    ):
        with pytest.raises(ValueError, match="cumulative budget differs"):
            check_authorization(bad, path, COMMIT, "calibration")
    with pytest.raises(ValueError, match="versioned qualification batch"):
        check_authorization(
            {**auth, "d2r110_analysis_sha256": "a" * 64}, path, COMMIT, "calibration"
        )
    auth["qualification_batch"]["predecessor"]["role"] = "D4"
    with pytest.raises(ValueError, match="approved same-condition predecessor"):
        check_authorization(auth, path, COMMIT, "calibration")


def released(index, role, *, complete=True):
    start = datetime(2026, 11, 3) + timedelta(minutes=20 * index)
    return {
        "job_id": str(index + 1),
        "role": role,
        "phase": "calibration",
        "state": "COMPLETED" if complete else "FAILED",
        "exit_code": "0:0" if complete else "1:0",
        "slurm_start": start,
        "slurm_end": start + timedelta(minutes=19),
        "requested_minutes": 20,
        "max_attempts": 20,
        "terminal_sha256": f"{index + 1:064x}",
        "complete": complete,
    }


def stable_review(module, plan, predecessor):
    counts = dict.fromkeys(("planned", "submitted", "completed", "timely_first_token"), 60)
    gates = {
        "overhead_ttft_ns": 2_000_000,
        "overhead_ttft_rel": 0.05,
        "overhead_throughput_rel": 0.05,
    }
    metrics = {
        "strata": {"all": {"counts": counts, "ttft_ns": {"p95": 100_000_000}}},
        "throughput": {"last_terminal_requests_per_s": 8.0, "last_terminal_tokens_per_s": 128.0},
    }
    in_window = {
        "complete": True,
        "temperature_median_c": 63,
        "sm_median_mhz": 1875,
        "memory_mode_mhz": 9501,
    }
    runs = {module.RUNS[0]: {"status": "target-frozen", "prelude": {"verified": True}}}
    for name in module.RUNS[1:]:
        runs[name] = deepcopy(
            {
                "status": "admitted",
                "prelude": {"verified": True},
                "pre_t0_verified": True,
                "endpoint_evidence_complete": True,
                "counts": counts,
                "metrics": metrics,
                "in_window": in_window,
            }
        )
    return {
        "role": ROLE,
        "status": "diagnostic-complete",
        "proposal_sha256": PLAN_SHA,
        "input_sha256": D1_INPUT_SHA,
        "retained_original_C3": {"status": "blocked", "stage_result_sha256": C3_BLOCKED_SHA},
        "execution_commit": PRIOR_COMMIT,
        "analysis_commit": PRIOR_COMMIT,
        "resource_release": json.loads(json.dumps(predecessor, default=str)),
        "verified_thermal_target": {
            "temperature_c": 63,
            "sm_clock_mhz": 1875,
            "memory_clock_mhz": 9501,
        },
        "runs": runs,
        "pairs": [
            {
                "reference": left,
                "observed": right,
                "thermal": module.compare_thermal(in_window, in_window, plan["conditioning"]),
                "performance": module.compare_pair(metrics, metrics, gates),
            }
            for left, right in module.PAIRS
        ],
        "decision": {
            "thermal_control": "verified",
            "observed_endpoint_stability": "stable",
            "endpoint_stability_under_control": "stable",
            "qualification_started": False,
        },
    }


@pytest.fixture
def qualified_bundle(tmp_path, monkeypatch, bundle_module):
    module = bundle_module
    config, config_path = prospective_config(tmp_path)
    history = tmp_path / "attempts"
    history.mkdir()
    prior = [released(index, "diagnostic-history") for index in range(9)]
    predecessor = released(9, ROLE)
    prior.append(predecessor)
    plan = {**json.loads((ROOT / "configs/m5-d2r110.json").read_text()), "role": ROLE}
    review = stable_review(module, plan, predecessor)
    review_dir = tmp_path / "successor-review"
    review_dir.mkdir()
    review_path = put(review_dir / "analysis.json", review)
    auth = approval(
        config_path,
        config,
        len(prior),
        analysis_sha=digest(review_path),
        history_sha=module.qualification_history_sha256(prior),
    )
    auth_path = put(tmp_path / "approval.json", auth)

    def add(row, name):
        path = history / name
        path.mkdir()
        put(path / "submission.json", {"stdout": row["job_id"], "returncode": 0})
        put(
            path / "planned.json",
            {
                "role": row["role"],
                "execution_commit": PRIOR_COMMIT if name == "predecessor-01" else COMMIT,
                "auxiliary_sha256": PLAN_SHA,
                "config_sha256": digest(config_path),
                "authorization_sha256": digest(auth_path),
            },
        )
        return path

    for row in prior[:-1]:
        add(row, f"history-{row['job_id']}")
    attempt = add(predecessor, "predecessor-01")
    monkeypatch.setattr(
        module, "audit_attempt", lambda path, **_: predecessor if path == attempt else None
    )
    monkeypatch.setattr(module, "replay_predecessor", lambda *_: (deepcopy(review), plan))
    return module, config_path, history, auth_path, review_path, prior, add


def test_qualification_checks_prior_prefix_and_ordered_batch(qualified_bundle):
    module, config, history, authorization, _, prior, add = qualified_bundle

    def verify(rows, complete=False):
        return module.verify_split_qualification(
            config, history, authorization, audits=rows, require_complete=complete
        )

    assert verify(prior)["qualification_authorization_sha256"] == digest(authorization)
    with pytest.raises(ValueError, match="all released roles"):
        verify(prior, True)
    changed = deepcopy(prior)
    changed[0]["terminal_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="release prefix"):
        verify(changed)
    failed = released(10, "C1", complete=False)
    add(failed, "c1-failed")
    assert verify([*prior, failed])
    c2 = released(11, "C2")
    add(c2, "c2-early")
    with pytest.raises(ValueError, match="out-of-order"):
        verify([*prior, failed, c2])


def test_qualification_rejects_rehashed_summary_and_stale_release(qualified_bundle):
    module, config, history, auth_path, review_path, prior, add = qualified_bundle
    valid = json.loads(review_path.read_text())
    valid["decision"]["qualification_started"] = True
    put(review_path, valid)
    auth = json.loads(auth_path.read_text())
    auth["qualification_batch"]["predecessor"]["analysis_sha256"] = digest(review_path)
    put(auth_path, auth)
    with pytest.raises(ValueError, match="raw review differs from independent approval"):
        module.verify_split_qualification(config, history, auth_path, audits=prior)
    c1 = released(10, "C1")
    c1["slurm_start"] = prior[-1]["slurm_end"] - timedelta(seconds=1)
    add(c1, "c1-stale")
    with pytest.raises(ValueError, match="out-of-order"):
        module.verify_split_qualification(config, history, auth_path, audits=[*prior, c1])


def test_summary_stable_cannot_override_native_pair_metrics(bundle_module):
    module = bundle_module
    plan = {**json.loads((ROOT / "configs/m5-d2r110.json").read_text()), "role": ROLE}
    resource = released(9, ROLE)
    report = stable_review(module, plan, resource)
    predecessor = {"role": ROLE, "proposal_sha256": PLAN_SHA, "execution_commit": PRIOR_COMMIT}
    module.verify_same_condition_report(report, plan, predecessor, resource)
    report["runs"]["E4"]["metrics"]["strata"]["all"]["ttft_ns"]["p95"] = 140_000_000
    with pytest.raises(ValueError, match="original thermal or paired performance gate"):
        module.verify_same_condition_report(report, plan, predecessor, resource)
