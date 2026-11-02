"""Replacement-campaign export checks; synthetic fixtures are never GPU proof."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler import m3_contract

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m3_campaign_export_under_test", ROOT / "scripts/export_m3_campaign.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def projection_fixture(tmp_path):
    from m3_trace_fixture import build_fixture

    fixture = build_fixture(tmp_path / "raw")
    gpu = m3_contract.audit_trace(
        m3_contract.read_events(fixture["trace_path"]),
        fixture["config"],
        resources=fixture["resources"],
        config_sha256=m3_contract.sha256(fixture["config_path"]),
        frozen_inputs=fixture["inputs"],
        inputs_sha256=m3_contract.sha256(fixture["inputs_path"]),
    )
    report = {
        "schema_version": 1,
        "scope": exporter.SCOPE,
        "campaign_id": exporter.CAMPAIGN_ID,
        "campaign_acceptance": "passed",
        "campaign_resource_acceptance": "passed",
        **exporter.EXCEPTION_FIELDS,
        "task_budget_status": "within_8",
        "project_commit": fixture["sources"]["project_commit"],
        "gpu_audit": gpu,
        "history": {
            "allocation_count": 4,
            "unverified_terminal_count": 1,
            "index_sha256": "a" * 64,
            "ledger_sha256": "b" * 64,
            "internal_job_id": "PRIVATE JOB",
            "path": "/private/HISTORY",
        },
        "budget": {
            "historical_allocations": 4,
            "new_allocations": 1,
            "total_allocations": 5,
            "cumulative_limit": 8,
        },
        "native": {"compat_cases": 432, "m1_cases": 12, "private_account": "PRIVATE ACCOUNT"},
        "provenance": {"private_path": "/PRIVATE/raw"},
    }
    return report, fixture


def test_complete_gpu_projection_retains_campaign_scope_and_history_exception(tmp_path):
    report, fixture = projection_fixture(tmp_path)
    public = exporter.public_campaign_summary(report, fixture["sources"], fixture["config"])
    serialized = json.dumps(public)
    assert public["campaign_pass"] is True
    assert public["history_terminal_unverifiable"] is True
    assert public["original_full_history_gate"] == "blocked"
    assert public["original_full_history_gate_blocked"] is True
    assert public["scope"] == exporter.SCOPE
    assert public["functional"]["case_count"] == 20
    assert "accepted" not in serialized
    assert "PRIVATE" not in serialized
    assert "pid" not in serialized and "job_id" not in serialized and "/private/" not in serialized


@pytest.mark.parametrize(
    "mutation",
    [
        "no_exception",
        "historical_pass",
        "unscoped",
        "wrong_campaign",
        "partial_gpu",
        "failed_campaign",
        "failed_resources",
        "old_count",
        "new_budget",
        "cumulative_budget",
        "missing_native",
        "wrong_source",
    ],
)
def test_projection_rejects_scope_history_failure_and_budget_mutations(tmp_path, mutation):
    report, fixture = projection_fixture(tmp_path)
    if mutation == "no_exception":
        del report["history_terminal_unverifiable"]
    elif mutation == "historical_pass":
        report["original_full_history_gate"] = "passed"
    elif mutation == "unscoped":
        report["scope"] = "complete"
    elif mutation == "wrong_campaign":
        report["campaign_id"] = "relabelled-old-campaign"
    elif mutation == "partial_gpu":
        report["gpu_audit"]["scope"] = "partial"
    elif mutation == "failed_campaign":
        report["campaign_acceptance"] = "failed"
    elif mutation == "failed_resources":
        report["campaign_resource_acceptance"] = "failed"
    elif mutation == "old_count":
        report["history"]["allocation_count"] = 3
    elif mutation == "new_budget":
        report["budget"]["new_allocations"] = 5
    elif mutation == "cumulative_budget":
        report["budget"]["total_allocations"] = 4
    elif mutation == "missing_native":
        report["native"]["compat_cases"] = 0
    else:
        report["project_commit"] = "d" * 40
    with pytest.raises(ValueError):
        exporter.public_campaign_summary(report, fixture["sources"], fixture["config"])


def make_git_source_fixture(tmp_path):
    repo = tmp_path / "git-source"
    repo.mkdir()
    names = (
        "scripts/export_m3.py",
        "scripts/export_m3_campaign.py",
        "src/vllm_prefix_scheduler/m3_contract.py",
        "src/vllm_prefix_scheduler/m3_campaign_contract.py",
        "configs/m3-real-model.json",
        "configs/m3-campaign-v1.json",
        "configs/m3-campaign-authorization.json",
        "configs/m3-closeout-authorization.json",
    )
    for name in names:
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=M3 contract fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "synthetic source fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    sources = {
        "project_commit": commit,
        "project_dirty": False,
        "source_sha256": {name: m3_contract.sha256(repo / name) for name in names},
    }
    return repo, sources


def test_real_git_source_binding_includes_both_authorizations_and_both_exporters(tmp_path):
    repo, sources = make_git_source_fixture(tmp_path)
    options = {
        "campaign_config": repo / "configs/m3-campaign-v1.json",
        "authorization": repo / "configs/m3-campaign-authorization.json",
        "config": repo / "configs/m3-real-model.json",
    }
    hashes = exporter.verify_frozen_campaign_sources(repo, sources, **options)
    assert set(hashes) == set(sources["source_sha256"])
    broken = copy.deepcopy(sources)
    broken["source_sha256"]["configs/m3-closeout-authorization.json"] = "0" * 64
    with pytest.raises(ValueError, match="Git blob"):
        exporter.verify_frozen_campaign_sources(repo, broken, **options)
    (repo / "scripts/export_m3.py").write_text("old export modified after freeze\n")
    with pytest.raises(ValueError, match="frozen source"):
        exporter.verify_frozen_campaign_sources(repo, sources, **options)


@pytest.mark.parametrize("optimize", [False, True])
def test_real_cli_refuses_unscoped_or_optimized_publication(tmp_path, optimize):
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"schema_version": 1, "scope": "complete", "accepted": True}))
    output = tmp_path / "public"
    command = [
        sys.executable,
        *(["-O"] if optimize else []),
        str(ROOT / "scripts/export_m3_campaign.py"),
    ]
    for name in (*exporter.PATH_ARGUMENTS, "sources", "audit"):
        command.extend(["--" + name.replace("_", "-"), str(payload)])
    command.extend(["--output", str(output)])
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert ("assertions enabled" if optimize else "unrecognized campaign scope") in result.stderr
    assert not output.exists()


def test_indirect_history_native_resource_hashes_are_rechecked_before_publication(tmp_path):
    raw = tmp_path / "PRIVATE-native-original.json"
    raw.write_text('{"PRIVATE account":"retained only in raw"}')
    audit = {
        "provenance": {
            "artifact_inventory": [
                {
                    "path": str(raw),
                    "sha256": m3_contract.sha256(raw),
                    "bytes": raw.stat().st_size,
                }
            ]
        }
    }
    public = exporter.verified_artifact_hashes(audit)
    assert public == [{"sha256": m3_contract.sha256(raw), "bytes": raw.stat().st_size}]
    assert "PRIVATE" not in json.dumps(public)
    raw.write_text('{"PRIVATE account":"changed after audit"}')
    with pytest.raises(ValueError, match="changed after audit"):
        exporter.verified_artifact_hashes(audit)
    with pytest.raises(ValueError, match="inventory missing"):
        exporter.verified_artifact_hashes({"provenance": {}})


def test_campaign_auditor_to_public_projection_with_explicit_native_git_test_seams(
    tmp_path, monkeypatch
):
    """Interface integration only; mocked Git/native seams are not acceptance evidence."""
    from m3_campaign_fixture import build_campaign_fixture

    from vllm_prefix_scheduler import m3_campaign_contract

    fixture = build_campaign_fixture(tmp_path / "campaign")
    monkeypatch.setattr(
        m3_campaign_contract, "frozen_sources", lambda root, sources: sources["source_sha256"]
    )
    monkeypatch.setattr(
        m3_campaign_contract,
        "native_raw_audit",
        lambda root, kind, trace: {
            "cases": 432 if kind == "compat" else 12,
            "raw_audit_sha256": "f" * 64,
        },
    )
    audit = m3_campaign_contract.audit_campaign(**fixture["paths"])
    events = m3_contract.read_events(fixture["paths"]["trace_dir"])
    sources = next(event["sources"] for event in events if event["event"] == "engine_ready")
    public = exporter.public_campaign_summary(
        audit, sources, json.loads(fixture["paths"]["config"].read_text())
    )
    indirect = exporter.verified_artifact_hashes(audit)
    assert public["campaign_pass"] is True
    assert public["functional"]["case_count"] == 20
    assert public["historical_resource_status"] == "exception_unverified_terminal"
    assert public["original_full_history_gate"] == "blocked"
    assert public["budget"] == {
        "historical_allocations": 4,
        "new_allocations": 1,
        "total_allocations": 5,
        "cumulative_limit": 8,
    }
    assert len(indirect) >= 33
    serialized = json.dumps({"summary": public, "private_artifact_hashes": indirect})
    assert "accepted" not in serialized
    assert str(tmp_path) not in serialized
    assert "campaign_job_id" not in serialized
    # No monkeypatch of the production CLI/exporter's source gate is allowed:
    # actual source-binding positive/negative tests above exercise real Git.
