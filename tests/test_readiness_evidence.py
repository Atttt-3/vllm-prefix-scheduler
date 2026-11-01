"""Protect published readiness claims; these are not native Scheduler tests."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from vllm_prefix_scheduler.evidence import check_readiness

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "2026-11-01-rtx3090"


@pytest.fixture
def record():
    return json.loads((EVIDENCE / "validation.json").read_text())


@pytest.fixture
def pins():
    return json.loads((ROOT / "configs" / "upstream.json").read_text())


def test_archived_native_run_supports_the_readiness_claims(record, pins):
    summary = check_readiness(record, pins)
    archived = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary == archived
    assert len(record["requests"]) == 7


def test_frozen_baseline_matches_the_archived_engine_arguments(record):
    baseline = json.loads((ROOT / "configs" / "baseline-3090.json").read_text())
    for key, value in baseline.items():
        assert record["requested_engine_args"][key] == value


@pytest.mark.parametrize(
    "corruption",
    ["async_disabled", "kv_mismatch", "apc_no_reuse", "unfinished_request"],
)
def test_rejects_unsupported_or_corrupted_readiness_claim(record, pins, corruption):
    changed = deepcopy(record)
    if corruption == "async_disabled":
        changed["worker_after_requests"][0]["async_scheduling"] = False
    elif corruption == "kv_mismatch":
        changed["kv_capacity"]["capacity_tokens"] += 16
    elif corruption == "apc_no_reuse":
        changed["requests"][2]["num_cached_tokens"] = 0
    elif corruption == "unfinished_request":
        changed["requests"][-1]["finished"] = False
    with pytest.raises(ValueError):
        check_readiness(changed, pins)


def test_exported_files_match_their_provenance_hashes():
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    for name, expected in provenance["exported_file_sha256"].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected


def test_exported_data_contains_no_cluster_identity_fields():
    forbidden = {
        "hostname",
        "username",
        "slurm_job_id",
        "cuda_visible_devices",
        "uuid",
        "index",
        "python_executable",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, str):
            assert "/userhome/" not in value
            assert "/Users/" not in value

    for name in ("validation.json", "environment.json"):
        inspect(json.loads((EVIDENCE / name).read_text()))


def test_installed_core_sources_match_the_pinned_commit(pins):
    source = json.loads((EVIDENCE / "pinned-source-verification.json").read_text())
    assert source["commit"] == pins["vllm"]["commit"]
    assert source["all_match"]
    for item in source["files"]:
        assert item["matches"]
        assert item["installed_sha256"] == item["upstream_sha256"]


def test_exporter_removes_unexpected_private_fields_and_preserves_numbers(record, tmp_path):
    raw = deepcopy(record)
    raw["python"] = raw.pop("python_version")
    raw["hostname"] = "private-login-host"
    raw["slurm_job_id"] = "private-job-id"
    raw["kv_capacity"]["unrecognized_private_field"] = "private-value"
    raw["worker_after_requests"][0]["kv_cache_groups"][0]["private_field"] = "private-value"
    raw["gpu_after_requests"]["gpus"][0]["uuid"] = "private-gpu-uuid"
    raw["requested_engine_args"]["model"] = "/private/model/path"
    run_dir = tmp_path / "raw"
    run_dir.mkdir()
    (run_dir / "validation.json").write_text(json.dumps(raw))
    exported = tmp_path / "exported"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_readiness.py"),
        "--run-dir",
        str(run_dir),
        "--output",
        str(exported),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    public = json.loads((exported / "validation.json").read_text())
    assert public["kv_capacity"] == record["kv_capacity"]
    assert public["requests"] == record["requests"]
    assert "private-" not in (exported / "validation.json").read_text()
    assert "private_field" not in (exported / "validation.json").read_text()
    before = (exported / "validation.json").read_bytes()
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert (exported / "validation.json").read_bytes() == before
