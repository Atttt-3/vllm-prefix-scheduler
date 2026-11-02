"""Export projection/source tests are synthetic tooling tests, never GPU proof."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler import m3_contract

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m3_export_under_test", ROOT / "scripts/export_m3.py")
export_m3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_m3)


def projection_fixture():
    config = json.loads((ROOT / "configs/m3-real-model.json").read_text())
    commit = "a" * 40
    sources = {
        "project_commit": commit,
        "python_version": "3.12.16",
        "packages": {
            "vllm": "0.32.0",
            "torch": "2.13.0+cu129",
            "transformers": "5.0.0",
            "tokenizers": "0.22.0",
            "safetensors": "0.7.0",
        },
    }
    cases = {}
    for name, meta in m3_contract.case_catalog().items():
        roles = (
            ["S0", "S1", "S2"]
            if meta["scenario"] == "smoke"
            else ["X"]
            if meta["scenario"] == "cancel"
            else ["A", "B", "C"]
        )
        case = {
            "inputs": {role: {"text": "PRIVATE PROMPT", "token_ids": [1, 2, 3]} for role in roles},
            "results": {
                role: {"text": "PRIVATE GENERATION", "token_ids": [4], "finish_reason": "length"}
                for role in roles
            },
            "begin": {"file": "/private/node/trace.jsonl", "pid": 711, "seq": 1, "line": 2},
            "end": {"file": "/private/node/trace.jsonl", "pid": 711, "seq": 3, "line": 4},
        }
        if meta["scenario"] == "mechanism":
            case["mechanism"] = {
                "first_order": [name + "::B", name + "::C"],
                "host": "PRIVATE HOST",
            }
        cases[name] = case
    engines = {
        f"{phase}-{mode}": {
            "kv_blocks": 100,
            "kv_storage_bytes": 1024,
            "model_parameter_count": 4_000_000_000,
            "project_commit": commit,
            "smoke_direct_scheduler_object": phase == "diagnostic",
        }
        for phase in ("smoke", "diagnostic")
        for mode in m3_contract.MODES
    }
    audit = {
        "schema_version": 1,
        "task_id": m3_contract.TASK_ID,
        "scope": "complete",
        "accepted": True,
        "project_commit": commit,
        "case_count": len(cases),
        "cases": cases,
        "engines": engines,
        "batch_audit": {
            "batches": 100,
            "positive_batches": 90,
            "scheduled_tokens": 1000,
            "physical_gpu_work": "unmeasured",
        },
        "guard_lifecycle": {"diagnostic-guarded": {"registered": 20, "first_dispatch_cleared": 20}},
        "comparisons": [
            {"repeat": repeat, "native_off_exact_tokens": True, "unexpired_prefix_decision": True}
            for repeat in range(3)
        ],
        "resource_release": {"allocations": 1, "released_job_ids": ["PRIVATE JOB"]},
        "limitations": ["PRIVATE PATH /internal/account"],
    }
    return audit, sources, config


def test_projection_rebuilds_only_finite_names_numbers_and_hashes():
    audit, sources, config = projection_fixture()
    public = export_m3.public_summary(audit, sources, config)
    encoded = json.dumps(public)
    assert "PRIVATE" not in encoded
    assert "/private/" not in encoded
    assert "pid" not in encoded and "job_id" not in encoded
    assert public["case_count"] == 20
    assert public["engines"]["diagnostic-prefix"]["kv_storage_bytes"] == 1024
    assert public["cases"]["mechanism:prefix:0"]["first_order"] == ["B", "C"]


@pytest.mark.parametrize(
    "mutation", ["partial", "unaccepted", "unknown", "commit", "package", "number"]
)
def test_projection_rejects_unreviewed_or_incomplete_data(mutation):
    audit, sources, config = projection_fixture()
    if mutation == "partial":
        audit["scope"] = "partial"
    elif mutation == "unaccepted":
        audit["accepted"] = False
    elif mutation == "unknown":
        audit["private_secret"] = "cannot enter whitelist"
    elif mutation == "commit":
        sources["project_commit"] = "b" * 40
    elif mutation == "package":
        sources["packages"]["torch"] = "/private/account"
    else:
        audit["engines"]["smoke-off"]["kv_blocks"] = "private-host"
    with pytest.raises(ValueError):
        export_m3.public_summary(audit, sources, config)


def test_source_hashes_bind_to_git_content_not_manifest_labels(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    sources = {
        "project_commit": commit,
        "project_dirty": False,
        "source_sha256": {"src/example.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    assert export_m3.frozen_sources(tmp_path, sources) == sources["source_sha256"]
    changed = copy.deepcopy(sources)
    changed["source_sha256"]["src/example.py"] = "0" * 64
    with pytest.raises(ValueError, match="Git blob"):
        export_m3.frozen_sources(tmp_path, changed)


@pytest.mark.parametrize("optimize", [False, True])
def test_real_export_subprocess_fails_closed_for_partial_audit(tmp_path, optimize):
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"scope": "partial", "accepted": True}))
    output = tmp_path / "public"
    command = [sys.executable, *(["-O"] if optimize else []), str(ROOT / "scripts/export_m3.py")]
    for name, path in (
        ("config", empty),
        ("trace-dir", tmp_path),
        ("resources", empty),
        ("sources", empty),
        ("audit", audit),
        ("output", output),
    ):
        command.extend([f"--{name}", str(path)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert (
        "assertions enabled" in result.stderr if optimize else "partial or failed" in result.stderr
    )
    assert not output.exists()


def test_location_normalization_preserves_line_and_process_links():
    original = {"file": "/remote/raw.jsonl", "line": 3, "pid": 17, "seq": 6}
    relocated = {**original, "file": "/local/raw.jsonl"}
    assert export_m3.normalized_audit(original) == export_m3.normalized_audit(relocated)
    assert export_m3.normalized_audit(original) != export_m3.normalized_audit(
        {**relocated, "line": 4}
    )
