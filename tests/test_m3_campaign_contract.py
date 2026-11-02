"""Campaign combination tests use labelled synthetic evidence, never real GPU proof."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from m3_campaign_fixture import build_campaign_fixture, write

from vllm_prefix_scheduler import m3_campaign_contract as contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "campaign_controls", ROOT / "scripts/check_m3_campaign_fail_closed.py"
)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


@pytest.fixture
def campaign(tmp_path):
    return build_campaign_fixture(tmp_path)


def seams(monkeypatch):
    # These seams test composition; genuine Git/native verification has no CLI bypass.
    monkeypatch.setattr(contract, "frozen_sources", lambda root, sources: sources["source_sha256"])
    monkeypatch.setattr(
        contract,
        "native_raw_audit",
        lambda root, kind, trace: {
            "cases": 432 if kind == "compat" else 12,
            "raw_audit_sha256": "f" * 64,
        },
    )


def test_complete_synthetic_combination_preserves_blocked_history(campaign, monkeypatch):
    seams(monkeypatch)
    result = contract.audit_campaign(**campaign["paths"])
    assert result["campaign_acceptance"] == "passed"
    assert result["gpu_audit"]["case_count"] == 20
    assert result["original_full_history_gate"] == "blocked"
    assert result["history_terminal_unverifiable"] is True
    assert result["budget"] == {
        "historical_allocations": 4,
        "new_allocations": 1,
        "total_allocations": 5,
        "cumulative_limit": 8,
    }
    assert result["native"]["compat_cases"] == 432 and result["native"]["m1_cases"] == 12
    assert len(result["provenance"]["artifact_inventory"]) >= 33


WRAPPER = """
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + '/src')
from vllm_prefix_scheduler import m3_campaign_contract as contract
# Explicit SYNTHETIC composition seams; never used by production campaign CLI.
contract.frozen_sources = lambda root,sources:sources['source_sha256']
contract.native_raw_audit = lambda root,kind,trace:{'cases':432 if kind=='compat' else 12,'raw_audit_sha256':'f'*64}
args = {k:Path(v) for k,v in json.loads(Path(sys.argv[2]).read_text()).items()}
result=contract.audit_campaign(**args)
print(json.dumps({'campaign':result['campaign_acceptance'],'history':result['original_full_history_gate']}))
"""


def subprocess_fixture(paths, directory, flags=()):
    arguments = directory / "arguments.json"
    write(arguments, {key: str(value) for key, value in paths.items()})
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    return subprocess.run(
        [sys.executable, *flags, "-c", WRAPPER, str(ROOT), str(arguments)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def test_real_subprocess_synthetic_combination_positive(campaign, tmp_path):
    result = subprocess_fixture(campaign["paths"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"campaign": "passed", "history": "blocked"}


@pytest.mark.parametrize("label", controls.CONTROLS)
def test_campaign_extra_mutations_reject_specific_gate_in_real_subprocess(
    campaign, tmp_path, label
):
    directory = tmp_path / label
    directory.mkdir()
    paths, flags = controls.mutate(label, campaign["paths"], directory)
    result = subprocess_fixture(paths, directory, flags)
    assert result.returncode != 0
    assert controls.CONTROLS[label] in result.stderr, result.stderr


def test_native_raw_checker_is_really_invoked_and_rejects_synthetic_trace(campaign):
    for kind in ("compat", "m1"):
        path = Path(campaign["native"][kind]["trace"]["path"])
        with pytest.raises(ValueError, match=f"{kind} independent raw native audit failed"):
            contract.native_raw_audit(ROOT, kind, path)


def test_missing_history_row_cannot_be_relabelled_as_hash_valid(campaign):
    plan = copy.deepcopy(campaign["plan"])
    path = campaign["paths"]["history_index"]
    history = json.loads(path.read_text())
    history["allocations"].pop(2)
    plan["history"]["index_sha256"] = write(path, history)["sha256"]
    with pytest.raises(ValueError, match="historical allocation catalog differs"):
        contract.history_gate(plan, path)


def test_changed_indirect_artifact_inventory_detects_raw_change(campaign):
    paths = [campaign["paths"][key] for key in ("history_index", "resources", "native_index")]
    before = contract.artifact_inventory(paths)
    raw = Path(campaign["resources"]["allocations"][0]["evidence"]["exit"]["path"])
    value = json.loads(raw.read_text())
    value["gpu"] = 9
    write(raw, value)
    assert before != contract.artifact_inventory(paths)


def test_subprocess_cli_optimized_rejects_before_missing_inputs(tmp_path):
    command = [sys.executable, "-O", str(ROOT / "scripts/check_m3_campaign.py")]
    for name in (
        "campaign-config",
        "authorization",
        "history-index",
        "config",
        "inputs",
        "trace-dir",
        "resources",
        "native-index",
    ):
        command += ["--" + name, str(tmp_path / "missing")]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "requires assertions" in result.stderr


def test_monitor_originals_are_in_recursive_stability_inventory(campaign):
    paths = [campaign["paths"][key] for key in ("history_index", "resources", "native_index")]
    inventory = contract.artifact_inventory(paths)
    ledger = json.loads(Path(campaign["resources"]["submission_ledger"]["path"]).read_text())
    attempt = ledger["attempts"][0]
    observed = {item["path"] for item in inventory}
    for role in ("monitor_ready", "monitor_trace", "monitor_exit"):
        assert str(Path(attempt[role]["path"]).resolve()) in observed
    watch = Path(attempt["monitor_trace"]["path"])
    watch.write_text(watch.read_text() + "\n")
    assert inventory != contract.artifact_inventory(paths)


@pytest.mark.parametrize("kind", ("compat", "m1"))
def test_native_model_metadata_bound_to_real_gpu_config(campaign, monkeypatch, kind):
    seams(monkeypatch)
    entry = campaign["native"][kind]
    path = Path(entry["runtime"]["path"])
    runtime = json.loads(path.read_text())
    if kind == "compat":
        runtime["model_config_sha256"] = "d" * 64
    else:
        runtime["model_config"]["sha256"] = "d" * 64
    entry["runtime"] = write(path, runtime)
    collector = Path(entry["collector_dir"])
    manifest = json.loads((collector / "manifest.json").read_text())
    record = next(r for r in manifest["files"] if r["path"] == "artifacts/runtime.json")
    record.update(sha256=entry["runtime"]["sha256"], bytes=path.stat().st_size)
    manifest_ref = write(collector / "manifest.json", manifest)
    acceptance = json.loads((collector / "acceptance.json").read_text())
    acceptance["manifest_sha256"] = manifest_ref["sha256"]
    write(collector / "acceptance.json", acceptance)
    write(campaign["paths"]["native_index"], campaign["native"])
    with pytest.raises(ValueError, match=f"{kind} model metadata differs from real GPU model"):
        contract.audit_campaign(**campaign["paths"])


def test_failed_submission_remains_in_inventory_without_consuming_allocation(campaign, monkeypatch):
    seams(monkeypatch)
    catalog = Path(campaign["resources"]["submission_catalog_dir"])
    failed = catalog / "rejected-submit/submission.json"
    write(
        failed,
        {
            "command": ["sbatch", "synthetic"],
            "returncode": 1,
            "stdout": "",
            "stderr": "synthetic rejection",
        },
    )
    result = contract.audit_campaign(**campaign["paths"])
    assert result["budget"]["new_allocations"] == 1
    assert str(failed) in {r["path"] for r in result["provenance"]["artifact_inventory"]}


def test_submission_catalog_symlink_is_rejected(campaign, monkeypatch):
    seams(monkeypatch)
    catalog = Path(campaign["resources"]["submission_catalog_dir"])
    linked = catalog / "linked-attempt"
    linked.symlink_to(catalog / "attempt-01", target_is_directory=True)
    with pytest.raises(ValueError, match="submission receipt catalog contains symlink"):
        contract.audit_campaign(**campaign["paths"])
