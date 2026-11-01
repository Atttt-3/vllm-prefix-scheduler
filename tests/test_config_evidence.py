"""Detect config drift even when Git source fingerprints cannot see the input."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from vllm_prefix_scheduler.config_evidence import check_config_inputs
from vllm_prefix_scheduler.run_evidence import check_run, collect_run, write_json


@pytest.fixture(params=["external", "ignored"])
def configuration(tmp_path, request):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("/runs\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source = (
        tmp_path / "workload.json" if request.param == "external" else root / "runs/workload.json"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({"requests": 500}))
    if request.param == "ignored":
        subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", str(source)], check=True)
    snapshot = root / "runs/one/configs/001-workload.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    record = {
        "path": "configs/001-workload.json",
        "source": str(source.resolve()),
        "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "bytes": snapshot.stat().st_size,
    }
    return source, snapshot, record


def test_unchanged_config_passes_and_ignores_other_archived_files(configuration):
    source, snapshot, record = configuration
    checks, errors = check_config_inputs([record, {"path": "stdout.log", "sha256": "unused"}])
    assert not errors
    assert len(checks) == 1
    assert checks[0]["status"] == "unchanged"
    assert checks[0]["actual_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert checks[0]["expected_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()


def test_subprocess_config_change_is_recorded_without_changing_snapshot(configuration):
    source, snapshot, record = configuration
    original = snapshot.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
            str(source),
            json.dumps({"requests": 1}),
        ],
        check=True,
    )
    checks, errors = check_config_inputs([record])
    assert checks[0]["status"] == "changed"
    assert checks[0]["actual_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert errors and str(source) in errors[0]
    assert snapshot.read_bytes() == original


def test_deleted_config_remains_distinguishable_from_an_unchanged_input(configuration):
    source, snapshot, record = configuration
    original = snapshot.read_bytes()
    source.unlink()
    checks, errors = check_config_inputs([record])
    assert checks[0]["status"] == "missing"
    assert checks[0]["actual_sha256"] is None
    assert "FileNotFoundError" in checks[0]["error"]
    assert errors and "missing" in errors[0]
    assert snapshot.read_bytes() == original


def test_input_replaced_by_directory_is_reported_as_unreadable(configuration):
    source, snapshot, record = configuration
    original = snapshot.read_bytes()
    source.unlink()
    source.mkdir()
    checks, errors = check_config_inputs([record])
    assert checks[0]["status"] == "unreadable"
    assert checks[0]["actual_sha256"] is None
    assert "IsADirectoryError" in checks[0]["error"]
    assert errors and "unreadable" in errors[0]
    assert snapshot.read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="POSIX special-file configuration input")
def test_input_retargeted_to_device_is_reported_as_unreadable(configuration):
    source, snapshot, record = configuration
    original = snapshot.read_bytes()
    source.unlink()
    source.symlink_to(os.devnull)
    checks, errors = check_config_inputs([record])
    assert checks[0]["status"] == "unreadable"
    assert checks[0]["actual_sha256"] is None
    assert "not a regular file" in checks[0]["error"]
    assert errors and "unreadable" in errors[0]
    assert snapshot.read_bytes() == original


def test_empty_config_registry_cannot_be_reported_as_checked():
    checks, errors = check_config_inputs([{"path": "stdout.log"}])
    assert not checks
    assert errors


def test_relative_source_is_not_resolved_against_the_reviewers_working_directory(configuration):
    _, snapshot, record = configuration
    original = snapshot.read_bytes()
    record["source"] = "workload.json"
    checks, errors = check_config_inputs([record])
    assert checks[0]["status"] == "unreadable"
    assert "absolute path" in checks[0]["error"]
    assert errors
    assert snapshot.read_bytes() == original


@pytest.fixture(params=["external", "ignored"])
def config_run_repository(tmp_path, request):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "configs/upstream.json").write_text('{"vllm":{"commit":"fixture-only"}}')
    (root / ".gitignore").write_text("/runs\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "configuration evidence fixture",
        ],
        check=True,
    )
    source = (
        tmp_path / "workload.json" if request.param == "external" else root / "runs/workload.json"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({"requests": 500}))
    if request.param == "ignored":
        subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", str(source)], check=True)
    return root, source


def collect_config_command(root, source, snippet, *arguments):
    directory = root / "runs/collected"
    manifest, code = collect_run(
        root,
        directory,
        [sys.executable, "-c", snippet, str(source), *map(str, arguments)],
        task_id="config-input-regression",
        kind="local",
        timeout=5,
        configs=[source],
        artifacts=[],
        criteria=["Record the actual command observation"],
    )
    return directory, manifest, code


def collect_config_subprocess(root, source, snippet, *arguments):
    """Bound the entire collector, including its final configuration recheck."""
    program = """
from pathlib import Path
import sys
from vllm_prefix_scheduler.run_evidence import collect_run

root, source = map(Path, sys.argv[1:3])
_, code = collect_run(
    root, root / "runs/collected",
    [sys.executable, "-c", sys.argv[3], str(source), *sys.argv[4:]],
    task_id="config-special-file-regression", kind="local", timeout=1,
    configs=[source], artifacts=[], criteria=["Record the actual command observation"],
)
raise SystemExit(code)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(root), str(source), snippet, *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
    assert process.returncode == 1, (stdout, stderr)
    directory = root / "runs/collected"
    return directory, json.loads((directory / "manifest.json").read_text())


def mark_config_run_accepted(directory):
    receipt = json.loads((directory / "acceptance.json").read_text())
    receipt.update(decision="accepted", reviewer="configuration fixture reviewer")
    receipt["criteria"][0].update(status="passed", evidence=["stdout.log"])
    write_json(directory / "acceptance.json", receipt)


@pytest.mark.parametrize("operation, status", [("change", "changed"), ("delete", "missing")])
def test_config_drift_blocks_collector_and_review_gates(config_run_repository, operation, status):
    root, source = config_run_repository
    original = source.read_bytes()
    snippet = (
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.write_text(sys.argv[2]) if sys.argv[3] == 'change' else p.unlink(); "
        "print(p.read_text() if p.exists() else 'configuration deleted')"
    )
    directory, manifest, code = collect_config_command(
        root, source, snippet, json.dumps({"requests": 1}), operation
    )
    assert code != 0
    assert manifest["command_status"] == "succeeded"
    assert manifest["source_changed_during_run"] is False
    assert manifest["config_inputs_stable"] is False
    record = next(item for item in manifest["files"] if item.get("source") == str(source))
    assert (directory / record["path"]).read_bytes() == original
    config_check = next(item for item in manifest["config_checks"] if item["source"] == str(source))
    assert config_check["status"] == status
    assert any(str(source) in error for error in manifest["collection_errors"])
    assert check_run(directory)["integrity"] == "valid"
    with pytest.raises(ValueError, match="success gate"):
        check_run(directory, require_success=True)
    mark_config_run_accepted(directory)
    with pytest.raises(ValueError, match="stable configuration inputs"):
        check_run(directory, require_accepted=True)


def test_unchanged_external_or_ignored_config_can_be_accepted(config_run_repository):
    root, source = config_run_repository
    directory, manifest, code = collect_config_command(
        root, source, "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())"
    )
    assert code == 0
    assert manifest["config_inputs_stable"] is True
    assert manifest["source_changed_during_run"] is False
    assert all(item["status"] == "unchanged" for item in manifest["config_checks"])
    assert check_run(directory, require_success=True)["integrity"] == "valid"
    mark_config_run_accepted(directory)
    assert check_run(directory, require_accepted=True)["acceptance"] == "accepted"


def test_registered_symlink_is_rechecked_after_retargeting(config_run_repository):
    root, source = config_run_repository
    original = source.read_bytes()
    first_target = source.parent / "workload-first.json"
    second_target = source.parent / "workload-second.json"
    first_target.write_bytes(original)
    second_target.write_text(json.dumps({"requests": 1}))
    source.unlink()
    source.symlink_to(first_target)
    snippet = (
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.unlink(); p.symlink_to(sys.argv[2]); print(p.read_text())"
    )
    directory, manifest, code = collect_config_command(root, source, snippet, second_target)
    assert code != 0
    assert manifest["config_inputs_stable"] is False
    assert manifest["source_changed_during_run"] is False
    assert first_target.read_bytes() == original
    config_check = next(item for item in manifest["config_checks"] if item["source"] == str(source))
    assert config_check["status"] == "changed"
    assert (directory / config_check["snapshot_path"]).read_bytes() == original
    assert json.loads((directory / "stdout.log").read_text())["requests"] == 1
    assert check_run(directory)["integrity"] == "valid"
    mark_config_run_accepted(directory)
    with pytest.raises(ValueError, match="stable configuration inputs"):
        check_run(directory, require_accepted=True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO configuration input")
@pytest.mark.parametrize("operation", ["replace", "retarget"])
def test_fifo_configuration_is_sealed_without_waiting_for_a_writer(
    config_run_repository, operation
):
    root, source = config_run_repository
    original = source.read_bytes()
    fifo = source.parent / "workload-fifo"
    if operation == "retarget":
        first_target = source.parent / "workload-first.json"
        first_target.write_bytes(original)
        source.unlink()
        source.symlink_to(first_target)
        os.mkfifo(fifo)
    snippet = (
        "from pathlib import Path; import os, sys; p=Path(sys.argv[1]); p.unlink(); "
        "os.mkfifo(p) if sys.argv[2] == 'replace' else p.symlink_to(sys.argv[3]); "
        "print('configuration replaced with FIFO')"
    )
    directory, manifest = collect_config_subprocess(root, source, snippet, operation, fifo)
    assert manifest["command_status"] == "succeeded"
    assert manifest["returncode"] == 0
    assert manifest["source_changed_during_run"] is False
    assert manifest["config_inputs_stable"] is False
    config_check = next(item for item in manifest["config_checks"] if item["source"] == str(source))
    assert config_check["status"] == "unreadable"
    assert config_check["actual_sha256"] is None
    assert "not a regular file" in config_check["error"]
    assert any(str(source) in error for error in manifest["collection_errors"])
    assert (directory / config_check["snapshot_path"]).read_bytes() == original
    assert (directory / "stdout.log").read_text().strip() == "configuration replaced with FIFO"
    assert check_run(directory)["integrity"] == "valid"
    with pytest.raises(ValueError, match="success gate"):
        check_run(directory, require_success=True)
    mark_config_run_accepted(directory)
    with pytest.raises(ValueError, match="stable configuration inputs"):
        check_run(directory, require_accepted=True)
