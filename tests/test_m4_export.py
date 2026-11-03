"""Independent local export binding tests; no native or GPU execution claims.

Collectors and Git blobs are real small local fixtures. Only the external
subprocess boundary of fresh-control dispatch is replaced in the dispatch tests.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler.run_evidence import check_run, collect_run

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m4_export_under_test", ROOT / "scripts/export_m4.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def frozen_repository(tmp_path):
    root = tmp_path / "source-fixture"
    root.mkdir()
    files = {
        ".gitignore": "/runs\n",
        "configs/upstream.json": '{"vllm":{"commit":"local-contract-fixture"}}\n',
        "configs/m4-stability.json": '{"scope":"local-export-test-only"}\n',
        "scripts/export_m4.py": "# Frozen local exporter fixture.\n",
        "scripts/check_m4_trace.py": "# Frozen local checker fixture.\n",
        "src/vllm_prefix_scheduler/m4_contract.py": "# Frozen local auditor fixture.\n",
        "tests/test_m4_export.py": "# Frozen local test fixture.\n",
        "Makefile": "check:\n\t@true\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=M4 local export fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "freeze local fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return root, commit


@pytest.fixture
def sealed_collector(frozen_repository, tmp_path, monkeypatch):
    root, commit = frozen_repository
    raw = tmp_path / "actual-input" / "trace.jsonl"
    raw.parent.mkdir()
    raw.write_text('{"scope":"local-original-only","value":1}\n')
    # This is explicitly kind=local. The fabricated environment ID only exercises
    # identity binding; no Slurm submission, hardware query or GPU code is called.
    job = "12345"
    monkeypatch.setenv("SLURM_JOB_ID", job)
    directory = root / "runs" / "local-collector"
    manifest, code = collect_run(
        root,
        directory,
        [sys.executable, "-c", "print('local collector binding fixture')"],
        task_id="M4-local-export-contract",
        kind="local",
        timeout=5,
        configs=[],
        artifacts=[raw],
        criteria=["Retain one actual local artifact"],
    )
    assert code == 0 and manifest["schema_version"] == 2 and manifest["kind"] == "local"
    assert check_run(directory, require_success=True)["integrity"] == "valid"
    return directory, commit, job, raw


def test_collector_binds_actual_sealed_raw_bytes(sealed_collector):
    directory, commit, job, raw = sealed_collector
    bound = exporter.bind_collector(directory, commit, job, [raw])
    assert bound[str(raw)] == exporter.sha256(raw)
    assert bound[str(directory / "manifest.json")] == exporter.sha256(directory / "manifest.json")


def test_valid_collector_cannot_back_different_same_named_raw_bytes(sealed_collector, tmp_path):
    directory, commit, job, raw = sealed_collector
    changed = tmp_path / "other-input" / raw.name
    changed.parent.mkdir()
    changed.write_text('{"scope":"different-local-input","value":999}\n')
    sealed_manifest_before = (directory / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="not the collector sealed artifact"):
        exporter.bind_collector(directory, commit, job, [changed])
    # The rejection is about detachment, not an invalid or corrupted collector.
    assert check_run(directory, require_success=True)["integrity"] == "valid"
    assert (directory / "manifest.json").read_bytes() == sealed_manifest_before
    assert raw.read_text() == '{"scope":"local-original-only","value":1}\n'


@pytest.mark.parametrize("mismatch", ["commit", "job"])
def test_valid_collector_rejects_different_freeze_or_job(sealed_collector, mismatch):
    directory, commit, job, raw = sealed_collector
    requested_commit = "f" * 40 if mismatch == "commit" else commit
    requested_job = "99999" if mismatch == "job" else job
    with pytest.raises(ValueError, match="collector freeze/job/stability differs"):
        exporter.bind_collector(directory, requested_commit, requested_job, [raw])
    assert check_run(directory, require_success=True)["integrity"] == "valid"


def test_execution_binding_reads_actual_tested_git_blobs(frozen_repository):
    root, commit = frozen_repository
    bound = exporter.bind_execution(commit, root=root)
    expected = {
        str(root / name)
        for name in (
            "configs/upstream.json",
            "configs/m4-stability.json",
            "scripts/export_m4.py",
            "scripts/check_m4_trace.py",
            "src/vllm_prefix_scheduler/m4_contract.py",
            "tests/test_m4_export.py",
            "Makefile",
        )
    }
    assert set(bound) == expected
    assert all(exporter.sha256(Path(path)) == checksum for path, checksum in bound.items())


@pytest.mark.parametrize(
    "name", ["scripts/export_m4.py", "src/vllm_prefix_scheduler/m4_contract.py"]
)
def test_execution_binding_rejects_current_exporter_or_auditor_drift(frozen_repository, name):
    root, commit = frozen_repository
    exporter.bind_execution(commit, root=root)
    (root / name).write_text("# Changed after the tested commit.\n")
    with pytest.raises(ValueError, match="currently executing source differs from tested freeze"):
        exporter.bind_execution(commit, root=root)


def test_existing_control_reports_never_replace_fresh_execution(tmp_path, monkeypatch):
    destination = tmp_path / "already-exists"
    old = destination / "native" / "results.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"all_controls_passed": True, "controls": [{}] * 17}))
    original = old.read_bytes()

    def unexpected_subprocess(*_args, **_kwargs):
        pytest.fail("Existing control destination should be rejected before executing anything")

    monkeypatch.setattr(exporter.subprocess, "run", unexpected_subprocess)
    with pytest.raises(FileExistsError):
        exporter.rerun_controls(
            tmp_path / "config.json", tmp_path / "data", tmp_path / "resources.json", destination
        )
    assert old.read_bytes() == original


def test_fresh_controls_invoke_fixed_tested_checkers_and_retain_commands(tmp_path, monkeypatch):
    config, data, resources = (
        tmp_path / "config.json",
        tmp_path / "data",
        tmp_path / "resources.json",
    )
    destination = tmp_path / "fresh-controls"
    calls = []
    expected_counts = {"native": 17, "cpu": 5, "gpu": 18, "resources": 9}

    def external_control_process(command, **kwargs):
        # Only dispatch is under test here; these reports are synthetic responses
        # from the replaced external command, not accepted control executions.
        assert kwargs == {"capture_output": True, "text": True, "timeout": 1800}
        output = Path(command[command.index("--output") + 1])
        layer = output.name
        checker = (
            "check_m4_resources_fail_closed.py"
            if layer == "resources"
            else "check_m4_fail_closed.py"
        )
        assert command[:2] == [sys.executable, str(ROOT / "scripts" / checker)]
        assert command[command.index("--config") + 1] == str(config)
        assert "--contract-fixture" not in command and "--partial" not in command
        if layer == "resources":
            assert command[command.index("--resources") + 1] == str(resources)
        else:
            assert command[command.index("--layer") + 1] == layer
            assert command[command.index("--trace") + 1] == str(data / f"{layer}-data/trace.jsonl")
            assert command[command.index("--sources") + 1] == str(
                data / f"{layer}-data" / ("sources.json" if layer == "gpu" else "runtime.json")
            )
            if layer == "gpu":
                assert command[command.index("--inputs") + 1] == str(data / "gpu-inputs.json")
                assert command[command.index("--resources") + 1] == str(resources)
        output.mkdir(parents=True, exist_ok=False)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "scope": "local-dispatch-test-only",
                    "all_controls_passed": True,
                    "controls": [{"local_test_only": True}] * expected_counts[layer],
                }
            )
        )
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="local dispatch fixture\n", stderr="")

    monkeypatch.setattr(exporter.subprocess, "run", external_control_process)
    counts, bound = exporter.rerun_controls(config, data, resources, destination)
    assert counts == expected_counts and len(calls) == 4
    for layer in expected_counts:
        record = json.loads((destination / f"{layer}-command.json").read_text())
        assert record["returncode"] == 0 and record["command"] in calls
        assert str(destination / layer / "results.json") in bound
    assert all(exporter.sha256(Path(path)) == checksum for path, checksum in bound.items())


def test_fresh_control_process_failure_is_not_accepted(tmp_path, monkeypatch):
    calls = []

    def failed_process(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 7, stdout="", stderr="local forced child failure\n"
        )

    monkeypatch.setattr(exporter.subprocess, "run", failed_process)
    destination = tmp_path / "failed-fresh-controls"
    with pytest.raises(ValueError, match="fresh fixed-source subprocess controls failed: native"):
        exporter.rerun_controls(
            tmp_path / "config.json", tmp_path / "data", tmp_path / "resources.json", destination
        )
    assert len(calls) == 1
    assert (destination / "native-stderr.log").read_text() == "local forced child failure\n"
    assert not (destination / "native" / "results.json").exists()
