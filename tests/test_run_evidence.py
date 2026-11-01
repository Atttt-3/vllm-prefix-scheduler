"""Exercise real subprocesses and evidence corruption; no native/GPU claims are made."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vllm_prefix_scheduler.run_evidence import check_run, collect_run, digest, write_json


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "configs/upstream.json").write_text('{"vllm": {"commit": "fixture-only"}}')
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
            "test fixture",
        ],
        check=True,
    )
    return root


def run(root, command, **kwargs):
    return collect_run(
        root,
        root / "runs/one",
        command,
        task_id="workflow-test",
        kind=kwargs.pop("kind", "local"),
        timeout=kwargs.pop("timeout", 5),
        configs=[],
        artifacts=kwargs.pop("artifacts", []),
        criteria=["Observe the expected command output"],
        **kwargs,
    )


def test_success_preserves_exact_argv_snapshots_and_pending_acceptance(repository):
    argument = "spaces; $(touch should-not-exist) `literal` 中文"
    command = [sys.executable, "-c", "import sys; print(sys.argv[1])", argument]
    manifest, code = run(repository, command, seed=7)
    directory = repository / "runs/one"
    assert code == 0
    assert manifest["argv"] == command
    assert manifest["seed"] == 7
    assert manifest["source_changed_during_run"] is False
    assert not (repository / "should-not-exist").exists()
    assert (directory / "stdout.log").read_text().strip() == argument
    (repository / "configs/upstream.json").write_text("changed after the run")
    assert json.loads((directory / "configs/000-upstream.json").read_text())["vllm"]
    assert check_run(directory)["acceptance"] == "pending"
    with pytest.raises(ValueError, match="not been accepted"):
        check_run(directory, require_accepted=True)
    before = (directory / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        run(repository, command)
    assert (directory / "manifest.json").read_bytes() == before


def test_failure_and_launch_failure_retain_logs_and_fail_only_the_success_gate(repository):
    manifest, code = run(
        repository, [sys.executable, "-c", "import sys; print('negative result'); sys.exit(9)"]
    )
    directory = repository / "runs/one"
    assert code == 9 and manifest["command_status"] == "failed"
    assert check_run(directory)["integrity"] == "valid"
    with pytest.raises(ValueError, match="success gate"):
        check_run(directory, require_success=True)
    second = repository / "runs/missing"
    manifest, code = collect_run(
        repository,
        second,
        [str(repository / "missing-executable")],
        task_id="launch",
        kind="local",
        timeout=2,
        configs=[],
        artifacts=[],
        criteria=[],
    )
    assert code == 127 and manifest["command_status"] == "launch_failed"
    assert (second / "stderr.log").read_text()
    assert check_run(second)["command_status"] == "launch_failed"


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup is POSIX-specific")
def test_timeout_stops_command_and_its_descendant(repository):
    child = (
        "import time; from pathlib import Path; time.sleep(.8); Path('leaked').write_text('bad')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "print('started', flush=True); time.sleep(10)"
    )
    manifest, code = run(repository, [sys.executable, "-c", parent], timeout=0.3)
    assert code == 124 and manifest["command_status"] == "timed_out"
    assert check_run(repository / "runs/one")["command_status"] == "timed_out"
    time.sleep(0.9)
    assert not (repository / "leaked").exists()


@pytest.mark.skipif(os.name != "posix", reason="Signal interruption is POSIX-specific")
@pytest.mark.parametrize("interrupt_signal", [signal.SIGINT, signal.SIGTERM])
def test_interruption_preserves_a_finished_record(repository, interrupt_signal):
    source = Path(__file__).resolve().parents[1] / "src"
    snippet = (
        "from pathlib import Path; import sys; "
        "from vllm_prefix_scheduler.run_evidence import collect_run; "
        "root=Path(sys.argv[1]); "
        "collect_run(root, root/'runs/one', "
        "[sys.executable, '-c', 'import time; print(123, flush=True); time.sleep(20)'], "
        "task_id='interrupt', kind='local', timeout=30, configs=[], artifacts=[], criteria=[])"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", snippet, str(repository)],
        env={**os.environ, "PYTHONPATH": str(source)},
    )
    log = repository / "runs/one/stdout.log"
    try:
        deadline = time.monotonic() + 5
        while not log.exists() or not log.read_text():
            if time.monotonic() > deadline:
                pytest.fail("Child command did not start")
            time.sleep(0.02)
        process.send_signal(interrupt_signal)
        process.wait(timeout=5)
        assert check_run(repository / "runs/one")["command_status"] == "interrupted"
        manifest = json.loads((repository / "runs/one/manifest.json").read_text())
        assert manifest["termination_signal"] == interrupt_signal
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize(
    "corruption", ["log", "manifest", "traversal", "skipped", "no_criteria", "replaced_criterion"]
)
def test_corrupt_evidence_and_unsupported_acceptance_are_rejected(repository, corruption):
    run(repository, [sys.executable, "-c", "print('actual output')"])
    directory = repository / "runs/one"
    receipt = json.loads((directory / "acceptance.json").read_text())
    receipt.update(decision="accepted", reviewer="fixture reviewer")
    receipt["criteria"][0].update(status="passed", evidence=["stdout.log"])
    if corruption == "log":
        (directory / "stdout.log").write_text("tampered")
    elif corruption == "manifest":
        with (directory / "manifest.json").open("a") as stream:
            stream.write("\n")
    elif corruption == "traversal":
        receipt["criteria"][0]["evidence"] = ["../../configs/upstream.json"]
    elif corruption == "skipped":
        receipt["criteria"][0]["status"] = "skipped"
    elif corruption == "replaced_criterion":
        receipt["criteria"][0]["description"] = "Unrelated replacement standard"
    else:
        receipt["criteria"] = []
    write_json(directory / "acceptance.json", receipt)
    with pytest.raises(ValueError):
        check_run(directory, require_accepted=True)


def test_review_can_accept_archived_evidence_but_missing_artifacts_cannot(repository):
    run(repository, [sys.executable, "-c", "print('expected')"])
    directory = repository / "runs/one"
    receipt = json.loads((directory / "acceptance.json").read_text())
    receipt.update(decision="accepted", reviewer="fixture reviewer")
    receipt["criteria"][0].update(status="passed", evidence=["stdout.log"])
    write_json(directory / "acceptance.json", receipt)
    assert check_run(directory, require_accepted=True)["acceptance"] == "accepted"
    manifest, code = collect_run(
        repository,
        repository / "runs/missing-artifact",
        [sys.executable, "-c", "print('ok')"],
        task_id="artifact",
        kind="local",
        timeout=3,
        configs=[],
        artifacts=[Path("missing-trace.jsonl")],
        criteria=[],
    )
    assert code == 1 and manifest["collection_errors"]
    assert check_run(repository / "runs/missing-artifact")["command_status"] == "succeeded"
    with pytest.raises(ValueError, match="success gate"):
        check_run(repository / "runs/missing-artifact", require_success=True)


def test_gpu_kind_requires_an_existing_allocation(repository, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="existing Slurm allocation"):
        run(repository, [sys.executable, "-c", "print('must not execute')"], kind="gpu")
    assert not (repository / "runs/one").exists()


def test_changed_sources_are_recorded_and_cannot_be_accepted(repository):
    manifest, _ = run(
        repository,
        [sys.executable, "-c", "from pathlib import Path; Path('changed.py').write_text('# new')"],
    )
    assert manifest["source_changed_during_run"] is True
    directory = repository / "runs/one"
    receipt = json.loads((directory / "acceptance.json").read_text())
    receipt.update(decision="accepted", reviewer="fixture reviewer")
    receipt["criteria"][0].update(status="passed", evidence=["stdout.log"])
    write_json(directory / "acceptance.json", receipt)
    with pytest.raises(ValueError, match="stable sources"):
        check_run(directory, require_accepted=True)


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup is POSIX-specific")
def test_successful_parent_cannot_leave_a_background_writer(repository):
    child = "import time; time.sleep(.8); print('late log', flush=True)"
    parent = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
    manifest, code = run(repository, [sys.executable, "-c", parent])
    assert code == 1
    assert manifest["command_status"] == "succeeded"
    assert manifest["collection_errors"]
    before = (repository / "runs/one/stdout.log").read_bytes()
    time.sleep(0.9)
    assert (repository / "runs/one/stdout.log").read_bytes() == before
    assert check_run(repository / "runs/one")["integrity"] == "valid"
    with pytest.raises(ValueError, match="success gate"):
        check_run(repository / "runs/one", require_success=True)


def test_private_logs_cannot_be_collected_in_public_evidence_directory(repository):
    with pytest.raises(ValueError, match="must stay under runs"):
        collect_run(
            repository,
            repository / "evidence/private",
            [sys.executable, "-c", "print('secret')"],
            task_id="private",
            kind="local",
            timeout=1,
            configs=[],
            artifacts=[],
            criteria=[],
        )


@pytest.mark.skipif(os.name != "posix", reason="Process-group probing is POSIX-specific")
def test_process_group_permission_error_is_archived_and_blocks_acceptance(repository, monkeypatch):
    def denied(_pid, _signal):
        raise PermissionError("fixture group permission denied")

    monkeypatch.setattr(os, "killpg", denied)
    manifest, code = run(repository, [sys.executable, "-c", "print('parent complete')"])
    assert code == 1
    assert any("cleanup could not be confirmed" in error for error in manifest["collection_errors"])
    assert check_run(repository / "runs/one")["integrity"] == "valid"
    with pytest.raises(ValueError, match="success gate"):
        check_run(repository / "runs/one", require_success=True)


def test_legacy_review_is_readable_but_cannot_pass_the_new_configuration_gate(repository):
    run(repository, [sys.executable, "-c", "print('legacy input fixture')"])
    directory = repository / "runs/one"
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["schema_version"] = 1
    manifest.pop("config_checks")
    manifest.pop("config_inputs_stable")
    write_json(directory / "manifest.json", manifest)
    receipt = json.loads((directory / "acceptance.json").read_text())
    receipt.update(
        manifest_sha256=digest(directory / "manifest.json"),
        decision="accepted",
        reviewer="legacy fixture reviewer",
    )
    receipt["criteria"][0].update(status="passed", evidence=["stdout.log"])
    write_json(directory / "acceptance.json", receipt)
    checked = check_run(directory)
    assert checked["integrity"] == "valid"
    assert checked["acceptance"] == "accepted"
    assert checked["configuration_verification"] == "not_verified"
    with pytest.raises(ValueError, match="verified stable configuration"):
        check_run(directory, require_accepted=True)
