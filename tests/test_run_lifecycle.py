"""Real collector signals during execution, cleanup and evidence sealing."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from vllm_prefix_scheduler.run_evidence import check_run, collect_run, write_json

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process and signal lifecycle")
SOURCE = Path(__file__).resolve().parents[1] / "src"

# Markers synchronize external signals with actual command/collector phases. These
# wrappers only hold a phase open; assertions inspect the completed public contract.
COLLECTOR = """
import sys, time
from pathlib import Path
import vllm_prefix_scheduler.run_evidence as evidence

root = Path(sys.argv[1])
phase = sys.argv[2]
control = root / 'runs/control'
command = sys.argv[3]

def hold_for_signal():
    (control / 'seal-entered').write_text('ready')
    deadline = time.monotonic() + 8
    while not (control / 'seal-release').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('External signal driver did not release the sealing phase')
        time.sleep(.01)

if phase == 'artifact':
    original = evidence.snapshot
    def held_snapshot(directory, source, group, index):
        if group == 'artifacts':
            hold_for_signal()
        return original(directory, source, group, index)
    evidence.snapshot = held_snapshot
elif phase == 'manifest':
    original = evidence.write_json
    held = False
    def held_write(path, value):
        global held
        if path.name == 'manifest.json' and value.get('command_status') != 'running' and not held:
            held = True
            hold_for_signal()
        return original(path, value)
    evidence.write_json = held_write

manifest, code = evidence.collect_run(
    root, root / 'runs/record', [sys.executable, '-c', command, str(control)],
    task_id='signal-lifecycle', kind='local', timeout=.2 if phase == 'timeout' else 20,
    configs=[], artifacts=[control / 'artifact.json'] if phase == 'artifact' else [],
    criteria=['The command and its process group terminated before evidence was sealed'],
)
raise SystemExit(code)
"""

SLEEPING_COMMAND = """
import os, signal, sys, time
from pathlib import Path
control = Path(sys.argv[1])
def ignore_term(signum, frame):
    (control / 'child-term-seen').write_text('TERM received; still alive')
signal.signal(signal.SIGTERM, ignore_term)
(control / 'child-pid').write_text(str(os.getpid()))
while True:
    time.sleep(.1)
"""

FINISHING_COMMAND = """
import os, sys
from pathlib import Path
(Path(sys.argv[1]) / 'child-pid').write_text(str(os.getpid()))
print('Command finished before sealing')
"""


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    (root / "configs").mkdir(parents=True)
    (root / "configs/upstream.json").write_text('{"vllm": {"commit": "lifecycle-fixture"}}')
    (root / ".gitignore").write_text("/runs\n")
    for argv in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "add", "."],
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
            "fixture",
        ],
    ):
        subprocess.run(argv, check=True, capture_output=True)
    control = root / "runs/control"
    control.mkdir(parents=True)
    (control / "artifact.json").write_text('{"result": "completed command artifact"}')
    return root


@contextmanager
def collector(root, phase, command=SLEEPING_COMMAND):
    process = subprocess.Popen(
        [sys.executable, "-c", COLLECTOR, str(root), phase, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        pid_path = root / "runs/control/child-pid"
        if pid_path.exists():
            try:
                os.killpg(int(pid_path.read_text()), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def wait_for_marker(root, name, process):
    marker = root / "runs/control" / name
    deadline = time.monotonic() + 8
    while not marker.exists() or not marker.read_text():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Collector exited before {name}: {stdout}\n{stderr}")
        if time.monotonic() > deadline:
            pytest.fail(f"Collector did not reach {name}")
        time.sleep(0.01)
    return marker


def assert_interrupted_record(root, process, expected_signal):
    stdout, stderr = process.communicate(timeout=8)
    assert process.returncode == 128 + expected_signal, (stdout, stderr)
    pid = int((root / "runs/control/child-pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    directory = root / "runs/record"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["command_status"] == "interrupted"
    assert manifest["termination_signal"] == expected_signal
    assert (directory / "acceptance.json").is_file()
    assert (directory / "report.md").is_file()
    assert check_run(directory)["integrity"] == "valid"
    with pytest.raises(ValueError, match="success gate"):
        check_run(directory, require_success=True)
    # A reviewer cannot convert an interrupted command into accepted evidence.
    receipt = json.loads((directory / "acceptance.json").read_text())
    receipt.update(decision="accepted", reviewer="lifecycle fixture reviewer")
    for criterion in receipt["criteria"]:
        criterion.update(status="passed", evidence=["stdout.log"])
    write_json(directory / "acceptance.json", receipt)
    with pytest.raises(ValueError, match="Acceptance requires"):
        check_run(directory, require_accepted=True)


def test_sigterm_during_timeout_cleanup_still_terminates_and_seals(repository):
    with collector(repository, "timeout") as process:
        # The marker is written by the still-running command's TERM handler,
        # proving the collector has entered its timeout cleanup, not normal wait.
        wait_for_marker(repository, "child-term-seen", process)
        process.send_signal(signal.SIGTERM)
        assert_interrupted_record(repository, process, signal.SIGTERM)


@pytest.mark.parametrize(
    "first,second",
    [
        (signal.SIGINT, signal.SIGINT),
        (signal.SIGTERM, signal.SIGTERM),
        (signal.SIGINT, signal.SIGTERM),
        (signal.SIGTERM, signal.SIGINT),
    ],
)
def test_repeated_signals_during_cleanup_cannot_leave_the_command_running(
    repository, first, second
):
    with collector(repository, "running") as process:
        wait_for_marker(repository, "child-pid", process)
        process.send_signal(first)
        wait_for_marker(repository, "child-term-seen", process)
        process.send_signal(second)
        assert_interrupted_record(repository, process, first)


@pytest.mark.parametrize("phase", ["artifact", "manifest"])
@pytest.mark.parametrize("interrupt_signal", [signal.SIGINT, signal.SIGTERM])
def test_signal_after_command_exit_is_reflected_in_sealed_evidence(
    repository, phase, interrupt_signal
):
    with collector(repository, phase, FINISHING_COMMAND) as process:
        wait_for_marker(repository, "seal-entered", process)
        process.send_signal(interrupt_signal)
        (repository / "runs/control/seal-release").write_text("signal sent")
        assert_interrupted_record(repository, process, interrupt_signal)


def test_collect_run_restores_both_existing_signal_handlers(repository):
    def original_interrupt(_signum, _frame):
        raise RuntimeError("Caller handler must be restored, not executed by this test")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, original_interrupt)
        manifest, code = collect_run(
            repository,
            repository / "runs/record",
            [sys.executable, "-c", "print('ordinary completed command')"],
            task_id="handler-restoration",
            kind="local",
            timeout=3,
            configs=[],
            artifacts=[],
            criteria=[],
        )
        assert code == 0 and manifest["command_status"] == "succeeded"
        assert check_run(repository / "runs/record")["integrity"] == "valid"
        for sig in previous:
            assert signal.getsignal(sig) is original_interrupt
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
