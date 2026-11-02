"""Real local subprocess controls with fake SSH/GPU commands; no remote/GPU work."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def wait_file(path, process, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return
        if process.poll() is not None:
            raise AssertionError(f"subprocess exited before {path.name}: {process.communicate()}")
        time.sleep(0.01)
    raise AssertionError(f"bounded wait expired: {path.name}")


def start_watcher(tmp_path, scenario, timeout=3):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "ssh"
    fake.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
path=pathlib.Path(os.environ['M3_FAKE_SSH_LOG'])
prior=path.read_text().splitlines() if path.exists() else []
command=sys.argv[-1]
with path.open('a') as stream: stream.write(json.dumps(command)+'\\n')
mode=os.environ['M3_FAKE_SSH_MODE']
if command.startswith('scontrol show job '):
    count=sum('scontrol show job ' in line for line in prior)
    state='RUNNING' if mode=='running_then_terminal' and count==0 else 'COMPLETED'
    print('JobId=43210 JobState='+state+' ExitCode=0:0')
    if mode=='terminal_query_failure':
        print('synthetic query failed despite stale stdout',file=sys.stderr)
        sys.exit(255)
elif command=='squeue -h -o "%i|%T|%j"':
    if mode=='active_query_failure':
        print('synthetic active query failed',file=sys.stderr)
        sys.exit(255)
else:
    raise RuntimeError('unexpected command: '+command)
""")
    fake.chmod(0o700)
    output, submission = tmp_path / "watch", tmp_path / "submission.json"
    log = tmp_path / "ssh-calls.jsonl"
    env = {
        **os.environ,
        "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", ""),
        "M3_FAKE_SSH_LOG": str(log),
        "M3_FAKE_SSH_MODE": scenario,
    }
    command = [
        sys.executable,
        str(ROOT / "scripts/watch_m3_campaign.py"),
        "--ssh-alias",
        "synthetic-never-contact-host",
        "--campaign-id",
        "m3-campaign-fixture",
        "--campaign-sha256",
        "a" * 64,
        "--attempt-id",
        "attempt-01",
        "--project-commit",
        "b" * 40,
        "--clock-domain",
        "monotonic_ns:test-controller",
        "--output",
        str(output),
        "--submission",
        str(submission),
        "--timeout",
        str(timeout),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    wait_file(output / "monitor-ready.json", process)
    return process, output, submission, log


def submit_fixture(path, returncode=0):
    value = {
        "returncode": returncode,
        "stdout": "43210;fixture-cluster\n",
        "stderr": "",
        "observed_monotonic_ns": time.monotonic_ns(),
    }
    path.write_text(json.dumps(value))
    return value


def test_watcher_ready_precedes_submission_then_preserves_queries_terminal_and_full_queue(tmp_path):
    process, output, submission, log = start_watcher(tmp_path, "running_then_terminal")
    try:
        ready = json.loads((output / "monitor-ready.json").read_text())
        assert ready["ready"] is True and ready["pid"] == process.pid
        assert not submission.exists() and not log.exists()
        receipt = submit_fixture(submission)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stdout + stderr
        assert ready["observed_monotonic_ns"] < receipt["observed_monotonic_ns"]
        trace = [json.loads(line) for line in (output / "watch.jsonl").read_text().splitlines()]
        assert len(trace) == 2
        assert "RUNNING" in trace[0]["stdout"] and "COMPLETED" in trace[1]["stdout"]
        terminal = json.loads((output / "slurm-final.json").read_text())
        active = json.loads((output / "active-final.json").read_text())
        assert terminal == trace[-1]
        assert active["returncode"] == 0 and active["stdout"] == ""
        assert active["command"][-1] == 'squeue -h -o "%i|%T|%j"'
        assert active["observed_monotonic_ns"] >= terminal["observed_monotonic_ns"]
        assert active["clock_domain"] == ready["clock_domain"]
        assert "WATCH_READY" in stdout and "TERMINAL_CAPTURED COMPLETED" in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_failed_terminal_query_cannot_accept_stale_completed_stdout(tmp_path):
    process, output, submission, _ = start_watcher(tmp_path, "terminal_query_failure", timeout=1)
    submit_fixture(submission)
    stdout, stderr = process.communicate(timeout=6)
    assert process.returncode == 2, stdout + stderr
    rows = [json.loads(line) for line in (output / "watch.jsonl").read_text().splitlines()]
    assert rows and all(row["returncode"] == 255 for row in rows)
    assert not (output / "slurm-final.json").exists()
    assert not (output / "active-final.json").exists()
    assert "TERMINAL_CAPTURED" not in stdout


def test_failed_final_queue_query_retains_terminal_but_returns_failure(tmp_path):
    process, output, submission, _ = start_watcher(tmp_path, "active_query_failure")
    submit_fixture(submission)
    stdout, stderr = process.communicate(timeout=6)
    assert process.returncode == 4, stdout + stderr
    assert json.loads((output / "slurm-final.json").read_text())["returncode"] == 0
    failed = json.loads((output / "active-final.json").read_text())
    assert failed["returncode"] == 255 and "failed" in failed["stderr"]


def test_failed_submission_never_queries_or_invents_terminal(tmp_path):
    process, output, submission, log = start_watcher(tmp_path, "running_then_terminal")
    submit_fixture(submission, returncode=1)
    process.communicate(timeout=6)
    assert process.returncode == 3
    assert not log.exists()
    assert not (output / "slurm-final.json").exists()


def launcher_fixture(tmp_path, gpu_script, native_script):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    for name in ("run_m3_campaign.py", "run_m3_suite.py", "m3_sources.py"):
        shutil.copyfile(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts/run_m3.sh").write_text(gpu_script)
    (repo / "scripts/run_m3_regressions.sh").write_text(native_script)
    runtime, auth, history = (
        repo / "configs/m3-real-model.json",
        repo / "configs/auth.json",
        tmp_path / "history.json",
    )
    for path in (runtime, auth, history):
        path.write_text("{}\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plan = {
        "campaign_id": "synthetic-launcher-only",
        "runtime_config": {"path": "configs/m3-real-model.json", "sha256": digest(runtime)},
        "authorization": {"path": "configs/auth.json", "sha256": digest(auth)},
        "history": {"index_sha256": digest(history)},
    }
    config = repo / "configs/campaign.json"
    config.write_text(json.dumps(plan))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "synthetic launcher fixture",
        ],
        check=True,
    )
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake = fakebin / "nvidia-smi"
    fake.write_text('#!/bin/sh\nprintf "SYNTHETIC GPU fixture, 24576, 1, 24575, 0\\n"\n')
    fake.chmod(0o700)
    # The desktop execution policy blocks the system ps command. This explicit
    # process-inventory fixture reports only the actual child PID owned by this
    # test, preserving real signal/killpg behavior without scanning user apps.
    fake_ps = fakebin / "ps"
    fake_ps.write_text("""#!/usr/bin/env python3
import os, pathlib
marker=pathlib.Path(os.environ['M3_TEST_MARKER'])
if marker.exists():
    try:
        pid=int(marker.read_text())
        os.kill(pid,0)
        group=os.getpgid(pid)
    except ProcessLookupError:
        pass
    else:
        print(f'{pid} 0 {group} SYNTHETIC-known-child-inventory')
""")
    fake_ps.chmod(0o700)
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "SLURM_JOB_ID": "43210",
        "NATIVE_PYTHON": sys.executable,
        "MODEL_CONFIG_DIR": str(tmp_path / "no-model"),
        "M3_CACHE": str(tmp_path / "no-cache"),
        "PATH": str(fakebin) + os.pathsep + os.environ.get("PATH", ""),
        "M3_TEST_MARKER": str(tmp_path / "marker"),
    }
    env.pop("PYTHONOPTIMIZE", None)
    command = [
        sys.executable,
        str(repo / "scripts/run_m3_campaign.py"),
        "--campaign-config",
        str(config),
        "--history-index",
        str(history),
        "--output",
        str(output),
    ]
    return command, env, output


@pytest.mark.parametrize("gpu_code,native_code", [(0, 0), (7, 0), (0, 9)])
def test_launcher_preserves_step_failures_and_always_writes_exit_resources(
    tmp_path, gpu_code, native_code
):
    command, env, output = launcher_fixture(tmp_path, f"exit {gpu_code}\n", f"exit {native_code}\n")
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == (0 if gpu_code == native_code == 0 else 1), (
        result.stdout + result.stderr
    )
    exits = json.loads((output / "campaign-exit.json").read_text())
    resources = json.loads((output / "allocation-resource-checkpoints.json").read_text())
    assert exits["gpu"] == gpu_code
    assert exits["native"] == (native_code if gpu_code == 0 else 125)
    assert exits["gpu_after_query"] == 0
    assert exits["owned_pids_remaining"] == resources["owned_pids_remaining"] == []
    assert resources["cleanup_errors"] == []
    assert len(resources["launcher_steps"]) == (2 if gpu_code == 0 else 1)
    assert resources["gpu_before"]["returncode"] == resources["gpu_after"]["returncode"] == 0


def test_launcher_signal_cleans_only_owned_live_process_group_and_preserves_evidence(tmp_path):
    command, env, output = launcher_fixture(
        tmp_path, 'echo $$ > "$M3_TEST_MARKER"\nexec sleep 60\n', "exit 0\n"
    )
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    unrelated = subprocess.Popen(["sleep", "60"], start_new_session=True)
    marker = Path(env["M3_TEST_MARKER"])
    try:
        wait_file(marker, process)
        owned_pid = int(marker.read_text())
        process.terminate()
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode != 0, stdout + stderr
        assert "interrupted by signal" in stderr
        exits = json.loads((output / "campaign-exit.json").read_text())
        resources = json.loads((output / "allocation-resource-checkpoints.json").read_text())
        assert exits["owned_pids_remaining"] == resources["owned_pids_remaining"] == []
        assert resources["cleanup_errors"] == []
        with pytest.raises(ProcessLookupError):
            os.kill(owned_pid, 0)
        assert unrelated.poll() is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("checkpoint", ["[]", '{"owned_pids_remaining":{}}'])
def test_launcher_malformed_checkpoint_still_preserves_exit_and_cleanup_failure(
    tmp_path, checkpoint
):
    script = (
        "mkdir -p \"${M3_RUN}-data\"\nprintf '%s\\n' '"
        + checkpoint
        + '\' > "${M3_RUN}-data/resource-checkpoints.json"\nexit 0\n'
    )
    command, env, output = launcher_fixture(tmp_path, script, "exit 0\n")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1, result.stdout + result.stderr
    exits = json.loads((output / "campaign-exit.json").read_text())
    resources = json.loads((output / "allocation-resource-checkpoints.json").read_text())
    assert exits["resource_cleanup"] == 127
    assert resources["cleanup_errors"]
    assert resources["owned_pids_remaining"] == []
    assert resources["gpu_after"]["returncode"] == 0
