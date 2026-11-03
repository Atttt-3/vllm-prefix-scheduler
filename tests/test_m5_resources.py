"""Exercise the M5 release gate through the actual resource auditor CLI."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = "M5-paired-performance"


def write(path, value):
    path.write_text(json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def gpu(ns, *, active=False):
    return {
        "monotonic_ns": ns,
        "devices": {
            "returncode": 0,
            "stdout": f"GPU-abc, NVIDIA GeForce RTX 3090, 24576, {4000 if active else 1}, 0\n",
            "stderr": "",
        },
        "applications": {
            "returncode": 0,
            "stdout": "GPU-abc, 150, 3999\n" if active else "",
            "stderr": "",
        },
    }


@pytest.fixture
def private_attempt(tmp_path):
    folder = tmp_path / "attempt-1"
    folder.mkdir()
    watch = folder / "watcher"
    node = folder / "node"
    watch.mkdir()
    node.mkdir()
    job = "100"
    role = "burst-01"
    config_hash = "d" * 64
    commit = "b" * 40
    authorization = {
        "task_id": TASK,
        "schema_version": 1,
        "phase": "formal",
        "execution_commit": commit,
        "config_sha256": config_hash,
        "user_approval_ref": "fixture-only",
        "roles": [role],
        "max_attempts": 1,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "max_requested_minutes": 20,
        "freeze_sha256": "e" * 64,
    }
    auth_hash = write(folder / "authorization.json", authorization)
    binding = {
        "task_id": TASK,
        "job_id": job,
        "phase": "formal",
        "role": role,
        "attempt": folder.name,
        "execution_commit": commit,
        "config_sha256": config_hash,
        "authorization_sha256": auth_hash,
        "auxiliary_sha256": "e" * 64,
        "started_monotonic_ns": 100,
    }
    domain = {"clock_domain": "single"}
    write(
        folder / "planned.json",
        {**binding, **domain, "job_name": "m5-paired-attempt-1", "requested_minutes": 20},
    )
    write(
        watch / "monitor-ready.json",
        {
            **domain,
            "ready": True,
            "attempt_id": folder.name,
            "campaign_id": TASK,
            "campaign_sha256": config_hash,
            "project_commit": commit,
            "observed_monotonic_ns": 10,
            "pid": 300,
        },
    )
    write(
        folder / "submission.json",
        {
            **domain,
            "returncode": 0,
            "stdout": job + "\n",
            "command_started_ns": 20,
            "observed_monotonic_ns": 30,
        },
    )
    terminal = {
        **domain,
        "returncode": 0,
        "stdout": (
            f"JobId={job} JobName=m5-paired-attempt-1 JobState=COMPLETED ExitCode=0:0\n"
            "TimeLimit=00:20:00 NumNodes=1 BatchHost=fixture-node TresPerNode=gres/gpu:rtx_3090:1\n"
            "StartTime=2026-11-03T12:00:00 EndTime=2026-11-03T12:10:00\n"
        ),
        "stderr": "",
        "command": [f"scontrol show job {job}"],
        "observed_monotonic_ns": 40,
    }
    write(watch / "slurm-final.json", terminal)
    write(watch / "watch.jsonl", terminal)
    write(
        watch / "active-final.json",
        {
            **domain,
            "returncode": 0,
            "stdout": "99|RUNNING|unrelated\n",
            "stderr": "",
            "command": ['squeue -h -o "%i|%T|%j"'],
            "observed_monotonic_ns": 50,
        },
    )
    write(
        folder / "watcher-exit.json",
        {**domain, "pid": 300, "returncode": 0, "observed_monotonic_ns": 60},
    )
    write(node / "binding.json", binding)
    (node / "gpu-during.jsonl").write_text(json.dumps(gpu(120, active=True)) + "\n")
    index_hash = write(
        node / "index.json", {"runs": [{"returncode": 0, "remaining_processes": []}]}
    )
    write(
        node / "allocation.json",
        {
            **binding,
            "ended_monotonic_ns": 200,
            "work_returncode": 0,
            "node": "fixture-node",
            "errors": [],
            "remaining_processes": [],
            "gpu_before": gpu(105),
            "gpu_after": gpu(170),
            "sampler_exited": True,
            "index_sha256": index_hash,
        },
    )
    return folder


def check(folder, *, optimized=False):
    return subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            str(ROOT / "scripts/check_m5_resources.py"),
            str(folder),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_resource_positive(private_attempt):
    result = check(private_attempt)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["attempts"][0]["elapsed_seconds"] == 600


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("late_watcher", "watcher time order"),
        ("failed_queue", "full queue query failed"),
        ("active_job", "still visible"),
        ("unknown_terminal", "slurm-final.json"),
        ("device_residue", "GPU process or memory residue"),
        ("process_residue", "process cleanup"),
        ("replay_residue", "replay group residue"),
        ("missing_boundary", "gpu-during.jsonl"),
    ],
)
def test_resource_semantic_negative(private_attempt, case, expected):
    folder = private_attempt
    if case == "unknown_terminal":
        (folder / "watcher/slurm-final.json").unlink()
    elif case == "missing_boundary":
        (folder / "node/gpu-during.jsonl").unlink()
    else:
        allocation_path = folder / "node/allocation.json"
        active_path = folder / "watcher/active-final.json"
        if case == "late_watcher":
            path = folder / "watcher/monitor-ready.json"
            write(path, {**read(path), "observed_monotonic_ns": 25})
        elif case == "failed_queue":
            write(active_path, {**read(active_path), "returncode": 1})
        elif case == "active_job":
            write(active_path, {**read(active_path), "stdout": "100|RUNNING|m5-paired-attempt-1\n"})
        elif case == "device_residue":
            write(allocation_path, {**read(allocation_path), "gpu_after": gpu(170, active=True)})
        elif case == "process_residue":
            write(allocation_path, {**read(allocation_path), "remaining_processes": ["foreign"]})
        else:
            path = folder / "node/index.json"
            new_hash = write(
                path, {"runs": [{"returncode": 0, "remaining_processes": ["residue"]}]}
            )
            write(allocation_path, {**read(allocation_path), "index_sha256": new_hash})
    result = check(folder)
    assert result.returncode == 1, result.stdout
    assert expected in result.stderr


def test_resource_optimized_is_refused(private_attempt):
    result = check(private_attempt, optimized=True)
    assert result.returncode == 1
    assert "unoptimized" in result.stderr
