"""Exercise the M5 release gate through the actual resource auditor CLI."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = "M5-paired-performance"
BOOT = "dfd3e9ad-f3f7-4ef1-9bcf-f295d62168c8"


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
        ("ipc_residue", "process cleanup"),
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
        elif case == "ipc_residue":
            write(allocation_path, {**read(allocation_path), "rpc_directory_removed": False})
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


@pytest.fixture
def v2_attempt(private_attempt):
    folder = private_attempt
    watch = folder / "watcher"
    node = folder / "node"
    allocation_path = node / "allocation.json"
    active = read(watch / "active-final.json")
    pre = {**active, "observed_monotonic_ns": 5}
    pre_hash = write(folder / "pre-submission-queue.json", [pre])
    record = read(folder / "planned.json")
    record.update(
        resource_schema_version=2,
        controller_boot_id=BOOT,
        remote_attempt=f"/scratch/m5/runs/{TASK}/node/{folder.name}",
        pre_submission_queue=pre["stdout"],
    )
    plan_hash = write(folder / "planned.json", record)
    receipt = read(folder / "submission.json")
    argv = ["sbatch", "--parsable", f"--job-name={record['job_name']}"]
    ready = watch / "monitor-ready.json"
    intent_hash = write(
        folder / "submission-intent.json",
        {
            "argv": argv,
            "clock_domain": receipt["clock_domain"],
            "command_started_ns": receipt["command_started_ns"],
            "planned_sha256": plan_hash,
            "watcher_ready_sha256": hashlib.sha256(ready.read_bytes()).hexdigest(),
            "pre_submission_queue_sha256": pre_hash,
        },
    )
    write(folder / "submission.json", {**receipt, "intent_sha256": intent_hash, "argv": argv})
    attempts = [
        {**active, "returncode": 255, "stderr": "connection reset", "observed_monotonic_ns": 45},
        active,
    ]
    (watch / "queue-watch.jsonl").write_text("".join(json.dumps(row) + "\n" for row in attempts))

    run = {
        "id": "native",
        "returncode": 0,
        "remaining_processes": [],
        "collector_pid": 800,
        "engine_pgid": 810,
        "started_monotonic_ns": 108,
        "engine_owner_ack_ns": 111,
        "ended_monotonic_ns": 155,
    }
    leaf = node / run["id"]
    leaf.mkdir()
    write(
        leaf / "owner.json",
        {
            "task_id": TASK,
            "job_id": "100",
            "pid": 810,
            "pgid": 810,
            "collector_pid": 800,
            "registered_monotonic_ns": 110,
        },
    )
    write(leaf / "ack.json", {"job_id": "100", "pgid": 810})
    sample = {
        **gpu(120, active=True),
        "application_sample_started_ns": 121,
        "application_sample_ended_ns": 122,
        "application_process_groups": [{"pid": 150, "pgid": 810}],
    }
    sample_path = node / "gpu-during.jsonl"
    sample_path.write_text(json.dumps(sample) + "\n")
    index_hash = write(node / "index.json", {"runs": [run]})
    write(node / "binding.json", {**read(node / "binding.json"), "resource_schema_version": 2})
    write(
        allocation_path,
        {
            **read(allocation_path),
            "resource_schema_version": 2,
            "rpc_directory_removed": True,
            "index_sha256": index_hash,
            "gpu_during_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        },
    )
    return folder


def test_v2_original_intent_retry_and_owned_gpu_release(v2_attempt):
    result = check(v2_attempt)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["attempts"][0]["job_id"] == "100"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("intent", "original submission intent/receipt binding differs"),
        ("receipt", "ambiguous or failed submission receipt"),
        ("watcher", "full queue retried after a successful observation"),
        ("pgid", "outside its owned engine process group"),
        ("time", "outside its owned engine process group"),
        ("pid", "GPU application PIDs differ"),
        ("owner", "GPU application engine ownership record differs"),
        ("schema", "M5 resource schema downgrade"),
    ],
)
def test_v2_release_rejects_unbound_evidence(v2_attempt, case, message):
    folder = v2_attempt
    if case == "intent":
        path = folder / "submission-intent.json"
        write(path, {**read(path), "watcher_ready_sha256": "f" * 64})
    elif case == "receipt":
        path = folder / "submission.json"
        write(path, {**read(path), "stdout": "100\nanother job\n"})
    elif case == "watcher":
        watch = folder / "watcher"
        active = read(watch / "active-final.json")
        path = watch / "queue-watch.jsonl"
        path.write_text(
            json.dumps({**active, "observed_monotonic_ns": 45}) + "\n" + json.dumps(active) + "\n"
        )
    elif case == "owner":
        path = folder / "node/native/owner.json"
        write(path, {**read(path), "job_id": "101"})
    elif case == "schema":
        path = folder / "planned.json"
        write(path, {**read(path), "resource_schema_version": 1})
    else:
        path = folder / "node/gpu-during.jsonl"
        sample = read(path)
        if case == "pgid":
            sample["application_process_groups"][0]["pgid"] = 999
        elif case == "pid":
            sample["application_process_groups"][0]["pid"] = 999
        else:
            sample.update(application_sample_started_ns=160, application_sample_ended_ns=161)
        sha = write(path, sample)
        allocation = folder / "node/allocation.json"
        write(allocation, {**read(allocation), "gpu_during_sha256": sha})
    result = check(folder)
    assert result.returncode == 1, result.stdout
    assert message in result.stderr


def test_pre_intent_attempt_does_not_consume_budget_but_unknown_submission_blocks(tmp_path):
    from vllm_prefix_scheduler.m5_resources import audit_history

    folder = tmp_path / "pre-intent"
    folder.mkdir()
    write(folder / "planned.json", {"resource_schema_version": 2})
    assert audit_history(tmp_path) == []
    write(folder / "submission-intent.json", {})
    with pytest.raises(FileNotFoundError):
        audit_history(tmp_path)


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("submit_m5")


def test_resume_uses_original_receipt_and_only_reacquires_release(
    controller, v2_attempt, monkeypatch
):
    from vllm_prefix_scheduler.m5_resources import QUEUE_QUERY

    folder = v2_attempt
    receipt = (folder / "submission.json").read_bytes()
    original_terminal = (folder / "watcher/slurm-final.json").read_bytes()
    exit_path = folder / "watcher-exit.json"
    write(exit_path, {**read(exit_path), "returncode": 4})
    monkeypatch.setattr(controller, "HISTORY", folder.parent)
    monkeypatch.setattr(controller, "controller_boot_id", lambda: BOOT)
    monkeypatch.setattr(
        controller, "connection", lambda: ("offline", {"m5_checkout": Path("/scratch/m5")}, "q")
    )
    monkeypatch.setattr(
        controller,
        "remote",
        lambda *args, **kwargs: pytest.fail("resume issued an allocating command"),
    )
    monkeypatch.setattr(controller, "fetch_evidence", lambda *args, **kwargs: None)

    def missing_process(*args):
        raise ProcessLookupError

    monkeypatch.setattr(controller.os, "kill", missing_process)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    queries = []

    def query(ssh, command, domain):
        queries.append(command)
        if command == "scontrol show job 100":
            row = read(folder / "watcher/slurm-final.json")
        else:
            row = read(folder / "watcher/active-final.json")
            if len(queries) == 2:
                row.update(returncode=255, stderr="temporary disconnect")
        return {**row, "observed_monotonic_ns": controller.time.monotonic_ns()}

    monkeypatch.setattr(controller, "remote_query", query)
    controller.resume(folder.name)
    assert queries == ["scontrol show job 100", QUEUE_QUERY, QUEUE_QUERY]
    assert (folder / "submission.json").read_bytes() == receipt
    assert (folder / "watcher/slurm-final.json").read_bytes() == original_terminal
    assert check(folder).returncode == 0


def test_resume_rejects_ambiguous_receipt_before_connection(controller, v2_attempt, monkeypatch):
    folder = v2_attempt
    path = folder / "submission.json"
    write(path, {**read(path), "stdout": "140001\n140002\n"})
    monkeypatch.setattr(controller, "HISTORY", folder.parent)
    monkeypatch.setattr(
        controller, "connection", lambda: pytest.fail("ambiguous receipt contacted cluster")
    )
    with pytest.raises(ValueError, match="ambiguous or failed submission receipt"):
        controller.resume(folder.name)


def test_node_transfer_retries_each_in_fresh_directory(controller, tmp_path, monkeypatch):
    transport = tmp_path / "transport"
    transport.mkdir()
    calls = []

    def copy(argv, **kwargs):
        staging = Path(argv[-1])
        staging.mkdir()
        calls.append(staging)
        (staging / "sample").write_text(str(len(calls)))
        return SimpleNamespace(returncode=0 if len(calls) == 3 else 1, stdout="", stderr="")

    monkeypatch.setattr(controller.subprocess, "run", copy)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    controller.transfer("offline", "/job", tmp_path / "node", transport)
    assert len(set(calls)) == 3
    assert [read(p)["returncode"] for p in sorted(transport.glob("*.json"))] == [1, 1, 0]
    assert (tmp_path / "node/sample").read_text() == "3"


def test_intent_is_durable_before_invoke_and_unreturned_submission_blocks(
    controller, tmp_path, monkeypatch
):
    from vllm_prefix_scheduler.m5_resources import audit_history

    history = tmp_path / "attempts"
    auth = tmp_path / "authorization.json"
    config = tmp_path / "config.json"
    commit = "a" * 40
    write(auth, {"max_attempts": 1})
    write(config, {"resource_schema_version": 2})
    monkeypatch.setattr(controller, "HISTORY", history)
    monkeypatch.setattr(controller, "CONFIG", config)
    monkeypatch.setattr(controller, "controller_boot_id", lambda: BOOT)
    monkeypatch.setattr(controller, "check_authorization", lambda *args, **kwargs: ["C1"])
    monkeypatch.setattr(
        controller,
        "connection",
        lambda: (
            "offline",
            {
                "m5_checkout": Path("/scratch/m5"),
                "remote_gpu_environment": Path("/scratch/python"),
                "remote_model_directory": Path("/scratch/model"),
                "remote_cache_directory": Path("/scratch/cache"),
            },
            "q",
        ),
    )

    def git(argv, **kwargs):
        return commit + "\n" if "rev-parse" in argv else b""

    class Watcher:
        pid = 600
        returncode = None

        def __init__(self, argv, **kwargs):
            def option(name):
                return argv[argv.index(name) + 1]

            folder = Path(option("--output"))
            folder.mkdir()
            write(
                folder / "monitor-ready.json",
                {
                    "ready": True,
                    "pid": self.pid,
                    "attempt_id": option("--attempt-id"),
                    "campaign_id": option("--campaign-id"),
                    "campaign_sha256": option("--campaign-sha256"),
                    "project_commit": option("--project-commit"),
                    "clock_domain": option("--clock-domain"),
                    "observed_monotonic_ns": controller.time.monotonic_ns(),
                },
            )

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    def remote(ssh, argv, **kwargs):
        if argv[0] == "sbatch":
            folder = history / "lost-reply"
            assert read(folder / "submission-intent.json")["argv"] == argv
            assert (folder / "watcher/monitor-ready.json").is_file()
            assert not (folder / "submission.json").exists()
            raise RuntimeError("simulated controller loss after remote invoke")
        return SimpleNamespace(returncode=0, stdout=commit if "rev-parse" in argv else "")

    def query(ssh, command, domain):
        return {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "command": [command],
            "clock_domain": domain,
            "observed_monotonic_ns": controller.time.monotonic_ns(),
        }

    monkeypatch.setattr(controller.subprocess, "check_output", git)
    monkeypatch.setattr(controller.subprocess, "Popen", Watcher)
    monkeypatch.setattr(controller, "remote", remote)
    monkeypatch.setattr(controller, "remote_query", query)
    with pytest.raises(RuntimeError, match="simulated controller loss"):
        controller.submit(
            SimpleNamespace(
                authorization=auth,
                phase="calibration",
                role="C1",
                attempt="lost-reply",
                freeze=None,
            )
        )
    with pytest.raises(FileNotFoundError, match="submission.json"):
        audit_history(history)


def test_m5_watcher_retries_full_queue_and_preserves_each_observation(tmp_path, monkeypatch):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    ssh = bin_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do last=$arg; done\n'
        'case "$last" in\n'
        '  "scontrol show job 100") printf "JobId=100 JobState=COMPLETED\\n" ;;\n'
        "  squeue*)\n"
        '    if [ ! -f "$M5_TEST_QUEUE_ATTEMPT" ]; then\n'
        '      : > "$M5_TEST_QUEUE_ATTEMPT"; printf "offline\\n" >&2; exit 255\n'
        "    fi ;;\n"
        "  *) exit 5 ;;\n"
        "esac\n"
    )
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(bin_path))
    monkeypatch.setenv("M5_TEST_QUEUE_ATTEMPT", str(tmp_path / "queue-attempt"))
    submission = tmp_path / "submission.json"
    write(submission, {"returncode": 0, "stdout": "100\n", "stderr": ""})
    folder = tmp_path / "watch"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/watch_m3_campaign.py"),
            "--ssh-alias",
            "offline",
            "--campaign-id",
            TASK,
            "--campaign-sha256",
            "a" * 64,
            "--attempt-id",
            "a",
            "--project-commit",
            "b" * 40,
            "--clock-domain",
            "domain",
            "--output",
            str(folder),
            "--submission",
            str(submission),
            "--timeout",
            "5",
            "--m5-release",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    journal = [json.loads(line) for line in (folder / "queue-watch.jsonl").read_text().splitlines()]
    assert [row["returncode"] for row in journal] == [255, 0]
    assert journal[-1] == read(folder / "active-final.json")
