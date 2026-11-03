"""Audit M5 allocation releases directly from controller and node records."""

import json
import re
from datetime import datetime
from pathlib import Path

from .m5_campaign import JOB_PREFIX, TASK, digest, exact_int, read_json, require

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def monotonic(value):
    require(type(value) is int and value > 0, "invalid monotonic timestamp")
    return value


def device(snapshot):
    for key in ("devices", "applications"):
        result = snapshot[key]
        require(
            type(result["returncode"]) is int
            and result["returncode"] == 0
            and not result["stderr"].strip(),
            "unsuccessful GPU query",
        )
    fields = [part.strip() for part in snapshot["devices"]["stdout"].strip().split(",")]
    require(len(fields) == 5, "one assigned GPU required")
    uuid, name, total, used, utilization = fields
    require(
        re.fullmatch(r"GPU-[a-fA-F0-9-]+", uuid) is not None
        and "RTX 3090" in name
        and exact_int(int(total), 23000, 25000)
        and exact_int(int(used), 0, int(total))
        and exact_int(int(utilization), 0, 100),
        "unexpected GPU device record",
    )
    processes = []
    for line in snapshot["applications"]["stdout"].splitlines():
        assigned, pid, memory = (part.strip() for part in line.split(","))
        require(
            assigned == uuid and int(pid) > 1 and exact_int(int(memory), 0, int(total)),
            "GPU process not bound to assigned device",
        )
        processes.append(int(pid))
    require(len(processes) == len(set(processes)), "duplicate GPU processes")
    return uuid, int(used), processes


def slurm_terminal(terminal, job, name):
    require(terminal["returncode"] == 0 and not terminal["stderr"].strip(), "terminal query failed")
    fields = dict(re.findall(r"(?<!\S)(\w+)=(\S+)", terminal["stdout"]))
    state, code = fields.get("JobState"), fields.get("ExitCode")
    require(
        fields.get("JobId") == job
        and fields.get("JobName") == name
        and state in TERMINAL
        and re.fullmatch(r"\d+:\d+", code or "") is not None
        and (state != "COMPLETED" or code == "0:0")
        and fields.get("TimeLimit") == "00:20:00"
        and fields.get("NumNodes") == "1"
        and fields.get("TresPerNode") == "gres/gpu:rtx_3090:1",
        "authoritative terminal/allocation identity differs",
    )
    started = datetime.fromisoformat(fields["StartTime"])
    ended = datetime.fromisoformat(fields["EndTime"])
    elapsed = (ended - started).total_seconds()
    require(0 <= elapsed <= 1200, "allocation wall budget exceeded")
    node = fields.get("BatchHost")
    require(type(node) is str and bool(node) and node != "(null)", "Slurm batch host missing")
    return state, code, started, ended, node


def queue_clear(active, job):
    require(active["returncode"] == 0 and not active["stderr"].strip(), "full queue query failed")
    require(
        active["command"][-1] == 'squeue -h -o "%i|%T|%j"',
        "queue evidence must cover the full queue",
    )
    for line in active["stdout"].splitlines():
        fields = line.split("|")
        require(len(fields) == 3 and all(fields), "invalid full queue row")
        require(
            fields[0].split("_")[0].split(".")[0] != job and not fields[2].startswith(JOB_PREFIX),
            "M5 allocation still visible in full queue",
        )


def audit_attempt(directory, *, require_completed=False):
    directory = Path(directory)
    record = read_json(directory / "planned.json")
    ready = read_json(directory / "watcher/monitor-ready.json")
    receipt = read_json(directory / "submission.json")
    terminal = read_json(directory / "watcher/slurm-final.json")
    active = read_json(directory / "watcher/active-final.json")
    watcher_exit = read_json(directory / "watcher-exit.json")
    job = receipt["stdout"].strip().split(";")[0]
    require(
        receipt["returncode"] == 0 and job.isascii() and job.isdecimal(),
        "ambiguous or failed submission receipt",
    )
    require(
        ready["ready"] is True
        and watcher_exit["returncode"] == 0
        and ready["pid"] == watcher_exit["pid"]
        and ready["attempt_id"] == record["attempt"]
        and ready["campaign_id"] == TASK
        and ready["campaign_sha256"] == record["config_sha256"]
        and ready["project_commit"] == record["execution_commit"]
        and record["attempt"] == directory.name,
        "watcher/attempt binding differs",
    )
    domain = record["clock_domain"]
    require(
        all(
            row["clock_domain"] == domain
            for row in (ready, receipt, terminal, active, watcher_exit)
        ),
        "controller clock domain mismatch",
    )
    marks = [ready["observed_monotonic_ns"], receipt["command_started_ns"]]
    marks.extend(row["observed_monotonic_ns"] for row in (receipt, terminal, active, watcher_exit))
    marks = [monotonic(value) for value in marks]
    require(
        all(left < right for left, right in zip(marks, marks[1:], strict=False)),
        "watcher time order invalid",
    )
    require(terminal["command"][-1] == f"scontrol show job {job}", "terminal query job mismatch")
    journal = [
        json.loads(line) for line in (directory / "watcher/watch.jsonl").read_text().splitlines()
    ]
    require(journal and journal[-1] == terminal, "terminal not in independent watcher journal")
    require(
        all(
            row["clock_domain"] == domain
            and row["command"][-1] == f"scontrol show job {job}"
            and marks[2] <= row["observed_monotonic_ns"] <= marks[3]
            for row in journal
        ),
        "watcher journal observation mismatch",
    )
    state, exit_code, slurm_start, slurm_end, node = slurm_terminal(
        terminal, job, record["job_name"]
    )
    queue_clear(active, job)

    payload = directory / "node"
    allocation = read_json(payload / "allocation.json")
    binding = read_json(payload / "binding.json")
    fields = (
        "task_id",
        "job_id",
        "phase",
        "role",
        "attempt",
        "execution_commit",
        "config_sha256",
        "authorization_sha256",
        "auxiliary_sha256",
    )
    expected = {**record, "task_id": TASK, "job_id": job}
    require(
        all(allocation[key] == binding[key] == expected[key] for key in fields),
        "allocation binding differs from submission",
    )
    authorization_path = directory / "authorization.json"
    authorization = read_json(authorization_path)
    require(
        digest(authorization_path) == record["authorization_sha256"]
        and authorization["task_id"] == TASK
        and authorization["schema_version"] == 1
        and all(
            authorization[key] == record[key]
            for key in ("phase", "config_sha256", "execution_commit")
        )
        and record["role"] in authorization["roles"]
        and type(authorization["user_approval_ref"]) is str
        and authorization["user_approval_ref"].strip()
        and exact_int(authorization["max_attempts"], 1, 99)
        and authorization["max_requested_minutes"] == 20 * authorization["max_attempts"]
        and authorization["minutes_per_attempt"] == record["requested_minutes"] == 20
        and authorization["gpus_per_attempt"] == 1
        and authorization["freeze_sha256"]
        == (record["auxiliary_sha256"] if record["phase"] == "formal" else None),
        "allocation authorization or phase budget differs",
    )
    started = monotonic(binding["started_monotonic_ns"])
    ended = monotonic(allocation["ended_monotonic_ns"])
    require(
        allocation["node"].split(".")[0] == node.split(".")[0]
        and allocation["started_monotonic_ns"] == started
        and 0 < ended - started <= 1200 * 10**9
        and allocation["remaining_processes"] == []
        and allocation["sampler_exited"] is True
        and allocation.get("rpc_directory_removed", True) is True
        and not any("cleanup" in text for text in allocation["errors"]),
        "allocation process cleanup or timing evidence invalid",
    )
    before, after = allocation["gpu_before"], allocation["gpu_after"]
    gpu, used, before_apps = device(before)
    final_gpu, final_used, after_apps = device(after)
    require(
        gpu == final_gpu and not before_apps and not after_apps and final_used <= used + 64,
        "GPU process or memory residue at allocation boundary",
    )
    during = [json.loads(line) for line in (payload / "gpu-during.jsonl").read_text().splitlines()]
    times = [monotonic(snapshot["monotonic_ns"]) for snapshot in during]
    require(
        times
        and started <= before["monotonic_ns"] < times[0]
        and times[-1] < after["monotonic_ns"] <= ended
        and all(left < right for left, right in zip(times, times[1:], strict=False)),
        "before/during/after GPU checkpoint order incomplete",
    )
    seen_apps = set()
    for snapshot in during:
        sample_gpu, _, apps = device(snapshot)
        require(gpu == sample_gpu, "different GPU during allocation")
        seen_apps.update(apps)

    complete = state == "COMPLETED" and exit_code == "0:0"
    require(not require_completed or complete, "allocation did not complete successfully")
    index = payload / "index.json"
    if complete:
        require(
            allocation["work_returncode"] == 0
            and allocation["errors"] == []
            and seen_apps
            and digest(index) == allocation["index_sha256"],
            "successful allocation lacks bound work/observed GPU",
        )
    if index.is_file():
        require(digest(index) == allocation["index_sha256"], "allocation index hash differs")
        runs = read_json(index)["runs"]
        require(not complete or runs, "completed allocation has no runs")
        for run in runs:
            if run.get("returncode") is None and not complete:
                continue
            require(run["remaining_processes"] == [], "replay group residue")
            if complete:
                require(run["returncode"] == 0, "completed allocation contains failed replay")
    return {
        "job_id": job,
        "phase": record["phase"],
        "role": record["role"],
        "complete": complete,
        "state": state,
        "exit_code": exit_code,
        "slurm_start": slurm_start,
        "slurm_end": slurm_end,
        "elapsed_seconds": (slurm_end - slurm_start).total_seconds(),
        "requested_minutes": 20,
        "max_attempts": authorization["max_attempts"],
    }


def audit_history(root):
    paths = [path for path in Path(root).iterdir() if path.is_dir()]
    audits = [audit_attempt(path) for path in sorted(paths)]
    jobs = {row["job_id"] for row in audits}
    require(len(jobs) == len(audits), "duplicate owned job receipts")
    audits.sort(key=lambda row: row["slurm_start"])
    require(
        all(
            left["slurm_end"] <= right["slurm_start"]
            for left, right in zip(audits, audits[1:], strict=False)
        ),
        "M5 allocations overlap",
    )
    used = {}
    for row in audits:
        phase = row["phase"]
        used[phase] = used.get(phase, 0) + 1
        require(used[phase] <= row["max_attempts"], "M5 phase allocation budget exceeded")
    return audits
