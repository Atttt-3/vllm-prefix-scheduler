"""Audit M5 allocation releases directly from controller and node records."""

import json
import re
from datetime import datetime
from pathlib import Path

from .m5_campaign import (
    D2_ROLE,
    D2R_ROLE,
    JOB_PREFIX,
    TASK,
    digest,
    exact_int,
    id_ok,
    is_hash,
    read_json,
    require,
)

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
RESOURCE_SCHEMA_VERSION = 2
QUEUE_QUERY = 'squeue -h -o "%i|%T|%j"'
QUEUE_ATTEMPTS = 3


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
    require(
        type(active["returncode"]) is int
        and active["returncode"] == 0
        and type(active["stdout"]) is str
        and type(active["stderr"]) is str
        and not active["stderr"].strip(),
        "full queue query failed",
    )
    require(
        active["command"][-1] == QUEUE_QUERY,
        "queue evidence must cover the full queue",
    )
    for line in active["stdout"].splitlines():
        fields = line.split("|")
        require(len(fields) == 3 and all(fields), "invalid full queue row")
        require(
            fields[0].split("_")[0].split(".")[0] != job and not fields[2].startswith(JOB_PREFIX),
            "M5 allocation still visible in full queue",
        )


def observe_clear_queue(query, job, pause, *, attempts=QUEUE_ATTEMPTS):
    """Retain every read-only full-queue observation, including failed attempts."""
    records = []
    for i in range(attempts):
        row = query(QUEUE_QUERY)
        records.append(row)
        try:
            queue_clear(row, job)
        except ValueError:
            if i + 1 < attempts:
                pause(1)
        else:
            return records, True
    return records, False


def receipt_job(receipt):
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9][A-Za-z0-9_.-]*)?\n?", receipt["stdout"])
    require(
        type(receipt["returncode"]) is int
        and receipt["returncode"] == 0
        and match is not None
        and not receipt.get("stderr", "").strip(),
        "ambiguous or failed submission receipt",
    )
    return match[1]


def audit_submission(directory):
    """Validate the original receipt and readiness, without inferring a job from the queue."""
    directory = Path(directory)
    record = read_json(directory / "planned.json")
    ready = read_json(directory / "watcher/monitor-ready.json")
    version = record.get("resource_schema_version", 1)
    require(
        type(version) is int and version in (1, RESOURCE_SCHEMA_VERSION),
        "unknown M5 resource schema version",
    )
    intent_path = directory / "submission-intent.json"
    if version == RESOURCE_SCHEMA_VERSION:
        intent = read_json(intent_path)
    else:
        require(not intent_path.exists(), "M5 resource schema downgrade")
    receipt = read_json(directory / "submission.json")
    job = receipt_job(receipt)
    domain = record["clock_domain"]
    require(
        record["task_id"] == TASK
        and record["job_name"] == JOB_PREFIX + record["attempt"]
        and ready["ready"] is True
        and ready["attempt_id"] == record["attempt"]
        and ready["campaign_id"] == TASK
        and ready["campaign_sha256"] == record["config_sha256"]
        and ready["project_commit"] == record["execution_commit"]
        and record["attempt"] == directory.name
        and ready["clock_domain"] == receipt["clock_domain"] == domain,
        "watcher/attempt binding differs",
    )
    ready_ns, started_ns, received_ns = map(
        monotonic,
        (
            ready["observed_monotonic_ns"],
            receipt["command_started_ns"],
            receipt["observed_monotonic_ns"],
        ),
    )
    require(ready_ns < started_ns < received_ns, "watcher time order invalid")
    if version == RESOURCE_SCHEMA_VERSION:
        argv = receipt["argv"]
        require(
            intent["argv"] == argv
            and argv[:2] == ["sbatch", "--parsable"]
            and f"--job-name={record['job_name']}" in argv
            and intent["clock_domain"] == domain
            and intent["command_started_ns"] == started_ns
            and intent["planned_sha256"] == digest(directory / "planned.json")
            and intent["watcher_ready_sha256"] == digest(directory / "watcher/monitor-ready.json")
            and intent["pre_submission_queue_sha256"]
            == digest(directory / "pre-submission-queue.json")
            and receipt["intent_sha256"] == digest(intent_path)
            and record["authorization_sha256"] == digest(directory / "authorization.json")
            and re.fullmatch(r"[a-fA-F0-9-]{36}", record["controller_boot_id"]) is not None,
            "original submission intent/receipt binding differs",
        )
        pre_queue = read_json(directory / "pre-submission-queue.json")
        queue_journal_gate(pre_queue, domain, None, None, ready_ns)
        require(
            record["pre_submission_queue"] == pre_queue[-1]["stdout"],
            "planned full queue differs from retained observation",
        )
    return record, ready, receipt, job


def queue_journal_gate(rows, domain, job, terminal_ns, end_ns):
    require(1 <= len(rows) <= QUEUE_ATTEMPTS, "full queue retry evidence missing or unbounded")
    stamps = [monotonic(row["observed_monotonic_ns"]) for row in rows]
    require(
        (terminal_ns is None or terminal_ns < stamps[0])
        and stamps[-1] < end_ns
        and all(a < b for a, b in zip(stamps, stamps[1:], strict=False))
        and all(
            row["clock_domain"] == domain and row["command"][-1] == QUEUE_QUERY for row in rows
        ),
        "full queue retry evidence ordering or query differs",
    )
    for row in rows[:-1]:
        try:
            queue_clear(row, job)
        except ValueError:
            continue
        raise ValueError("full queue retried after a successful observation")
    queue_clear(rows[-1], job)


def journal_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_controller_release(directory, submission=None):
    directory = Path(directory)
    record, ready, receipt, job = submission or audit_submission(directory)
    domain = record["clock_domain"]
    received_ns = receipt["observed_monotonic_ns"]
    recovery = directory / "recovery"
    if (recovery / "release.json").exists():
        require(record.get("resource_schema_version") == RESOURCE_SCHEMA_VERSION, "legacy recovery")
        release = read_json(recovery / "release.json")
        require(
            release["controller_boot_id"] == record["controller_boot_id"]
            and release["receipt_sha256"] == digest(directory / "submission.json")
            and release["intent_sha256"] == digest(directory / "submission-intent.json")
            and release["clock_domain"] == domain
            and release["original_pid_not_live"] is True,
            "read-only recovery detached from original submission",
        )
        original_exit = directory / "watcher-exit.json"
        original_end_ns = received_ns
        if original_exit.exists():
            original = read_json(original_exit)
            require(
                original["clock_domain"] == domain and original["pid"] == ready["pid"],
                "original watcher exit differs",
            )
            original_end_ns = monotonic(original["observed_monotonic_ns"])
        terminal_path = recovery / "slurm-final.json"
        terminal = read_json(terminal_path)
        active = read_json(recovery / "active-final.json")
        require(
            terminal["clock_domain"] == active["clock_domain"] == domain
            and received_ns
            <= original_end_ns
            < release["started_monotonic_ns"]
            < terminal["observed_monotonic_ns"]
            < active["observed_monotonic_ns"]
            < release["observed_monotonic_ns"]
            and release["terminal_sha256"] == digest(terminal_path)
            and release["active_sha256"] == digest(recovery / "active-final.json")
            and release["queue_journal_sha256"] == digest(recovery / "queue-watch.jsonl"),
            "read-only recovery evidence chronology or hashes differ",
        )
        rows = journal_rows(recovery / "queue-watch.jsonl")
        require(rows and rows[-1] == active, "recovery queue observation missing from journal")
        queue_journal_gate(
            rows, domain, job, terminal["observed_monotonic_ns"], release["observed_monotonic_ns"]
        )
        require(
            terminal["command"][-1] == f"scontrol show job {job}", "terminal query job mismatch"
        )
        return slurm_terminal(terminal, job, record["job_name"]), terminal_path

    terminal_path = directory / "watcher/slurm-final.json"
    terminal = read_json(terminal_path)
    active = read_json(directory / "watcher/active-final.json")
    watcher_exit = read_json(directory / "watcher-exit.json")
    require(
        watcher_exit["returncode"] == 0
        and ready["pid"] == watcher_exit["pid"]
        and all(row["clock_domain"] == domain for row in (terminal, active, watcher_exit)),
        "watcher exit or controller clock domain mismatch",
    )
    marks = [received_ns] + [
        monotonic(row["observed_monotonic_ns"]) for row in (terminal, active, watcher_exit)
    ]
    require(
        all(left < right for left, right in zip(marks, marks[1:], strict=False)),
        "watcher time order invalid",
    )
    require(terminal["command"][-1] == f"scontrol show job {job}", "terminal query job mismatch")
    journal = journal_rows(directory / "watcher/watch.jsonl")
    require(journal and journal[-1] == terminal, "terminal not in independent watcher journal")
    require(
        all(
            row["clock_domain"] == domain
            and row["command"][-1] == f"scontrol show job {job}"
            and marks[0] < row["observed_monotonic_ns"] <= marks[1]
            for row in journal
        ),
        "watcher journal observation mismatch",
    )
    queue_clear(active, job)
    if record.get("resource_schema_version") == RESOURCE_SCHEMA_VERSION:
        rows = journal_rows(directory / "watcher/queue-watch.jsonl")
        require(rows and rows[-1] == active, "final queue observation missing from journal")
        queue_journal_gate(rows, domain, job, marks[1], marks[3])
    return slurm_terminal(terminal, job, record["job_name"]), terminal_path


def engine_groups(runs, payload, job, started, ended):
    groups = {}
    for run in runs:
        if "engine_owner_ack_ns" not in run:
            require(run.get("returncode") is None, "completed replay lacks an owner ack")
            continue
        require(id_ok(run["id"]), "invalid replay owner record path")
        leaf = payload / run["id"]
        owner = read_json(leaf / "owner.json")
        ack = read_json(leaf / "ack.json")
        pgid = run["engine_pgid"]
        begin = monotonic(run["engine_owner_ack_ns"])
        end = monotonic(run.get("ended_monotonic_ns", ended))
        require(
            exact_int(pgid, 2, 2**31 - 1)
            and pgid not in groups
            and owner["task_id"] == TASK
            and owner["job_id"] == job
            and owner["collector_pid"] == run["collector_pid"]
            and owner["pid"] == owner["pgid"] == pgid
            and ack == {"job_id": job, "pgid": pgid}
            and started
            <= run["started_monotonic_ns"]
            <= owner["registered_monotonic_ns"]
            <= begin
            <= end
            <= ended,
            "GPU application engine ownership record differs",
        )
        groups[pgid] = (begin, end)
    return groups


def audit_sample_groups(snapshot, apps, groups, ended):
    ownership = snapshot.get("application_process_groups")
    require(
        type(ownership) is list and "application_groups_error" not in snapshot,
        "GPU application PGID evidence missing",
    )
    begin = monotonic(snapshot["application_sample_started_ns"])
    end = monotonic(snapshot["application_sample_ended_ns"])
    require(
        snapshot["monotonic_ns"] <= begin <= end <= ended,
        "GPU application sample outside allocation",
    )
    require(
        len(ownership) == len(apps) and [row["pid"] for row in ownership] == apps,
        "GPU application PIDs differ from the device sample",
    )
    seen = set()
    for row in ownership:
        pgid = row["pgid"]
        require(
            type(pgid) is int
            and pgid in groups
            and groups[pgid][0] <= end
            and begin <= groups[pgid][1],
            "GPU application outside its owned engine process group or time window",
        )
        seen.add(pgid)
    return seen


def audit_attempt(directory, *, require_completed=False):
    directory = Path(directory)
    submission = audit_submission(directory)
    record, _, _, job = submission
    terminal, terminal_path = audit_controller_release(directory, submission)
    state, exit_code, slurm_start, slurm_end, node = terminal

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
    version = record.get("resource_schema_version", 1)
    strict = version == RESOURCE_SCHEMA_VERSION
    require(
        all(allocation[key] == binding[key] == expected[key] for key in fields)
        and allocation.get("resource_schema_version", 1)
        == binding.get("resource_schema_version", 1)
        == version,
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
    if record["role"] == D2R_ROLE:
        require(
            strict
            and record["phase"] == "calibration"
            and authorization["roles"] == [D2R_ROLE]
            and authorization["max_attempts"] == 8
            and is_hash(authorization["d2r_plan_sha256"])
            and authorization["d2r_plan_sha256"] == record["auxiliary_sha256"],
            "D2-R resource binding differs from its one-slot reviewed authorization",
        )
    started = monotonic(binding["started_monotonic_ns"])
    ended = monotonic(allocation["ended_monotonic_ns"])
    require(
        allocation["node"].split(".")[0] == node.split(".")[0]
        and allocation["started_monotonic_ns"] == started
        and 0 < ended - started <= 1200 * 10**9
        and allocation["remaining_processes"] == []
        and allocation["sampler_exited"] is True
        and allocation.get("rpc_directory_removed", not strict) is True
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
    gpu_samples = payload / "gpu-during.jsonl"
    if strict:
        require(
            allocation["gpu_during_sha256"] == digest(gpu_samples), "allocation GPU samples differ"
        )
    during = journal_rows(gpu_samples)
    times = [monotonic(snapshot["monotonic_ns"]) for snapshot in during]
    require(
        times
        and started <= before["monotonic_ns"] < times[0]
        and times[-1] < after["monotonic_ns"] <= ended
        and all(left < right for left, right in zip(times, times[1:], strict=False)),
        "before/during/after GPU checkpoint order incomplete",
    )
    complete = state == "COMPLETED" and exit_code == "0:0"
    require(not require_completed or complete, "allocation did not complete successfully")
    index = payload / "index.json"
    runs = []
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
    groups = engine_groups(runs, payload, job, started, ended) if strict else {}
    seen_apps, seen_groups = set(), set()
    for snapshot in during:
        sample_gpu, _, apps = device(snapshot)
        require(gpu == sample_gpu, "different GPU during allocation")
        seen_apps.update(apps)
        if strict:
            seen_groups.update(audit_sample_groups(snapshot, apps, groups, ended))
    if complete:
        require(
            allocation["work_returncode"] == 0
            and allocation["errors"] == []
            and seen_apps
            and runs
            and (not strict or seen_groups == groups.keys()),
            "successful allocation lacks bound work/observed GPU",
        )
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
        "terminal_sha256": digest(terminal_path),
    }


def attempt_evidence_paths(directory):
    """Return existing evidence consumed by the resource audit plus transport logs."""
    directory = Path(directory)
    fixed = (
        "planned.json",
        "authorization.json",
        "submission-intent.json",
        "submission.json",
        "pre-submission-queue.json",
        "watcher.log",
        "watcher-exit.json",
        "watcher/monitor-ready.json",
        "watcher/slurm-final.json",
        "watcher/active-final.json",
        "watcher/watch.jsonl",
        "watcher/queue-watch.jsonl",
        "recovery/release.json",
        "recovery/slurm-final.json",
        "recovery/active-final.json",
        "recovery/queue-watch.jsonl",
        "slurm.log",
        "node/allocation.json",
        "node/binding.json",
        "node/index.json",
        "node/gpu-during.jsonl",
    )
    paths = [directory / name for name in fixed if (directory / name).is_file()]
    transport = directory / "transport"
    paths.extend(sorted(transport.glob("*.json")))
    for recovery in sorted(transport.glob("recovery-*")):
        paths.extend(
            recovery / name
            for name in ("slurm-final.json", "active-final.json", "queue-watch.jsonl")
            if (recovery / name).is_file()
        )
    index = directory / "node/index.json"
    if index.exists():
        for run in read_json(index)["runs"]:
            leaf = index.parent / run["id"]
            if leaf.parent == index.parent:
                paths.extend(
                    leaf / name for name in ("owner.json", "ack.json") if (leaf / name).is_file()
                )
    return paths


def submitted_attempts(root):
    """Preserve pre-intent archives without treating them as allocation receipts."""
    paths = []
    for path in sorted(Path(root).iterdir()):
        if not path.is_dir():
            continue
        plan = read_json(path / "planned.json")
        pre_intent = (
            plan.get("resource_schema_version") == RESOURCE_SCHEMA_VERSION
            and not (path / "submission-intent.json").exists()
            and not (path / "submission.json").exists()
        )
        if pre_intent:
            require(
                all(
                    not (path / name).exists()
                    for name in (
                        "node",
                        "recovery",
                        "watcher/slurm-final.json",
                        "watcher/active-final.json",
                    )
                ),
                "unsubmitted attempt contains allocation release evidence",
            )
            continue
        paths.append(path)
    return paths


def audit_history(root):
    audits = [audit_attempt(path) for path in submitted_attempts(root)]
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
    for index, row in enumerate(audits):
        phase = row["phase"]
        used[phase] = used.get(phase, 0) + 1
        require(used[phase] <= row["max_attempts"], "M5 phase allocation budget exceeded")
        if row["role"] == D2R_ROLE:
            require(
                index == 7
                and used[phase] == 8
                and audits[index - 1]["role"] == D2_ROLE
                and audits[index - 1]["complete"],
                "D2-R resource history differs from the approved eighth allocation",
            )
    return audits
