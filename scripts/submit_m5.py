"""Submit one explicitly approved M5 allocation and retain independent release evidence."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

from m4_processes import atomic_write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m5_c3_bundle import c3_bundle, c3_bundle_digest, verify_split_qualification  # noqa: E402

from vllm_prefix_scheduler.m5_campaign import (  # noqa: E402
    CALIBRATION,
    D1_ROLE,
    D2_ROLE,
    D2R110_PLAN_FILENAME,
    D2R110_PREVIOUS_ANALYSIS,
    D2R110_ROLE,
    D2R_PLAN_FILENAME,
    D2R_PREVIOUS_ANALYSIS,
    D2R_ROLE,
    D3_PLAN_FILENAME,
    D3_PREVIOUS_ANALYSIS,
    D3_ROLE,
    D4_PLAN_FILENAME,
    D4_ROLE,
    D5_PLAN_FILENAME,
    D5_ROLE,
    D5R_PLAN_FILENAME,
    D5R_ROLE,
    D6_PLAN_FILENAME,
    D6_ROLE,
    D7_PLAN_FILENAME,
    D7_ROLE,
    D8_PLAN_FILENAME,
    D8_ROLE,
    DIAGNOSTIC_ROLES,
    JOB_PREFIX,
    SPLIT_C3_ROLES,
    TASK,
    bind_d2r110_inputs,
    bind_d2r_inputs,
    bind_d3_inputs,
    check_authorization,
    config_matrix,
    digest,
    id_ok,
    qualification_roles,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_contract import (  # noqa: E402
    bind_formal_calibration,
    validate_formal_tables,
)
from vllm_prefix_scheduler.m5_d1 import d1_specs  # noqa: E402
from vllm_prefix_scheduler.m5_d2 import (  # noqa: E402
    D2_D1_REANALYSIS_SHA,
    D2R110_PROPOSAL_NAME,
    D2R_PROPOSAL_NAME,
    D3_PROPOSAL_NAME,
    d2_specs,
    d2r110_specs,
    d2r_specs,
    d3_specs,
)
from vllm_prefix_scheduler.m5_d4 import (  # noqa: E402
    D4_INPUT_6_NAME,
    D4_INPUT_8_NAME,
    D4_PREVIOUS_ANALYSIS,
    D4_PROPOSAL_NAME,
    d4_specs,
)
from vllm_prefix_scheduler.m5_d5 import (  # noqa: E402
    D5_PREVIOUS_ANALYSIS,
    D5_PROPOSAL_NAME,
    d5_specs,
)
from vllm_prefix_scheduler.m5_d5r import (  # noqa: E402
    D5R_PREVIOUS_ANALYSIS,
    D5R_PROPOSAL_NAME,
    d5r_specs,
)
from vllm_prefix_scheduler.m5_d6 import (  # noqa: E402
    D6_PREVIOUS_ANALYSIS,
    D6_PROPOSAL_NAME,
    d6_specs,
)
from vllm_prefix_scheduler.m5_d7 import (  # noqa: E402
    D7_LATE_TIMELINE,
    D7_PREVIOUS_ANALYSIS,
    D7_PROPOSAL_NAME,
    d7_specs,
)
from vllm_prefix_scheduler.m5_d8 import (  # noqa: E402
    D8_PREVIOUS_ANALYSIS,
    D8_PROPOSAL_NAME,
    d8_specs,
)
from vllm_prefix_scheduler.m5_resources import (  # noqa: E402
    RESOURCE_SCHEMA_VERSION,
    audit_attempt,
    audit_controller_release,
    audit_history,
    audit_submission,
    observe_clear_queue,
    slurm_terminal,
    submitted_attempts,
)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent / "vllm-prefix-scheduler/.local/workspace.json"
CONFIG = ROOT / "configs/m5-performance.json"
HISTORY = ROOT / "runs" / TASK / "attempts"
REMOTE_CONFIG = "configs/m5-performance.json"
TRANSFER_ATTEMPTS = 3


def connection():
    workspace = read_json(WORKSPACE)
    alias = workspace["ssh_login_alias"]
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", alias), "invalid SSH alias")
    paths = {
        key: PurePosixPath(workspace[key])
        for key in (
            "remote_checkout",
            "remote_gpu_environment",
            "remote_model_directory",
            "remote_cache_directory",
        )
    }
    require(
        all(path.is_absolute() and ".." not in path.parts for path in paths.values()),
        "cluster paths must be absolute and normalized",
    )
    checkout = paths["remote_checkout"]
    paths["m5_checkout"] = checkout.with_name("vllm-m5-paired-performance")
    require(paths["m5_checkout"] != checkout, "use the dedicated M5 checkout")
    partition = workspace["gpu_partition"]
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", partition), "invalid GPU partition")
    return alias, paths, partition


def remote(ssh, argv, *, data=None, timeout=30, check=True):
    return subprocess.run(
        [*ssh, shlex.join(map(str, argv))],
        input=data,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def durable_write(path, value):
    require(not path.exists(), "controller evidence already exists")
    atomic_write(path, value)
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def controller_boot_id():
    if sys.platform == "darwin":
        boot = subprocess.check_output(["sysctl", "-n", "kern.bootsessionuuid"], text=True).strip()
    else:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(
        re.fullmatch(r"[a-fA-F0-9-]{36}", boot) is not None, "controller boot identity unavailable"
    )
    return boot


def remote_query(ssh, command, domain):
    argv = [*ssh, command]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        data = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        data = dict(returncode=None, stdout="", stderr=str(error))
    return {
        **data,
        "command": argv,
        "clock_domain": domain,
        "observed_monotonic_ns": time.monotonic_ns(),
        "observed_unix_ns": time.time_ns(),
    }


def upload(ssh, path, content):
    # Create once, or prove this is the exact same staged input.
    stage = 'set -eu; umask 077; mkdir -p -- "$(dirname -- "$1")"; if [ -f "$1" ]; then cmp -s "$1" -; else (set -C; cat > "$1"); fi'
    remote(ssh, ["bash", "-c", stage, "m5-stage", path], data=content)


def preflight_diagnostic_stage(ssh, python, checkout, proposal, specs, role):
    # Validate the uploaded paths with the exact node interpreter before any allocation.
    program = """
import json, sys
from pathlib import Path
from vllm_prefix_scheduler.m5_campaign import read_json
from vllm_prefix_scheduler.m5_d2 import D2R110_PROPOSAL_NAME, D3_PROPOSAL_NAME
from vllm_prefix_scheduler.m5_d4 import D4_INPUT_8_NAME, D4_INPUT_6_NAME, D4_PREVIOUS_ANALYSIS, d4_specs
from vllm_prefix_scheduler.m5_d5 import D5_PREVIOUS_ANALYSIS, D5_PROPOSAL_NAME, d5_specs
from vllm_prefix_scheduler.m5_d5r import D5R_PREVIOUS_ANALYSIS, D5R_PROPOSAL_NAME, d5r_specs
from vllm_prefix_scheduler.m5_d6 import D6_PREVIOUS_ANALYSIS, D6_PROPOSAL_NAME, d6_specs
from vllm_prefix_scheduler.m5_d7 import D7_LATE_TIMELINE, D7_PREVIOUS_ANALYSIS, d7_specs
from vllm_prefix_scheduler.m5_d8 import D8_PREVIOUS_ANALYSIS, d8_specs
config, proposal = map(Path, sys.argv[1:3])
staged = proposal.parent
common = (staged / "original-c3-result.json", staged / D3_PROPOSAL_NAME)
if sys.argv[3] == "D4":
    specs = d4_specs(read_json(config), proposal, staged / D4_INPUT_8_NAME,
                     staged / D4_INPUT_6_NAME, *common, staged / D4_PREVIOUS_ANALYSIS)
elif sys.argv[3] == "D5":
    specs = d5_specs(read_json(config), proposal, staged / "input.json",
                     *common, staged / D5_PREVIOUS_ANALYSIS)
elif sys.argv[3] == "D5-R":
    specs = d5r_specs(read_json(config), proposal, staged / "input.json",
                      common[0], staged / D5_PROPOSAL_NAME, common[1],
                      staged / D5_PREVIOUS_ANALYSIS, staged / D5R_PREVIOUS_ANALYSIS)
elif sys.argv[3] == "D6":
    specs = d6_specs(read_json(config), proposal, staged / "input.json",
                     common[0], staged / D5R_PROPOSAL_NAME, staged / D5_PROPOSAL_NAME,
                     common[1], staged / D5_PREVIOUS_ANALYSIS, staged / D5R_PREVIOUS_ANALYSIS,
                     staged / D6_PREVIOUS_ANALYSIS)
elif sys.argv[3] == "D7":
    specs = d7_specs(read_json(config), proposal, staged / "input.json", common[0],
                     staged / D6_PROPOSAL_NAME, staged / D5R_PROPOSAL_NAME,
                     staged / D5_PROPOSAL_NAME, common[1], staged / D5_PREVIOUS_ANALYSIS,
                     staged / D5R_PREVIOUS_ANALYSIS, staged / D6_PREVIOUS_ANALYSIS,
                     staged / D7_PREVIOUS_ANALYSIS, staged / D7_LATE_TIMELINE)
else:
    assert sys.argv[3] == "D8"
    specs = d8_specs(read_json(config), proposal, staged / "input.json", common[0],
                     staged / D2R110_PROPOSAL_NAME, staged / D8_PREVIOUS_ANALYSIS)
print(json.dumps(specs))
"""
    result = remote(
        ssh,
        [
            "env",
            f"PYTHONPATH={checkout / 'src'}",
            python,
            "-c",
            program,
            checkout / REMOTE_CONFIG,
            proposal,
            role,
        ],
    )
    observed = json.loads(result.stdout)
    require(observed == specs, f"{role} staged node plan differs from controller preflight")
    return observed


def snapshot_previous(history, role, phase, args, commit):
    previous = [
        (path, read_json(path / "planned.json"), audit_attempt(path))
        for path in submitted_attempts(history)
    ]
    used = sum(row["phase"] == phase for _, row, _ in previous)
    require(used < args["max_attempts"], "M5 phase attempt budget exhausted")
    completed = {
        (row["phase"], row["role"]): (path, row)
        for path, row, audit in previous
        if audit["complete"] and row["execution_commit"] == commit
    }
    require((phase, role) not in completed, "this approved M5 role already completed")
    return completed


def transfer(alias, path, output, transport):
    require(not output.exists(), "node evidence destination already exists")
    batch = uuid.uuid4().hex
    for attempt in range(1, TRANSFER_ATTEMPTS + 1):
        staging = transport / f"{batch}-node-partial-{attempt}"
        argv = [
            "scp",
            "-q",
            "-r",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{alias}:{path}",
            str(staging),
        ]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
            data = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            data = dict(returncode=None, stdout="", stderr=str(error))
        durable_write(transport / f"{batch}-node-{attempt}.json", {**data, "command": argv})
        if data["returncode"] == 0:
            require(staging.is_dir(), "successful transport did not create node evidence")
            staging.rename(output)
            fsync_directory(output.parent)
            return
        if attempt < TRANSFER_ATTEMPTS:
            time.sleep(1)
    raise ValueError("node evidence transfer retries exhausted")


def fetch_evidence(alias, ssh, remote_attempt, log_file, destination):
    transport = destination / "transport"
    transport.mkdir(exist_ok=True)
    if not (destination / "node").exists():
        transfer(alias, remote_attempt, destination / "node", transport)
    if (destination / "slurm.log").exists():
        return
    batch = uuid.uuid4().hex
    for attempt in range(1, TRANSFER_ATTEMPTS + 1):
        row = remote_query(ssh, shlex.join(["cat", str(log_file)]), "transport")
        log = row.pop("stdout")
        if row["returncode"] == 0:
            row["stdout_sha256"] = hashlib.sha256(log.encode()).hexdigest()
        else:
            row["stdout"] = log
        durable_write(transport / f"{batch}-slurm-log-{attempt}.json", row)
        if row["returncode"] == 0:
            staging = transport / f"{batch}-slurm-log"
            staging.write_bytes(log.encode())
            staging.rename(destination / "slurm.log")
            fsync_directory(destination)
            return
        if attempt < TRANSFER_ATTEMPTS:
            time.sleep(1)
    raise ValueError("Slurm log transfer retries exhausted")


def recover_release(ssh, destination, submission):
    record, ready, _, job = submission
    require(record["controller_boot_id"] == controller_boot_id(), "controller boot changed")
    require(type(ready["pid"]) is int and ready["pid"] > 1, "invalid original watcher PID")
    try:
        os.kill(ready["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("original watcher PID is still live")
    recovery = destination / "recovery"
    require(not recovery.exists(), "existing recovery evidence cannot be overwritten")
    transport = destination / "transport"
    transport.mkdir(exist_ok=True)
    staging = transport / f"recovery-{uuid.uuid4().hex}"
    staging.mkdir()
    domain = record["clock_domain"]
    started = time.monotonic_ns()
    terminal = remote_query(ssh, f"scontrol show job {job}", domain)
    durable_write(staging / "slurm-final.json", terminal)
    slurm_terminal(terminal, job, record["job_name"])
    with (staging / "queue-watch.jsonl").open("x") as stream:

        def journal_query(command):
            row = remote_query(ssh, command, domain)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            return row

        records, clear = observe_clear_queue(journal_query, job, time.sleep)
    durable_write(staging / "active-final.json", records[-1])
    require(clear, "recovery full queue retries exhausted")
    durable_write(
        staging / "release.json",
        {
            "controller_boot_id": record["controller_boot_id"],
            "receipt_sha256": digest(destination / "submission.json"),
            "intent_sha256": digest(destination / "submission-intent.json"),
            "clock_domain": domain,
            "original_pid_not_live": True,
            "started_monotonic_ns": started,
            "observed_monotonic_ns": time.monotonic_ns(),
            "terminal_sha256": digest(staging / "slurm-final.json"),
            "active_sha256": digest(staging / "active-final.json"),
            "queue_journal_sha256": digest(staging / "queue-watch.jsonl"),
        },
    )
    staging.rename(recovery)
    fsync_directory(destination)
    audit_controller_release(destination, submission)


def resume(attempt):
    require(id_ok(attempt), "invalid attempt ID")
    destination = HISTORY / attempt
    with (HISTORY.parent / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submission = audit_submission(destination)
        record, _, _, job = submission
        require(
            record.get("resource_schema_version") == RESOURCE_SCHEMA_VERSION,
            "read-only recovery requires a durable original submission intent",
        )
        alias, paths, _ = connection()
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", alias]
        root = paths["m5_checkout"] / "runs" / TASK
        remote_attempt = root / "node" / attempt
        require(
            str(remote_attempt) == record["remote_attempt"]
            and record["job_name"] == JOB_PREFIX + attempt,
            "recovery path or job differs from original attempt",
        )
        try:
            audit_controller_release(destination, submission)
        except (KeyError, TypeError, ValueError, OSError):
            recover_release(ssh, destination, submission)
        fetch_evidence(alias, ssh, remote_attempt, root / "slurm" / (attempt + ".log"), destination)
        result = audit_attempt(destination)
        print(f"Job {job} release evidence recovered ({result['state']}/{result['exit_code']})")


def submit(args):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT),
        "local checkout dirty",
    )
    auth = read_json(args.authorization)
    allowed = check_authorization(auth, CONFIG, commit, args.phase, freeze_path=args.freeze)
    config = read_json(CONFIG)
    calibration_roles = qualification_roles(config)
    require(
        config["resource_schema_version"] == RESOURCE_SCHEMA_VERSION,
        "configuration M5 resource schema differs",
    )
    require(args.role in allowed and id_ok(args.attempt), "attempt or role outside authorization")
    HISTORY.mkdir(parents=True, exist_ok=True)
    with (HISTORY.parent / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        history = audit_history(HISTORY)
        completed = snapshot_previous(HISTORY, args.role, args.phase, auth, commit)
        diagnostic = args.role in DIAGNOSTIC_ROLES
        stage_index = 0
        retained_analyses = {}
        if diagnostic:
            previous_slots = {
                D1_ROLE: 5,
                D2_ROLE: 6,
                D2R_ROLE: 7,
                D2R110_ROLE: 8,
                D3_ROLE: 9,
                D4_ROLE: 10,
                D5_ROLE: 11,
                D5R_ROLE: 12,
                D6_ROLE: 13,
                D7_ROLE: 14,
                D8_ROLE: 15,
            }
            require(
                len(history) == previous_slots[args.role]
                and all(row["phase"] == "calibration" for row in history),
                "diagnostic history is outside its single approved allocation",
            )
            old_c3 = HISTORY / "c3-01" / "node"
            input_table = old_c3 / "mixed-2-endpoint" / "input.json"
            old_result = old_c3 / "calibration-result.json"
            diagnostic_inputs = {"input.json": input_table}
            if args.role == D1_ROLE:
                proposal = HISTORY.parent / "c3-retest-review" / "proposed-matrix.json"
                d1_specs(read_json(CONFIG), proposal, input_table, old_result)
            elif args.role == D2_ROLE:
                require(
                    history[-1]["role"] == D1_ROLE and history[-1]["complete"],
                    "D2 requires the completed D1 diagnostic",
                )
                review = HISTORY.parent / "d2-review"
                proposal = review / "proposed-d2.json"
                require(
                    digest(review / "d1-order-analysis.json") == D2_D1_REANALYSIS_SHA,
                    "D2 D1 reanalysis evidence differs",
                )
                d2_specs(read_json(CONFIG), proposal, input_table, old_result)
            elif args.role == D2R_ROLE:
                require(
                    history[-1]["role"] == D2_ROLE
                    and history[-1] == audit_attempt(HISTORY / "d2-01", require_completed=True),
                    "D2-R requires the retained D2 allocation and release",
                )
                proposal = CONFIG.with_name(D2R_PLAN_FILENAME)
                retained_analysis = HISTORY.parent / "d2-independent-review" / "analysis.json"
                bind_d2r_inputs(auth, proposal, retained_analysis)
                d2r_specs(read_json(CONFIG), proposal, input_table, old_result)
                retained_analyses[D2R_PREVIOUS_ANALYSIS] = retained_analysis
            elif args.role == D2R110_ROLE:
                require(
                    history[-1]["role"] == D2R_ROLE
                    and history[-1] == audit_attempt(HISTORY / "d2-r-01", require_completed=True),
                    "D2-R110 requires the retained D2-R allocation and release",
                )
                proposal = CONFIG.with_name(D2R110_PLAN_FILENAME)
                retained_analysis = HISTORY.parent / "d2-independent-review" / "analysis.json"
                retained_d2r = HISTORY.parent / "d2-r-review" / "analysis.json"
                bind_d2r110_inputs(auth, proposal, retained_analysis, retained_d2r)
                d2r110_specs(read_json(CONFIG), proposal, input_table, old_result)
                retained_analyses = {
                    D2R_PREVIOUS_ANALYSIS: retained_analysis,
                    D2R110_PREVIOUS_ANALYSIS: retained_d2r,
                }
            elif args.role == D3_ROLE:
                previous = HISTORY / "d2-r110-01"
                require(
                    history[-1]["role"] == D2R110_ROLE
                    and history[-1] == audit_attempt(previous, require_completed=True),
                    "D3 requires the retained D2-R110 allocation and release",
                )
                proposal = CONFIG.with_name(D3_PLAN_FILENAME)
                retained_analysis = HISTORY.parent / "d2-independent-review" / "analysis.json"
                retained_d2r = HISTORY.parent / "d2-r-review" / "analysis.json"
                retained_d2r110 = HISTORY.parent / "d2-r110-review" / "analysis.json"
                bind_d3_inputs(auth, proposal, retained_analysis, retained_d2r, retained_d2r110)
                prior_report = read_json(retained_d2r110)
                require(
                    prior_report["resource_release"]
                    == json.loads(json.dumps(history[-1], default=str))
                    and prior_report["execution_commit"]
                    == read_json(previous / "planned.json")["execution_commit"],
                    "D3 retained D2-R110 analysis is detached from the prior release",
                )
                d3_specs(config, proposal, input_table, old_result)
                retained_analyses = {
                    D2R_PREVIOUS_ANALYSIS: retained_analysis,
                    D2R110_PREVIOUS_ANALYSIS: retained_d2r,
                    D3_PREVIOUS_ANALYSIS: retained_d2r110,
                }
            elif args.role == D4_ROLE:
                previous = HISTORY / "d3-01"
                require(
                    history[-1]["role"] == D3_ROLE
                    and history[-1] == audit_attempt(previous, require_completed=True),
                    "D4 requires the retained D3 allocation and release",
                )
                proposal = CONFIG.with_name(D4_PLAN_FILENAME)
                inherited = CONFIG.with_name(D3_PLAN_FILENAME)
                prior_analysis = HISTORY.parent / "d3-review/analysis.json"
                input_6 = HISTORY.parent / "d4-review/prepared" / D4_INPUT_6_NAME
                diagnostic_run_specs = d4_specs(
                    config, proposal, input_table, input_6, old_result, inherited, prior_analysis
                )
                report = read_json(prior_analysis)
                require(
                    report["resource_release"] == json.loads(json.dumps(history[-1], default=str))
                    and report["execution_commit"]
                    == read_json(previous / "planned.json")["execution_commit"],
                    "D4 retained D3 analysis is detached from the prior release",
                )
                diagnostic_inputs = {D4_INPUT_8_NAME: input_table, D4_INPUT_6_NAME: input_6}
                retained_analyses = {
                    D3_PROPOSAL_NAME: inherited,
                    D4_PREVIOUS_ANALYSIS: prior_analysis,
                }
            elif args.role in (D6_ROLE, D7_ROLE):
                previous = HISTORY / "d5-r-01"
                require(
                    history[12]["role"] == D5R_ROLE
                    and history[12] == audit_attempt(previous, require_completed=True),
                    "D6 requires the retained D5-R allocation and release",
                )
                inherited_d6 = CONFIG.with_name(D6_PLAN_FILENAME)
                proposal = (
                    CONFIG.with_name(D7_PLAN_FILENAME) if args.role == D7_ROLE else inherited_d6
                )
                inherited_d5r = CONFIG.with_name(D5R_PLAN_FILENAME)
                inherited_d5 = CONFIG.with_name(D5_PLAN_FILENAME)
                inherited_d3 = CONFIG.with_name(D3_PLAN_FILENAME)
                previous_d4 = HISTORY.parent / "d4-review/analysis.json"
                previous_d5 = HISTORY.parent / "d5-review/analysis.json"
                previous_d5r = HISTORY.parent / "d5-r-review/analysis.json"
                lineage = (
                    inherited_d5r,
                    inherited_d5,
                    inherited_d3,
                    previous_d4,
                    previous_d5,
                    previous_d5r,
                )
                report = read_json(previous_d5r)
                require(
                    report["resource_release"] == json.loads(json.dumps(history[12], default=str))
                    and report["execution_commit"]
                    == read_json(previous / "planned.json")["execution_commit"],
                    "D6 retained D5-R analysis is detached from the prior release",
                )
                retained_analyses = {
                    D3_PROPOSAL_NAME: inherited_d3,
                    D5_PROPOSAL_NAME: inherited_d5,
                    D5R_PROPOSAL_NAME: inherited_d5r,
                    D5_PREVIOUS_ANALYSIS: previous_d4,
                    D5R_PREVIOUS_ANALYSIS: previous_d5,
                    D6_PREVIOUS_ANALYSIS: previous_d5r,
                }
                if args.role == D7_ROLE:
                    previous_d6 = HISTORY.parent / "d6-review/analysis-d6-02.json"
                    late_timeline = HISTORY.parent / "d6-review/late-independent-all-leaves.json"
                    diagnostic_run_specs = d7_specs(
                        config,
                        proposal,
                        input_table,
                        old_result,
                        inherited_d6,
                        *lineage,
                        previous_d6,
                        late_timeline,
                    )
                    report = read_json(previous_d6)
                    require(
                        history[-1]["role"] == D6_ROLE
                        and history[-1] == audit_attempt(HISTORY / "d6-02", require_completed=True)
                        and report["resource_release"]
                        == json.loads(json.dumps(history[-1], default=str))
                        and report["execution_commit"]
                        == read_json(HISTORY / "d6-02/planned.json")["execution_commit"],
                        "D7 retained D6 analysis is detached from the prior release",
                    )
                    retained_analyses.update(
                        {
                            D6_PROPOSAL_NAME: inherited_d6,
                            D7_PREVIOUS_ANALYSIS: previous_d6,
                            D7_LATE_TIMELINE: late_timeline,
                        }
                    )
                else:
                    diagnostic_run_specs = d6_specs(
                        config, proposal, input_table, old_result, *lineage
                    )
            elif args.role == D8_ROLE:
                previous = HISTORY / "d7-01"
                require(
                    history[-1]["role"] == D7_ROLE
                    and history[-1] == audit_attempt(previous, require_completed=True),
                    "D8 requires the retained D7 allocation and release",
                )
                proposal = CONFIG.with_name(D8_PLAN_FILENAME)
                inherited = CONFIG.with_name(D2R110_PLAN_FILENAME)
                previous_d7 = HISTORY.parent / "d7-review/analysis-d7-01.json"
                diagnostic_run_specs = d8_specs(
                    config, proposal, input_table, old_result, inherited, previous_d7
                )
                report = read_json(previous_d7)
                require(
                    report["resource_release"] == json.loads(json.dumps(history[-1], default=str))
                    and report["execution_commit"]
                    == read_json(previous / "planned.json")["execution_commit"],
                    "D8 retained D7 analysis is detached from the prior release",
                )
                retained_analyses = {
                    D2R110_PROPOSAL_NAME: inherited,
                    D8_PREVIOUS_ANALYSIS: previous_d7,
                }
            else:
                is_d5r = args.role == D5R_ROLE
                prior_role = D5_ROLE if is_d5r else D4_ROLE
                previous = HISTORY / ("d5-01" if is_d5r else "d4-01")
                require(
                    history[-1]["role"] == prior_role
                    and history[-1] == audit_attempt(previous, require_completed=True),
                    f"{args.role} requires the retained {prior_role} allocation and release",
                )
                d5_proposal = CONFIG.with_name(D5_PLAN_FILENAME)
                inherited = CONFIG.with_name(D3_PLAN_FILENAME)
                d4_analysis = HISTORY.parent / "d4-review/analysis.json"
                retained_analyses = {
                    D3_PROPOSAL_NAME: inherited,
                    D5_PREVIOUS_ANALYSIS: d4_analysis,
                }
                if is_d5r:
                    proposal = CONFIG.with_name(D5R_PLAN_FILENAME)
                    prior_analysis = HISTORY.parent / "d5-review/analysis.json"
                    diagnostic_run_specs = d5r_specs(
                        config,
                        proposal,
                        input_table,
                        old_result,
                        d5_proposal,
                        inherited,
                        d4_analysis,
                        prior_analysis,
                    )
                    retained_analyses.update(
                        {
                            D5_PROPOSAL_NAME: d5_proposal,
                            D5R_PREVIOUS_ANALYSIS: prior_analysis,
                        }
                    )
                else:
                    proposal, prior_analysis = d5_proposal, d4_analysis
                    diagnostic_run_specs = d5_specs(
                        config, proposal, input_table, old_result, inherited, prior_analysis
                    )
                report = read_json(prior_analysis)
                require(
                    report["resource_release"] == json.loads(json.dumps(history[-1], default=str))
                    and report["execution_commit"]
                    == read_json(previous / "planned.json")["execution_commit"],
                    f"{args.role} retained {prior_role} analysis is detached from the prior release",
                )
        else:
            from m5_c3_bundle import qualified_calibration_stage

            if args.phase == "calibration" and calibration_roles != CALIBRATION:
                verify_split_qualification(CONFIG, HISTORY, args.authorization, audits=history)
            stage_index = (
                calibration_roles.index(args.role)
                if args.phase == "calibration"
                else len(calibration_roles)
            )
            qualified_reports = {}
            for stage in calibration_roles[:stage_index]:
                require(
                    ("calibration", stage) in completed,
                    f"required qualified calibration stage {stage} missing",
                )
                path, _ = completed[("calibration", stage)]
                qualified_reports[stage] = qualified_calibration_stage(
                    CONFIG,
                    path,
                    stage,
                    commit,
                    rates=qualified_reports.get("C2", {}).get("selected_rates"),
                    dress_reports=(qualified_reports["C4"], qualified_reports["C5"])
                    if stage == "C6"
                    else (),
                )
            if args.phase == "formal":
                freeze = read_json(args.freeze)
                for stage in calibration_roles:
                    if stage in SPLIT_C3_ROLES:
                        continue
                    path, _ = completed[("calibration", stage)]
                    report = path / "node/calibration-result.json"
                    report_bytes = report.read_bytes()
                    require(
                        json.loads(report_bytes) == qualified_reports[stage]
                        and hashlib.sha256(report_bytes).hexdigest()
                        == freeze["calibration_sha256"][stage],
                        "formal freeze detached from qualified calibration",
                    )
                if calibration_roles != CALIBRATION:
                    stages = {
                        stage: completed[("calibration", stage)][0] for stage in calibration_roles
                    }
                    bundle = c3_bundle(CONFIG, HISTORY, stages, commit)
                    require(
                        c3_bundle_digest(bundle) == freeze["calibration_sha256"]["C3"],
                        "formal freeze detached from split C3 qualification",
                    )
                c2, _ = completed[("calibration", "C2")]
                c6, _ = completed[("calibration", "C6")]
                bind_formal_calibration(freeze, c2, c6)
                validate_formal_tables(c6 / "node/tables", config, config_matrix(config), freeze)

        alias, paths, partition = connection()
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", alias]
        checkout = paths["m5_checkout"]
        remote_commit = remote(ssh, ["git", "-C", checkout, "rev-parse", "HEAD"]).stdout.strip()
        require(remote_commit == commit, "remote checkout does not match execution commit")
        require(
            not remote(ssh, ["git", "-C", checkout, "status", "--porcelain"]).stdout,
            "remote checkout dirty",
        )
        root = checkout / "runs" / TASK
        remote_attempt = root / "node" / args.attempt
        remote_auth = root / "authorized" / args.attempt / "authorization.json"
        require(
            remote(ssh, ["test", "!", "-e", remote_attempt], check=False).returncode == 0,
            "remote attempt directory already exists or cannot be checked",
        )
        upload(ssh, remote_auth, args.authorization.read_text())
        aux = "-"
        auxiliary_hash = None
        node_specs = None
        if diagnostic:
            staged_names = {
                D2R_ROLE: D2R_PROPOSAL_NAME,
                D2R110_ROLE: D2R110_PROPOSAL_NAME,
                D3_ROLE: D3_PROPOSAL_NAME,
                D4_ROLE: D4_PROPOSAL_NAME,
                D5_ROLE: D5_PROPOSAL_NAME,
                D5R_ROLE: D5R_PROPOSAL_NAME,
                D6_ROLE: D6_PROPOSAL_NAME,
                D7_ROLE: D7_PROPOSAL_NAME,
                D8_ROLE: D8_PROPOSAL_NAME,
            }
            name = staged_names.get(args.role, proposal.name)
            aux = remote_auth.parent / name
            upload(ssh, aux, proposal.read_bytes().decode("utf-8"))
            for filename, table in diagnostic_inputs.items():
                upload(ssh, aux.parent / filename, table.read_bytes().decode("utf-8"))
            upload(
                ssh, aux.parent / "original-c3-result.json", old_result.read_bytes().decode("utf-8")
            )
            for filename, analysis in retained_analyses.items():
                upload(ssh, aux.parent / filename, analysis.read_bytes().decode("utf-8"))
            auxiliary_hash = digest(proposal)
            if args.role in (D4_ROLE, D5_ROLE, D5R_ROLE, D6_ROLE, D7_ROLE, D8_ROLE):
                node_specs = preflight_diagnostic_stage(
                    ssh,
                    paths["remote_gpu_environment"] / "bin/python",
                    checkout,
                    aux,
                    diagnostic_run_specs,
                    args.role,
                )
        elif args.phase == "formal":
            _, previous = completed[("calibration", "C6")]
            aux = PurePosixPath(previous["remote_attempt"]) / "formal-freeze.json"
            upload(ssh, aux, args.freeze.read_text())
            auxiliary_hash = digest(args.freeze)
        elif stage_index >= 2:
            prerequisites = ("C2", "C4", "C5") if args.role == "C6" else ("C2",)
            reports = []
            for stage in prerequisites:
                directory, previous = completed[("calibration", stage)]
                local = directory / "node/calibration-result.json"
                report_bytes = local.read_bytes()
                require(
                    json.loads(report_bytes) == qualified_reports[stage],
                    f"{stage} qualified report changed after audit",
                )
                reports.append(
                    {
                        "stage": stage,
                        "path": str(PurePosixPath(previous["remote_attempt"]) / local.name),
                        "sha256": hashlib.sha256(report_bytes).hexdigest(),
                    }
                )
            aux = root / "authorized" / args.attempt / "dependencies.json"
            dependencies = json.dumps({"config_sha256": digest(CONFIG), "reports": reports}) + "\n"
            upload(ssh, aux, dependencies)
            auxiliary_hash = hashlib.sha256(dependencies.encode()).hexdigest()

        log_path = root / "slurm"
        remote(ssh, ["mkdir", "-p", log_path])
        destination = HISTORY / args.attempt
        require(not destination.exists(), "local attempt already exists")
        clock_domain = uuid.uuid4().hex
        boot = controller_boot_id()
        before_queue, clear = observe_clear_queue(
            lambda command: remote_query(ssh, command, clock_domain), None, time.sleep
        )
        job_name = JOB_PREFIX + args.attempt
        staging = HISTORY.parent / f"pre-intent-{args.attempt}-{clock_domain}"
        staging.mkdir(mode=0o700)
        (staging / "authorization.json").write_bytes(args.authorization.read_bytes())
        if node_specs is not None:
            durable_write(
                staging / f"{args.role.lower()}-stage-preflight.json", {"runs": node_specs}
            )
        durable_write(staging / "pre-submission-queue.json", before_queue)
        durable_write(
            staging / "planned.json",
            {
                "resource_schema_version": RESOURCE_SCHEMA_VERSION,
                "controller_boot_id": boot,
                "attempt": args.attempt,
                "task_id": TASK,
                "job_name": job_name,
                "execution_commit": commit,
                "config_sha256": digest(CONFIG),
                "authorization_sha256": digest(args.authorization),
                "auxiliary_sha256": auxiliary_hash,
                "phase": args.phase,
                "role": args.role,
                "clock_domain": clock_domain,
                "remote_attempt": str(remote_attempt),
                "requested_minutes": 20,
                "pre_submission_queue": before_queue[-1]["stdout"],
            },
        )
        staging.rename(destination)
        fsync_directory(HISTORY)
        require(clear, "pre-submission full queue retries exhausted")
        watcher_dir = destination / "watcher"
        submission = destination / "submission.json"
        watch_argv = [
            sys.executable,
            str(ROOT / "scripts/watch_m3_campaign.py"),
            "--ssh-alias",
            alias,
            "--campaign-id",
            TASK,
            "--campaign-sha256",
            digest(CONFIG),
            "--attempt-id",
            args.attempt,
            "--project-commit",
            commit,
            "--clock-domain",
            clock_domain,
            "--output",
            str(watcher_dir),
            "--submission",
            str(submission),
            "--timeout",
            "86400",
            "--m5-release",
        ]
        with (destination / "watcher.log").open("xb") as log:
            watch = subprocess.Popen(watch_argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                ready_path = watcher_dir / "monitor-ready.json"
                ready_deadline = time.monotonic() + 12
                while (
                    not ready_path.exists()
                    and watch.poll() is None
                    and time.monotonic() < ready_deadline
                ):
                    time.sleep(0.02)
                require(
                    ready_path.exists() and watch.poll() is None, "independent watcher not ready"
                )
                ready = read_json(ready_path)
                require(
                    ready["ready"] is True and ready["pid"] == watch.pid,
                    "watcher readiness mismatch",
                )
                invoke = [
                    "sbatch",
                    "--parsable",
                    f"--partition={partition}",
                    f"--job-name={job_name}",
                    f"--chdir={checkout}",
                    f"--output={log_path / (args.attempt + '.log')}",
                    checkout / "scripts/m5_gpu.sbatch",
                    paths["remote_gpu_environment"] / "bin/python",
                    checkout / REMOTE_CONFIG,
                    remote_auth,
                    args.phase,
                    args.role,
                    args.attempt,
                    remote_attempt,
                    paths["remote_model_directory"],
                    paths["remote_cache_directory"],
                    aux,
                ]
                invoke = list(map(str, invoke))
                begun = time.monotonic_ns()
                intent = destination / "submission-intent.json"
                durable_write(
                    intent,
                    {
                        "argv": invoke,
                        "clock_domain": clock_domain,
                        "command_started_ns": begun,
                        "planned_sha256": digest(destination / "planned.json"),
                        "watcher_ready_sha256": digest(ready_path),
                        "pre_submission_queue_sha256": digest(
                            destination / "pre-submission-queue.json"
                        ),
                    },
                )
                try:
                    sent = remote(ssh, invoke, check=False)
                    receipt = dict(
                        returncode=sent.returncode, stdout=sent.stdout, stderr=sent.stderr
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    receipt = dict(returncode=None, stdout="", stderr=str(error))
                durable_write(
                    submission,
                    {
                        **receipt,
                        "command_started_ns": begun,
                        "clock_domain": clock_domain,
                        "observed_monotonic_ns": time.monotonic_ns(),
                        "argv": invoke,
                        "intent_sha256": digest(intent),
                    },
                )
                watch_code = watch.wait()
            finally:
                if watch.poll() is None:
                    watch.terminate()
                    watch.wait(timeout=10)
                durable_write(
                    destination / "watcher-exit.json",
                    {
                        "pid": watch.pid,
                        "returncode": watch.returncode,
                        "clock_domain": clock_domain,
                        "observed_monotonic_ns": time.monotonic_ns(),
                    },
                )
        require(watch_code == 0, "watcher did not verify terminal and full queue")
        audit_controller_release(destination)
        fetch_evidence(alias, ssh, remote_attempt, log_path / (args.attempt + ".log"), destination)
        result = audit_attempt(destination, require_completed=True)
        print(f"Job {result['job_id']} released and bound to {args.role}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorization", type=Path)
    p.add_argument("--phase", choices=("calibration", "formal"))
    p.add_argument("--role")
    p.add_argument("--attempt", required=True)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    try:
        if args.resume:
            require(
                all(
                    value is None
                    for value in (args.authorization, args.phase, args.role, args.freeze)
                ),
                "read-only resume only accepts an attempt ID",
            )
            resume(args.attempt)
        else:
            require(
                all(value is not None for value in (args.authorization, args.phase, args.role)),
                "submission requires authorization, phase and role",
            )
            submit(args)
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(1, f"M5 controller stopped: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
