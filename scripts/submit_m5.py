"""Submit one explicitly approved M5 allocation and retain independent release evidence."""

import argparse
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

from m4_processes import atomic_write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_prefix_scheduler.m5_campaign import (  # noqa: E402
    CALIBRATION,
    D1_ROLE,
    JOB_PREFIX,
    TASK,
    check_authorization,
    digest,
    id_ok,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_d1 import d1_specs  # noqa: E402
from vllm_prefix_scheduler.m5_resources import audit_attempt, audit_history  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent / "vllm-prefix-scheduler/.local/workspace.json"
CONFIG = ROOT / "configs/m5-performance.json"
HISTORY = ROOT / "runs" / TASK / "attempts"
REMOTE_CONFIG = "configs/m5-performance.json"


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


def upload(ssh, path, content):
    # Create once, or prove this is the exact same staged input.
    stage = 'set -eu; umask 077; mkdir -p -- "$(dirname -- "$1")"; if [ -f "$1" ]; then cmp -s "$1" -; else (set -C; cat > "$1"); fi'
    remote(ssh, ["bash", "-c", stage, "m5-stage", path], data=content)


def snapshot_previous(history, role, phase, args, commit):
    previous = [
        (path, read_json(path / "planned.json"), audit_attempt(path))
        for path in sorted(history.iterdir())
        if path.is_dir()
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


def transfer(alias, path, output):
    require(not output.exists(), "node evidence destination already exists")
    staging = output.with_name(f"{output.name}-transfer-{uuid.uuid4().hex}")
    subprocess.run(
        [
            "scp",
            "-q",
            "-r",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{alias}:{path}",
            str(staging),
        ],
        check=True,
        timeout=600,
    )
    staging.rename(output)


def submit(args):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT),
        "local checkout dirty",
    )
    auth = read_json(args.authorization)
    allowed = check_authorization(auth, CONFIG, commit, args.phase, freeze_path=args.freeze)
    require(args.role in allowed and id_ok(args.attempt), "attempt or role outside authorization")
    HISTORY.mkdir(parents=True, exist_ok=True)
    with (HISTORY.parent / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        history = audit_history(HISTORY)
        completed = snapshot_previous(HISTORY, args.role, args.phase, auth, commit)
        diagnostic = args.role == D1_ROLE
        stage_index = 0
        if diagnostic:
            require(
                len(history) == 5 and all(row["phase"] == "calibration" for row in history),
                "D1 requires exactly one remaining original calibration allocation",
            )
            old_c3 = HISTORY / "c3-01" / "node"
            proposal = HISTORY.parent / "c3-retest-review" / "proposed-matrix.json"
            input_table = old_c3 / "mixed-2-endpoint" / "input.json"
            old_result = old_c3 / "calibration-result.json"
            d1_specs(read_json(CONFIG), proposal, input_table, old_result)
        else:
            stage_index = (
                CALIBRATION.index(args.role) if args.phase == "calibration" else len(CALIBRATION)
            )
            for stage in CALIBRATION[:stage_index]:
                require(
                    ("calibration", stage) in completed,
                    f"required qualified calibration stage {stage} missing",
                )

        alias, paths, partition = connection()
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", alias]
        checkout = paths["m5_checkout"]
        remote_commit = remote(ssh, ["git", "-C", checkout, "rev-parse", "HEAD"]).stdout.strip()
        require(remote_commit == commit, "remote checkout does not match execution commit")
        require(
            not remote(ssh, ["git", "-C", checkout, "status", "--porcelain"]).stdout,
            "remote checkout dirty",
        )
        before_queue = remote(ssh, ["squeue", "-h", "-o", "%i|%T|%j"])
        require(
            all(
                not line.split("|")[-1].startswith(JOB_PREFIX)
                for line in before_queue.stdout.splitlines()
            ),
            "an M5 job is visible in the full queue",
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
        if diagnostic:
            aux = remote_auth.parent / "proposed-matrix.json"
            upload(ssh, aux, proposal.read_bytes().decode("utf-8"))
            upload(ssh, aux.parent / "input.json", input_table.read_bytes().decode("utf-8"))
            upload(
                ssh, aux.parent / "original-c3-result.json", old_result.read_bytes().decode("utf-8")
            )
            auxiliary_hash = digest(proposal)
        elif args.phase == "formal":
            c6, previous = completed[("calibration", "C6")]
            freeze = read_json(args.freeze)
            for stage in CALIBRATION:
                path, _ = completed[("calibration", stage)]
                require(
                    digest(path / "node/calibration-result.json")
                    == freeze["calibration_sha256"][stage],
                    "formal freeze detached from qualified calibration",
                )
            for pair, wanted in freeze["tables"].items():
                require(
                    digest(c6 / "node/tables" / f"{pair}.json") == wanted,
                    "formal table hash differs",
                )
            aux = PurePosixPath(previous["remote_attempt"]) / "formal-freeze.json"
            upload(ssh, aux, args.freeze.read_text())
            auxiliary_hash = digest(args.freeze)
        elif stage_index >= 2:
            prerequisites = ("C2", "C4", "C5") if args.role == "C6" else ("C2",)
            reports = []
            for stage in prerequisites:
                directory, previous = completed[("calibration", stage)]
                local = directory / "node/calibration-result.json"
                result = read_json(local)
                require(result["status"] == "qualified", f"{stage} report is not qualified")
                reports.append(
                    {
                        "stage": stage,
                        "path": str(PurePosixPath(previous["remote_attempt"]) / local.name),
                        "sha256": digest(local),
                    }
                )
            aux = root / "authorized" / args.attempt / "dependencies.json"
            dependencies = json.dumps({"config_sha256": digest(CONFIG), "reports": reports}) + "\n"
            upload(ssh, aux, dependencies)
            auxiliary_hash = hashlib.sha256(dependencies.encode()).hexdigest()

        destination = HISTORY / args.attempt
        destination.mkdir()
        watcher_dir = destination / "watcher"
        submission = destination / "submission.json"
        clock_domain = uuid.uuid4().hex
        job_name = JOB_PREFIX + args.attempt
        (destination / "authorization.json").write_bytes(args.authorization.read_bytes())
        atomic_write(
            destination / "planned.json",
            {
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
                "pre_submission_queue": before_queue.stdout,
            },
        )
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
                # Create the Slurm log directory on the gateway before submission.
                log_path = root / "slurm"
                remote(ssh, ["mkdir", "-p", log_path])
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
                begun = time.monotonic_ns()
                try:
                    sent = remote(ssh, invoke, check=False)
                    receipt = dict(
                        returncode=sent.returncode, stdout=sent.stdout, stderr=sent.stderr
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    receipt = dict(returncode=None, stdout="", stderr=str(error))
                atomic_write(
                    submission,
                    {
                        **receipt,
                        "command_started_ns": begun,
                        "clock_domain": clock_domain,
                        "observed_monotonic_ns": time.monotonic_ns(),
                        "argv": list(map(str, invoke)),
                    },
                )
                watch_code = watch.wait()
            finally:
                if watch.poll() is None:
                    watch.terminate()
                    watch.wait(timeout=10)
                atomic_write(
                    destination / "watcher-exit.json",
                    {
                        "pid": watch.pid,
                        "returncode": watch.returncode,
                        "clock_domain": clock_domain,
                        "observed_monotonic_ns": time.monotonic_ns(),
                    },
                )
        require(watch_code == 0, "watcher did not verify terminal and full queue")
        transfer(alias, remote_attempt, destination / "node")
        slurm_log = remote(ssh, ["cat", log_path / (args.attempt + ".log")])
        (destination / "slurm.log").write_text(slurm_log.stdout)
        result = audit_attempt(destination, require_completed=True)
        print(f"Job {result['job_id']} released and bound to {args.role}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--phase", choices=("calibration", "formal"), required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--freeze", type=Path)
    args = p.parse_args()
    try:
        submit(args)
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(1, f"M5 controller stopped: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
