"""Independent local Slurm terminal watcher; ready before the submitter starts sbatch.

All ready/submission/query observation clocks belong to the local controller
host. Slurm StartTime/EndTime remain in the scheduler's separately recorded domain.
This command never allocates, cancels, installs, or runs model code remotely.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "ssh-alias",
        "campaign-id",
        "campaign-sha256",
        "attempt-id",
        "project-commit",
        "clock-domain",
    ):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ready = {
        "ready": True,
        "campaign_id": args.campaign_id,
        "campaign_sha256": args.campaign_sha256,
        "attempt_id": args.attempt_id,
        "project_commit": args.project_commit,
        "clock_domain": args.clock_domain,
        "observed_monotonic_ns": time.monotonic_ns(),
        "observed_unix_ns": time.time_ns(),
        "pid": __import__("os").getpid(),
    }
    with (args.output / "monitor-ready.json").open("x") as stream:
        json.dump(ready, stream, indent=2)
        stream.write("\n")
    print("WATCH_READY", flush=True)
    deadline = time.monotonic() + args.timeout
    job = None
    previous = None

    def query(command):
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.ssh_alias, command]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
            data = dict(
                command=argv,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            data = dict(command=argv, returncode=None, stdout="", stderr="query timeout")
        return {
            **data,
            "clock_domain": args.clock_domain,
            "observed_monotonic_ns": time.monotonic_ns(),
            "observed_unix_ns": time.time_ns(),
        }

    with (args.output / "watch.jsonl").open("x") as stream:
        while time.monotonic() < deadline:
            if job is None:
                if not args.submission.exists():
                    time.sleep(0.1)
                    continue
                submission = json.loads(args.submission.read_text())
                if submission.get("returncode") != 0:
                    return 3
                job = submission["stdout"].strip().split(";")[0]
                if not job.isdigit():
                    raise RuntimeError("Invalid submitted job identity")
            data = query(f"scontrol show job {job}")
            stream.write(json.dumps(data) + "\n")
            stream.flush()
            fields = dict(item.split("=", 1) for item in data["stdout"].split() if "=" in item)
            state = fields.get("JobState")
            if state != previous:
                print(f"SLURM_STATE {state}", flush=True)
                previous = state
            if data["returncode"] == 0 and state in {
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
            }:
                active = query('squeue -h -o "%i|%T|%j"')
                for name, value in [
                    ("slurm-final.json", data),
                    ("active-final.json", active),
                ]:
                    with (args.output / name).open("x") as target:
                        json.dump(value, target, indent=2)
                        target.write("\n")
                print(f"TERMINAL_CAPTURED {state}", flush=True)
                return 0 if active["returncode"] == 0 else 4
            time.sleep(1)
    return 2


if __name__ == "__main__":
    sys.exit(main())
