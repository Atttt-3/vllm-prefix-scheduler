"""M4-only ownership handshake and cleanup for separately grouped engine children."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from run_m3_suite import owned_processes


def atomic_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def cleanup_engines(directory, *, timeout=20):
    """The allocation owns this exact registry, even if the suite was SIGKILLed."""
    records = []
    directory = Path(directory)
    for path in sorted(directory.glob("*.owner.json")):
        owner = json.loads(path.read_text())
        pgid = owner["pgid"]
        if owner.get("task_id") != "M4-scheduler-stability" or owner.get("uid") != os.getuid():
            raise RuntimeError("Foreign M4 process owner record")
        if type(pgid) is not int or pgid <= 1 or owner.get("pid") != pgid or pgid == os.getpgrp():
            raise RuntimeError("Unsafe M4 engine process group")
        if owner.get("job_id") != os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Process owner belongs to a different allocation")
        before = owned_processes(pgid)
        if before:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + timeout
            while owned_processes(pgid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if owned_processes(pgid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 3
                while owned_processes(pgid) and time.monotonic() < deadline:
                    time.sleep(0.1)
        records.append(
            {
                "owner_file": str(path),
                "pgid": pgid,
                "before": before,
                "remaining": owned_processes(pgid),
            }
        )
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", type=Path, required=True)
    p.add_argument("--ack", type=Path, required=True)
    p.add_argument("--engine-id", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or os.getpid() != os.getpgrp():
        raise RuntimeError("M4 engine bootstrap requires a new owned process group")
    args.owner.parent.mkdir(parents=True, exist_ok=True)
    if args.owner.exists() or args.ack.exists():
        raise RuntimeError("M4 ownership path already exists")
    atomic_write(
        args.owner,
        {
            "task_id": "M4-scheduler-stability",
            "engine_id": args.engine_id,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "parent": os.getppid(),
            "uid": os.getuid(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "command": command,
            "registered_monotonic_ns": time.monotonic_ns(),
        },
    )
    deadline = time.monotonic() + 10
    while not args.ack.exists():
        if time.monotonic() >= deadline or os.getppid() == 1:
            return 125
        time.sleep(0.01)
    if json.loads(args.ack.read_text()).get("pgid") != os.getpgrp():
        raise RuntimeError("M4 process ownership acknowledgement differs")
    os.execv(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
