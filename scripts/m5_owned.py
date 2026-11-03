"""Register the collector child before it can start an M5 engine."""

import json
import os
import sys
import time
from pathlib import Path

from m4_processes import atomic_write

TASK = "M5-paired-performance"


def main():
    owner, ack, *command = sys.argv[1:]
    owner, ack = Path(owner), Path(ack)
    pid = os.getpid()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or os.getpgrp() != pid or not command or owner.exists() or ack.exists():
        raise RuntimeError("M5 requires a new allocation-owned child group")
    atomic_write(
        owner,
        {
            "task_id": TASK,
            "job_id": job,
            "pid": pid,
            "pgid": pid,
            "uid": os.getuid(),
            "collector_pid": os.getppid(),
            "registered_monotonic_ns": time.monotonic_ns(),
        },
    )
    deadline = time.monotonic() + 12
    while not ack.is_file():
        if time.monotonic() > deadline or os.getppid() == 1:
            raise RuntimeError("M5 owner acknowledgement not received")
        time.sleep(0.01)
    if json.loads(ack.read_text()) != {"job_id": job, "pgid": pid}:
        raise RuntimeError("M5 owner acknowledgement differs")
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
