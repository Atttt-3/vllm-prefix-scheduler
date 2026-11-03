"""Real local subprocess control: kill the launcher, then release its detached child."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from m4_processes import cleanup_engines

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vllm_prefix_scheduler.run_evidence import stop_process  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output.resolve()
    heartbeat = directory / "heartbeat.txt"
    child = directory / "child.py"
    child.write_text(
        "import pathlib,sys,time\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        " with p.open('a') as f:f.write('tick\\n');f.flush()\n"
        " time.sleep(.03)\n"
    )
    parent = directory / "parent.py"
    parent.write_text(
        "import json,pathlib,signal,subprocess,sys,time\n"
        "signal.signal(signal.SIGTERM,lambda *_: time.sleep(20))\n"
        "root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])\n"
        "owner=out/'child.owner.json'; ack=out/'child.ack.json'\n"
        "p=subprocess.Popen([sys.executable,str(root/'scripts/m4_processes.py'),'--owner',str(owner),'--ack',str(ack),'--engine-id','cleanup-control','--',sys.executable,str(out/'child.py'),str(out/'heartbeat.txt')],start_new_session=True)\n"
        "while not owner.exists():time.sleep(.01)\n"
        "ack.write_text(json.dumps({'pgid':p.pid}))\n"
        "while True:time.sleep(1)\n"
    )
    process = None
    cleanup = []
    result = {"task_id": "M4-scheduler-stability", "layer": "local", "accepted": False}
    try:
        process = subprocess.Popen(
            [sys.executable, str(parent), str(ROOT), str(directory)], start_new_session=True
        )
        deadline = time.monotonic() + 10
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not heartbeat.exists():
            raise RuntimeError("real child never started")
        start = time.monotonic()
        errors = stop_process(process)
        result.update(
            launcher_returncode=process.poll(),
            collector_cleanup_errors=errors,
            collector_cleanup_seconds=time.monotonic() - start,
        )
        before = heartbeat.stat().st_size
        time.sleep(0.12)
        after = heartbeat.stat().st_size
        if after <= before:
            raise RuntimeError("control did not preserve detached child after launcher kill")
        cleanup = cleanup_engines(directory, timeout=2)
        if not cleanup or any(row["remaining"] for row in cleanup):
            raise RuntimeError("independent M4 cleanup failed")
        final_size = heartbeat.stat().st_size
        time.sleep(0.12)
        if heartbeat.stat().st_size != final_size:
            raise RuntimeError("owned child still running after M4 cleanup")
        result.update(
            accepted=True, detached_child_was_observed=True, heartbeat_stopped=True, cleanup=cleanup
        )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        cleanup_engines(directory, timeout=1)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Real local detached-engine cleanup control passed")


if __name__ == "__main__":
    main()
