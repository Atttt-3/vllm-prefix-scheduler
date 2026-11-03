"""Serial M4 engine launcher with failure-preserving logs and owned-process cleanup."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

from m3_sources import collect_sources
from m4_processes import atomic_write, cleanup_engines


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=10)
    except Exception as error:
        return {"monotonic_ns": time.monotonic_ns(), "returncode": None, "error": str(error)}
    return {
        "monotonic_ns": time.monotonic_ns(),
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


def owned_processes(pgid):
    p = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,args="],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    return [
        line
        for line in p.stdout.splitlines()
        if len(line.split()) >= 3 and line.split()[2] == str(pgid)
    ]


def emit_process_exit(task_id, phase, result):
    from vllm_prefix_scheduler.m3_events import emit

    # The envelope identifies the launcher stream; the exited child is payload.
    emit(
        "m4_process_exit",
        task_id=task_id,
        layer="gpu",
        phase=phase,
        child_pid=result["pid"],
        **{key: value for key, value in result.items() if key not in {"engine_id", "pid"}},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m4-stability.json"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--engine-timeout", type=int, default=300)
    args = parser.parse_args()
    if not __debug__ or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("M4 requires unoptimized Python in an existing allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for name, value in config["environment"].items():
        os.environ[name] = value
    os.environ.update(
        M3_CAMPAIGN_ID=config["task_id"],
        M3_CAMPAIGN_SHA256=__import__("hashlib").sha256(args.config.read_bytes()).hexdigest(),
    )
    os.environ.pop("PYTHONOPTIMIZE", None)
    root = Path(__file__).resolve().parents[1]
    stop = threading.Event()
    samples = []
    results = []
    status = 1
    sources = {}
    remaining = []
    active_process = None
    ownership = args.output / "engine-ownership"
    ownership.mkdir()

    def interrupted(signum, _frame):
        raise RuntimeError(f"M4 launcher interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def sample():
        with (args.output / "gpu-samples.jsonl").open("w") as stream:
            while not stop.is_set():
                try:
                    row = gpu_snapshot()
                except Exception as error:
                    row = {"error": str(error)}
                samples.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stop.wait(0.5)

    before = gpu_snapshot()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        sources = collect_sources(root, args.model, config)
        (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
        for mode in ["guarded"] if args.calibration else ["native", "off", "prefix", "guarded"]:
            phase = "calibration" if args.calibration else "formal"
            engine_id = "m4-" + mode
            command = [
                sys.executable,
                "-u",
                str(root / "scripts/m4_engine_fixture.py"),
                "--config",
                str(args.config),
                "--model",
                str(args.model),
                "--inputs",
                str(args.inputs),
                "--sources",
                str(args.output / "sources.json"),
                "--output",
                str(args.output / engine_id),
                "--mode",
                mode,
            ]
            if args.calibration:
                command.append("--calibration")
            owner_path = ownership / f"{engine_id}.owner.json"
            ack_path = ownership / f"{engine_id}.ack.json"
            bootstrap = [
                sys.executable,
                str(root / "scripts/m4_processes.py"),
                "--owner",
                str(owner_path),
                "--ack",
                str(ack_path),
                "--engine-id",
                engine_id,
                "--",
                *command,
            ]
            result = {
                "engine_id": engine_id,
                "command": command,
                "exitcode": None,
                "timed_out": False,
                "owned_pids_remaining": [],
                "status": "launching",
            }
            results.append(result)
            atomic_write(args.output / "engine-attempts.json", results)
            with (args.output / f"{engine_id}.log").open("w") as stream:
                proc = subprocess.Popen(
                    bootstrap, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                active_process = proc
                result.update(pid=proc.pid, pgid=proc.pid, status="ownership_pending")
                atomic_write(args.output / "engine-attempts.json", results)
                deadline = time.monotonic() + 8
                while (
                    not owner_path.exists() and proc.poll() is None and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                if not owner_path.exists():
                    raise RuntimeError("M4 child ownership handshake failed")
                owner = json.loads(owner_path.read_text())
                if owner.get("pgid") != proc.pid:
                    raise RuntimeError("M4 child ownership identity mismatch")
                atomic_write(ack_path, {"pgid": proc.pid})
                result["status"] = "running"
                atomic_write(args.output / "engine-attempts.json", results)
                timed_out = False
                try:
                    code = proc.wait(timeout=args.engine_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        code = proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        code = proc.wait(timeout=5)
                residual = owned_processes(proc.pid)
                if residual:
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(1)
                    residual_after_term = owned_processes(proc.pid)
                    if residual_after_term:
                        os.killpg(proc.pid, signal.SIGKILL)
                        time.sleep(0.2)
                remaining = owned_processes(proc.pid)
            active_process = None
            result.update(
                exitcode=code,
                timed_out=timed_out,
                status="exited",
                owned_pids_before_cleanup=residual,
                owned_pids_remaining=remaining,
                gpu_after_process_exit=gpu_snapshot(),
            )
            atomic_write(args.output / "engine-attempts.json", results)
            os.environ.update(
                M3_TRACE_DIR=str(args.output / "launcher"),
                M3_ENGINE_ID=engine_id,
                M3_MODE=mode,
                M3_PHASE="diagnostic",
                M4_PHASE=phase,
            )
            emit_process_exit(config["task_id"], phase, result)
            if code != 0 or timed_out or residual or remaining:
                raise RuntimeError(f"M4 engine failed: {engine_id}")
        status = 0
    finally:
        # Repeated signals must not interrupt failure evidence sealing.
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.signal(signal.SIGINT, lambda *_: None)
        # Child engines own separate groups; clean only our recorded group on any exit.
        try:
            if active_process is not None:
                try:
                    os.killpg(active_process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                cleanup_deadline = time.monotonic() + 15
                while owned_processes(active_process.pid) and time.monotonic() < cleanup_deadline:
                    active_process.poll()
                    time.sleep(0.1)
                if owned_processes(active_process.pid):
                    try:
                        os.killpg(active_process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                active_process.wait(timeout=5)
                remaining = owned_processes(active_process.pid)
                if results:
                    results[-1].update(
                        exitcode=active_process.returncode,
                        status="interrupted",
                        owned_pids_remaining=remaining,
                    )
            independent_cleanup = cleanup_engines(ownership, timeout=1)
            remaining.extend(item for row in independent_cleanup for item in row["remaining"])
            atomic_write(args.output / "engine-group-cleanup.json", independent_cleanup)
            atomic_write(args.output / "engine-attempts.json", results)
        except Exception as error:
            remaining = [f"owned process cleanup could not be verified: {error}"]
        stop.set()
        sampler.join(timeout=12)
        after = gpu_snapshot()
        (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
        (args.output / "results.json").write_text(
            json.dumps({"status": status, "engines": results}, indent=2) + "\n"
        )
        resources = {
            "job_id": os.environ["SLURM_JOB_ID"],
            "gpu_before": before,
            "gpu_during": samples,
            "gpu_after": after,
            "owned_pids_remaining": remaining,
            "engines": results,
            "sampling_interval_seconds": 0.5,
            "memory_definition": "nvidia-smi device used MiB; no performance inference",
        }
        (args.output / "resource-checkpoints.json").write_text(
            json.dumps(resources, indent=2) + "\n"
        )
        # Preserve each per-process original and also make one deterministic audit file.
        with (args.output / "trace.jsonl").open("wb") as target:
            for path in sorted(args.output.glob("*/*.jsonl")):
                target.write(path.read_bytes())
        with tarfile.open(args.output / "raw-artifacts.tar.gz", "w:gz") as archive:
            for path in sorted(args.output.rglob("*")):
                if path.is_file() and path.name != "raw-artifacts.tar.gz":
                    archive.add(path, arcname=str(path.relative_to(args.output)))
    return status


if __name__ == "__main__":
    sys.exit(main())
