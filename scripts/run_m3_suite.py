"""Serial M3 engine launcher with failure-preserving logs and owned-process cleanup."""

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


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    p = subprocess.run(command, text=True, capture_output=True, timeout=10)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m3-real-model.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=[
            "smoke-native",
            "smoke-off",
            "smoke-prefix",
            "smoke-guarded",
            "diagnostic-native",
            "diagnostic-off",
            "diagnostic-prefix",
            "diagnostic-guarded",
        ],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engine-timeout", type=int, default=180)
    args = parser.parse_args()
    if not __debug__ or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("M3 requires unoptimized Python in an existing allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for name, value in config["environment"].items():
        os.environ[name] = value
    os.environ.pop("PYTHONOPTIMIZE", None)
    root = Path(__file__).resolve().parents[1]
    stop = threading.Event()
    samples = []
    results = []
    status = 1
    sources = {}
    remaining = []
    active_process = None

    def interrupted(signum, _frame):
        raise RuntimeError(f"M3 launcher interrupted by signal {signum}")

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
        for engine_id in args.engines:
            phase, mode = engine_id.split("-", 1)
            command = [
                sys.executable,
                "-u",
                str(root / "scripts/run_m3_engine.py"),
                "--config",
                str(args.config),
                "--model",
                str(args.model),
                "--sources",
                str(args.output / "sources.json"),
                "--output",
                str(args.output / engine_id),
                "--mode",
                mode,
                "--phase",
                phase,
                "--repeats",
                str(args.repeats),
            ]
            with (args.output / f"{engine_id}.log").open("w") as stream:
                proc = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                active_process = proc
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
            result = {
                "engine_id": engine_id,
                "command": command,
                "exitcode": code,
                "timed_out": timed_out,
                "owned_pids_before_cleanup": residual,
                "owned_pids_remaining": remaining,
                "gpu_after_process_exit": gpu_snapshot(),
            }
            results.append(result)
            os.environ.update(
                M3_TRACE_DIR=str(args.output / "launcher"),
                M3_ENGINE_ID=engine_id,
                M3_MODE=mode,
                M3_PHASE=phase,
            )
            from vllm_prefix_scheduler.m3_events import emit

            emit("process_exit", **{k: v for k, v in result.items() if k != "engine_id"})
            if code != 0 or timed_out or residual or remaining:
                raise RuntimeError(f"M3 engine failed: {engine_id}")
        status = 0
    finally:
        # Child engines own separate groups; clean only our recorded group on any exit.
        if active_process is not None:
            try:
                os.killpg(active_process.pid, signal.SIGTERM)
                active_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(active_process.pid, signal.SIGKILL)
                active_process.wait(timeout=5)
            except ProcessLookupError:
                pass
            remaining = owned_processes(active_process.pid)
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
