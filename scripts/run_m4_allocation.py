"""Bounded M4 execution inside one existing allocation; preserve every failed step."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from m4_processes import cleanup_engines
from run_m3_suite import gpu_snapshot, owned_processes

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration", action="store_true")
    args = p.parse_args()
    if not __debug__ or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Unoptimized execution in an existing allocation required")
    for name in ("NATIVE_PYTHON", "MODEL_CONFIG_DIR", "M4_CACHE"):
        if not os.environ.get(name):
            raise RuntimeError(f"Existing private environment required: {name}")
    os.chdir(ROOT)
    if subprocess.check_output(["git", "status", "--porcelain=v1"]):
        raise RuntimeError("M4 requires clean frozen checkout")
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    python = os.environ["NATIVE_PYTHON"]
    model = os.environ["MODEL_CONFIG_DIR"]
    cache = os.environ["M4_CACHE"]
    os.environ.update(config["environment"])
    os.environ.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}:{ROOT / 'tests/native'}",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        HF_HOME=f"{cache}/huggingface",
        XDG_CACHE_HOME=cache,
        TMPDIR=str(ROOT / ".tmp"),
    )
    (ROOT / ".tmp").mkdir(exist_ok=True)
    for name in ("M3_TRACE_DIR", "M3_CAMPAIGN_ID", "M3_CAMPAIGN_SHA256"):
        os.environ.pop(name, None)
    binding = {
        "task_id": config["task_id"],
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "phase": "calibration" if args.calibration else "formal",
        "started_unix_ns": time.time_ns(),
        "started_monotonic_ns": time.monotonic_ns(),
    }
    write(args.output / "binding.json", binding)
    started = time.monotonic()
    before = gpu_snapshot()
    active = None
    steps = []

    def stop():
        if active is None:
            return
        if active.poll() is None or owned_processes(active.pid):
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                active.poll()
                if not owned_processes(active.pid):
                    break
                time.sleep(0.1)
            if owned_processes(active.pid):
                os.killpg(active.pid, signal.SIGKILL)
            active.wait(timeout=5)

    def interrupted(signum, _frame):
        raise RuntimeError(f"M4 interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def step(name, command, timeout):
        nonlocal active
        remaining = config["timing"]["allocation_work_timeout_s"] - (time.monotonic() - started)
        record = {
            "name": name,
            "command": command,
            "started_monotonic_ns": time.monotonic_ns(),
            "returncode": None,
            "remaining": [],
            "status": "launching",
        }
        steps.append(record)
        record["gpu_checkpoint"] = gpu_snapshot()
        if remaining < 30:
            record.update(returncode=125, not_started="allocation deadline", remaining=[])
            return 125
        with (args.output / f"{name}-launcher.log").open("xb") as log:
            active = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["pgid"] = active.pid
            record["status"] = "running"
            try:
                code = active.wait(timeout=min(timeout, remaining))
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                stop()
                code = 124
            unexpected = owned_processes(active.pid)
            if unexpected:
                record["unexpected_owned_processes"] = unexpected
                stop()
                code = 126
            record.update(
                returncode=code,
                status="exited",
                remaining=owned_processes(active.pid),
                ended_monotonic_ns=time.monotonic_ns(),
            )
            active = None
            if name == "gpu":
                cleanup = cleanup_engines(args.output / "gpu-data/engine-ownership", timeout=15)
                write(args.output / "post-gpu-independent-cleanup.json", cleanup)
                if any(row["before"] or row["remaining"] for row in cleanup):
                    record["returncode"] = code = 126
            return code

    def collect(name, kind, command, timeout, artifacts):
        data = args.output / f"{name}-data"
        argv = [
            python,
            "scripts/run_evidence.py",
            "--task-id",
            f"M4-{name}",
            "--kind",
            kind,
            "--output",
            str(args.output / name),
            "--timeout",
            str(timeout),
            "--seed",
            "42",
            "--config",
            str(args.config),
            "--config",
            "configs/upstream.json",
        ]
        for artifact in artifacts:
            argv += ["--artifact", str(data / artifact)]
        if name == "gpu":
            argv += ["--config", str(args.output / "gpu-inputs.json")]
        argv += [
            "--criterion",
            f"M4 {name}: independently reconstruct complete raw matrix; command success alone is insufficient",
            "--",
            *command,
        ]
        return step(name, argv, timeout + 15)

    try:
        # The first allocation validates the native fixture before spending model time.
        collect(
            "native",
            "native",
            [
                python,
                "scripts/replay_m4.py",
                "--config",
                str(args.config),
                "--model-config",
                model,
                "--output",
                str(args.output / "native-data"),
            ],
            360,
            ["runtime.json", "trace.jsonl", "summary.json"],
        )
        collect(
            "cpu",
            "native",
            [
                python,
                "scripts/measure_m4_cpu.py",
                "--config",
                str(args.config),
                "--model-config",
                model,
                "--output",
                str(args.output / "cpu-data"),
            ],
            180,
            ["runtime.json", "trace.jsonl", "summary.json"],
        )
        prepared = step(
            "inputs",
            [
                python,
                "scripts/prepare_m4_inputs.py",
                "--config",
                str(args.config),
                "--model",
                model,
                "--output",
                str(args.output / "gpu-inputs.json"),
            ],
            60,
        )
        if prepared == 0:
            command = [
                python,
                "-u",
                "scripts/run_m4_suite.py",
                "--config",
                str(args.config),
                "--model",
                model,
                "--inputs",
                str(args.output / "gpu-inputs.json"),
                "--output",
                str(args.output / "gpu-data"),
            ]
            if args.calibration:
                command.append("--calibration")
            collect(
                "gpu",
                "gpu",
                command,
                600,
                [
                    "sources.json",
                    "trace.jsonl",
                    "results.json",
                    "resource-checkpoints.json",
                    "raw-artifacts.tar.gz",
                ],
            )
        # Reuse unchanged native baseline mechanisms under M4 allocation provenance.
        os.environ["M3_REGRESSION_RUN"] = str(args.output / "regressions")
        step("regressions", ["bash", "scripts/run_m4_regressions.sh"], 360)
    finally:
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.signal(signal.SIGINT, lambda *_: None)
        errors = []
        try:
            stop()
        except Exception as error:
            errors.append(str(error))
        independent_cleanup = []
        try:
            independent_cleanup = cleanup_engines(
                args.output / "gpu-data/engine-ownership", timeout=15
            )
        except Exception as error:
            errors.append(f"independent engine cleanup failed: {error}")
        write(args.output / "independent-engine-cleanup.json", independent_cleanup)
        after = gpu_snapshot()
        for record in steps:
            if "pgid" in record:
                try:
                    record["remaining"] = owned_processes(record["pgid"])
                except Exception as error:
                    record["remaining"] = [f"process query failed: {error}"]
        remaining = [item for record in steps for item in record.get("remaining", [])]
        remaining += [item for row in independent_cleanup for item in row["remaining"]]
        gpu_file = args.output / "gpu-data/resource-checkpoints.json"
        gpu = json.loads(gpu_file.read_text()) if gpu_file.exists() else {}
        resources = {
            **binding,
            "gpu_before": before,
            "gpu_during": gpu.get("gpu_during", [])
            + [record["gpu_checkpoint"] for record in steps],
            "gpu_after": after,
            "owned_pids_remaining": remaining + gpu.get("owned_pids_remaining", []),
            "engines": gpu.get("engines", []),
            "launcher_steps": steps,
            "cleanup_errors": errors,
        }
        write(args.output / "allocation-resource-checkpoints.json", resources)
        ok = len(steps) == 5 and all(record.get("returncode") == 0 for record in steps)
        code = (
            0
            if ok
            and not resources["owned_pids_remaining"]
            and not errors
            and after["returncode"] == 0
            else 1
        )
        write(
            args.output / "exit.json",
            {
                **binding,
                "returncode": code,
                "steps": steps,
                "ended_unix_ns": time.time_ns(),
                "ended_monotonic_ns": time.monotonic_ns(),
                "owned_pids_remaining": resources["owned_pids_remaining"],
            },
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
