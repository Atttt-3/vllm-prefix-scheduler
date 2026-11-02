"""Run one source-bound replacement campaign in an existing owned allocation."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_m3_suite import gpu_snapshot, owned_processes

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--history-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__ or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Unoptimized execution in an existing allocation required")
    for name in ("NATIVE_PYTHON", "MODEL_CONFIG_DIR", "M3_CACHE"):
        if not os.environ.get(name):
            raise RuntimeError(f"Existing private environment required: {name}")
    os.chdir(ROOT)
    if subprocess.check_output(["git", "status", "--porcelain=v1"]):
        raise RuntimeError("Campaign requires clean frozen checkout")
    plan = json.loads(args.campaign_config.read_text())
    for field in ("runtime_config", "authorization"):
        item = plan[field]
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen {field} hash differs")
    if plan["runtime_config"]["path"] != "configs/m3-real-model.json":
        raise RuntimeError("Legacy inference entry requires the original profile")
    if digest(args.history_index) != plan["history"]["index_sha256"]:
        raise RuntimeError("Frozen historical index differs")
    args.output.mkdir(parents=True, exist_ok=False)
    campaign_sha = digest(args.campaign_config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    binding = {
        "campaign_id": plan["campaign_id"],
        "campaign_sha256": campaign_sha,
        "project_commit": commit,
        "job_id": os.environ["SLURM_JOB_ID"],
        "runtime_config_sha256": plan["runtime_config"]["sha256"],
        "history_index_sha256": plan["history"]["index_sha256"],
    }
    write(args.output / "campaign-binding.json", binding)
    os.environ.update(
        M3_CAMPAIGN_ID=plan["campaign_id"],
        M3_CAMPAIGN_SHA256=campaign_sha,
        M3_RUN=str(args.output / "gpu"),
        M3_REGRESSION_RUN=str(args.output / "native"),
    )
    active = None
    steps = []
    codes = {"gpu": 125, "native": 125, "gpu_after_query": 125}
    before = gpu_snapshot()

    def stop():
        nonlocal active
        if active is None:
            return
        if active.poll() is None or owned_processes(active.pid):
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                active.poll()
                if not owned_processes(active.pid):
                    break
                time.sleep(0.1)
            if owned_processes(active.pid):
                try:
                    os.killpg(active.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            active.wait(timeout=10)

    def interrupted(signum, _frame):
        raise RuntimeError(f"campaign interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def step(name, command, timeout):
        nonlocal active
        with (args.output / f"{name}-launcher.log").open("xb") as log:
            active = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record = {"name": name, "command": command, "pgid": active.pid}
            steps.append(record)
            try:
                code = active.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                stop()
                code = 124
            unexpected = owned_processes(active.pid)
            if unexpected:
                record["unexpected_owned_processes"] = unexpected
                stop()
                code = 126
            record.update(returncode=code, remaining=owned_processes(active.pid))
            active = None
            return code

    try:
        codes["gpu"] = step(
            "gpu",
            [
                "bash",
                "scripts/run_m3.sh",
                "--engines",
                "smoke-prefix",
                "diagnostic-guarded",
                "smoke-native",
                "smoke-off",
                "smoke-guarded",
                "diagnostic-native",
                "diagnostic-off",
                "diagnostic-prefix",
                "--repeats",
                "3",
                "--engine-timeout",
                "240",
            ],
            600,
        )
        if codes["gpu"] == 0:
            codes["native"] = step("native", ["bash", "scripts/run_m3_regressions.sh"], 480)
    finally:
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.signal(signal.SIGINT, lambda *_: None)
        errors = []
        try:
            stop()
        except Exception as error:
            errors.append(f"owned process cleanup failed: {error}")
        after = gpu_snapshot()
        codes["gpu_after_query"] = after.get("returncode")
        raw_path = args.output / "gpu-data/resource-checkpoints.json"
        resources = {"engines": [], "gpu_during": [before]}
        if raw_path.exists():
            try:
                checkpoint = json.loads(raw_path.read_text())
                if (
                    type(checkpoint) is not dict
                    or type(checkpoint.get("owned_pids_remaining", [])) is not list
                ):
                    raise ValueError("GPU checkpoint object/process list is malformed")
                resources = checkpoint
            except Exception as error:
                errors.append(f"GPU checkpoint could not be read: {error}")
        for record in steps:
            try:
                record["remaining"] = owned_processes(record["pgid"])
            except Exception as error:
                record["remaining"] = [f"owned group query failed: {error}"]
        if errors:
            codes["resource_cleanup"] = 127
        remaining = [item for record in steps for item in record.get("remaining", [])]
        resources.update(
            job_id=binding["job_id"],
            gpu_before=before,
            gpu_after=after,
            owned_pids_remaining=resources.get("owned_pids_remaining", []) + remaining,
            campaign_binding=binding,
            launcher_steps=steps,
            cleanup_errors=errors,
        )
        write(args.output / "allocation-resource-checkpoints.json", resources)
        write(
            args.output / "campaign-exit.json",
            {**binding, **codes, "owned_pids_remaining": resources["owned_pids_remaining"]},
        )
    return (
        0
        if all(value == 0 for value in codes.values()) and not resources["owned_pids_remaining"]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
