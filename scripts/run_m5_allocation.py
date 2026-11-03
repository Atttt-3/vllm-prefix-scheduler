"""Run one approved M5 stage or pair inside an existing allocation."""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from m3_sources import collect_sources
from m4_processes import atomic_write
from run_m3_suite import owned_processes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_prefix_scheduler.m5_campaign import (  # noqa: E402
    D2_ROLE,
    D2R_PREVIOUS_ANALYSIS,
    D2R_ROLE,
    DIAGNOSTIC_ROLES,
    TASK,
    bind_d2r_inputs,
    check_authorization,
    digest,
    formal_specs,
    id_ok,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA, d1_specs  # noqa: E402
from vllm_prefix_scheduler.m5_d2 import d2_specs, d2r_specs  # noqa: E402
from vllm_prefix_scheduler.m5_resources import RESOURCE_SCHEMA_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def query_gpu(option):
    try:
        proc = subprocess.run(
            ["nvidia-smi", option, "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def process_groups():
    try:
        data = subprocess.check_output(["ps", "-eo", "pid=,pgid="], text=True, timeout=2)
        return {int(pid): int(pgid) for pid, pgid in (line.split() for line in data.splitlines())}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def application_groups(apps, before):
    processes = []
    if apps["returncode"] == 0:
        for line in apps["stdout"].splitlines():
            _, pid, _ = (part.strip() for part in line.split(","))
            pid = int(pid)
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                pgid = before.get(pid)
            except OSError:
                pgid = None
            processes.append({"pid": pid, "pgid": pgid})
    return processes


def snapshot(d2=False, ownership=False):
    result = {"monotonic_ns": time.monotonic_ns()}
    base = "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu"
    thermal_query = None
    if d2:
        thermal_query = query_gpu(base + ",temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw")
        stamp = time.monotonic_ns()
        fields = thermal_query["stdout"].strip().split(",")
        if thermal_query["returncode"] == 0 and len(fields) == 10:
            result["devices"] = {**thermal_query, "stdout": ",".join(fields[:5]) + "\n"}
        result["thermal_query"] = thermal_query
    if "devices" not in result:
        result["devices"] = query_gpu(base)
    groups_started = time.monotonic_ns()
    before = process_groups() if ownership else {}
    apps = query_gpu("--query-compute-apps=gpu_uuid,pid,used_memory")
    result["applications"] = apps
    try:
        processes = application_groups(apps, before) if ownership or d2 else []
        if ownership:
            result["application_process_groups"] = processes
            result["application_sample_started_ns"] = groups_started
            result["application_sample_ended_ns"] = time.monotonic_ns()
    except ValueError as error:
        result["application_groups_error"] = str(error)
        processes = []
    if d2 and thermal_query["returncode"] == 0 and len(fields) == 10 and apps["returncode"] == 0:
        try:
            gpu, _, _, memory, utilization, temperature, sm, mem, pstate, power = (
                value.strip() for value in fields
            )
            for line in apps["stdout"].splitlines():
                uuid, _, _ = (part.strip() for part in line.split(","))
                if uuid != gpu:
                    raise ValueError("foreign GPU application record")
            thermal = {
                "monotonic_ns": stamp,
                "gpu_uuid": gpu,
                "temperature_c": float(temperature),
                "sm_clock_mhz": float(sm),
                "memory_clock_mhz": float(mem),
                "gpu_utilization_percent": float(utilization),
                "power_w": None if power in {"N/A", "[N/A]", "[Not Supported]"} else float(power),
                "p_state": pstate,
                "memory_used_mib": float(memory),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "gpu_apps": processes,
            }
            if any(type(value) is float and not math.isfinite(value) for value in thermal.values()):
                raise ValueError("nonfinite GPU telemetry")
            result["thermal"] = thermal
        except ValueError as error:
            result["thermal_parse_error"] = str(error)
    return result


def d1_device_boundary():
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    return {
        "monotonic_ns": time.monotonic_ns(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu": {"returncode": query.returncode, "stdout": query.stdout, "stderr": query.stderr},
    }


def owner_group(path, job, collector_pid):
    owner = read_json(path)
    pgid = owner["pgid"]
    require(
        owner["task_id"] == TASK
        and owner["job_id"] == job
        and owner["collector_pid"] == collector_pid
        and owner["uid"] == os.getuid()
        and type(pgid) is int
        and pgid > 1
        and pgid != os.getpgrp()
        and owner["pid"] == pgid,
        "unexpected M5 child owner",
    )
    return pgid


def stop_group(pgid, process=None):
    if pgid is None:
        return []
    for sig, seconds in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        if process is not None:
            process.poll()
        if not owned_processes(pgid):
            return []
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process is not None:
                process.poll()
            if not owned_processes(pgid):
                return []
            time.sleep(0.05)
    return owned_processes(pgid)


def input_command(python, config, model, table, spec):
    cmd = [
        python,
        "scripts/prepare_m5_inputs.py",
        "--config",
        str(config),
        "--model",
        str(model),
        "--output",
        str(table),
        "--scenario",
        spec["scenario"],
        "--seed",
        str(spec["seed"]),
        "--count",
        str(spec["count"]),
    ]
    if spec["rate_rps"] is not None:
        cmd += ["--rate", str(spec["rate_rps"])]
    if spec["interval_ms"] is not None:
        cmd += ["--interval-ms", str(spec["interval_ms"])]
    if spec.get("mechanism"):
        cmd += ["--mechanism", spec["mechanism"]]
    return cmd


def auxiliary_command(command, timeout):
    if timeout < 1:
        raise TimeoutError("M5 allocation work deadline")
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("authorization", type=Path)
    p.add_argument("phase", choices=("calibration", "formal"))
    p.add_argument("role")
    p.add_argument("attempt")
    p.add_argument("output", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument(
        "auxiliary", help="calibration dependencies, D1 proposal, or formal freeze; '-' if unused"
    )
    args = p.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        sys.platform == "linux" and not sys.flags.optimize and job.isdigit(),
        "unoptimized Linux interpreter in an owned Slurm allocation required",
    )
    require(id_ok(args.attempt), "invalid attempt ID")
    args.output = args.output.resolve()
    require(args.output.is_relative_to(ROOT / "runs" / TASK), "use the M5 private runs directory")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT),
        "dirty GPU checkout",
    )
    config = read_json(args.config)
    require(
        config["resource_schema_version"] == RESOURCE_SCHEMA_VERSION,
        "configuration M5 resource schema differs",
    )
    auxiliary = None if args.auxiliary == "-" else Path(args.auxiliary)
    freeze_path = auxiliary if args.phase == "formal" else None
    authorization = read_json(args.authorization)
    roles = check_authorization(
        authorization, args.config, commit, args.phase, freeze_path=freeze_path
    )
    require(args.role in roles, "M5 role outside approved budget")
    freeze = read_json(freeze_path) if freeze_path else None
    diagnostic = args.role in DIAGNOSTIC_ROLES
    d2r = args.role == D2R_ROLE
    d2 = args.role in (D2_ROLE, D2R_ROLE)
    retained_analysis = None
    retained_d2_sha = None
    rates = None
    dress_reports = []
    if auxiliary and freeze is None and not diagnostic:
        dependencies = read_json(auxiliary)
        require(
            dependencies["config_sha256"] == digest(args.config),
            "calibration dependencies detached",
        )
        for item in dependencies["reports"]:
            path = Path(item["path"])
            require(digest(path) == item["sha256"], "calibration dependency hash differs")
            report = read_json(path)
            require(
                report["status"] == "qualified"
                and report["config_sha256"] == digest(args.config)
                and report["stage"] == item["stage"],
                "calibration dependency not qualified",
            )
            if item["stage"] == "C2":
                rates = report["selected_rates"]
            else:
                dress_reports.append(report)
    if diagnostic:
        require(auxiliary is not None, "diagnostic requires its reviewed plan and archived input")
        diagnostic_table = auxiliary.parent / "input.json"
        if d2r:
            retained_analysis = auxiliary.parent / D2R_PREVIOUS_ANALYSIS
            retained_d2_sha = bind_d2r_inputs(authorization, auxiliary, retained_analysis)
        plan_specs = {D2R_ROLE: d2r_specs, D2_ROLE: d2_specs}.get(args.role, d1_specs)
        specs = plan_specs(
            config, auxiliary, diagnostic_table, auxiliary.parent / "original-c3-result.json"
        )
    elif freeze:
        specs = formal_specs(config, freeze, args.role)
    else:
        from plan_m5_calibration import plan_stage

        specs = plan_stage(config, args.role, rates=rates)
        if isinstance(specs, dict):
            specs = specs["runs"]
    require(specs and all(id_ok(row["id"]) for row in specs), "invalid M5 run plan")
    args.output.mkdir(parents=True, exist_ok=False)
    cache_tmp = args.output / "tmp"
    cache_tmp.mkdir()
    os.environ.update(config["environment"])
    os.environ.update(
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        HF_HOME=str(args.cache / "huggingface"),
        XDG_CACHE_HOME=str(args.cache),
        TMPDIR=str(cache_tmp),
    )
    start = time.monotonic_ns()
    deadline = start + config["gates"]["max_allocation_work_s"] * 10**9

    def remaining():
        return (deadline - time.monotonic_ns()) / 10**9

    binding = {
        "resource_schema_version": RESOURCE_SCHEMA_VERSION,
        "task_id": TASK,
        "job_id": job,
        "attempt": args.attempt,
        "role": args.role,
        "phase": args.phase,
        "execution_commit": commit,
        "config_sha256": digest(args.config),
        "authorization_sha256": digest(args.authorization),
        "auxiliary_sha256": digest(auxiliary) if auxiliary else None,
        "started_monotonic_ns": start,
    }
    atomic_write(args.output / "binding.json", binding)
    python = sys.executable
    sources = args.output / "allocation-sources.json"
    results = []
    active = None
    owner_path = None
    errors = []
    table_hashes = {}
    rpc_directory = None
    code = 1

    def interrupt(signum, _frame):
        raise InterruptedError(f"M5 allocation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    stop_sampling = threading.Event()
    before = snapshot()

    d2_samples = args.output / "d2-device.jsonl"
    gpu_samples = args.output / "gpu-during.jsonl"

    def sample():
        with ExitStack() as stack:
            stream = stack.enter_context(gpu_samples.open("x"))
            thermal_stream = stack.enter_context(d2_samples.open("x")) if d2 else None
            next_sample = time.monotonic()
            while not stop_sampling.is_set():
                observed = snapshot(d2=d2, ownership=True)
                stream.write(json.dumps(observed) + "\n")
                stream.flush()
                if thermal_stream is not None and "thermal" in observed:
                    thermal_stream.write(json.dumps(observed["thermal"], allow_nan=False) + "\n")
                    thermal_stream.flush()
                if d2:
                    next_sample = max(next_sample + 1, time.monotonic())
                    stop_sampling.wait(max(0, next_sample - time.monotonic()))
                else:
                    stop_sampling.wait(2)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    def run_spec(spec):
        nonlocal active, owner_path
        leaf = args.output / spec["id"]
        leaf.mkdir()
        if diagnostic:
            table = leaf / "input.json"
            shutil.copyfile(diagnostic_table, table)
            require(digest(table) == D1_INPUT_SHA, "diagnostic input copy differs")
            if not d2:
                atomic_write(leaf / "diagnostic-device-before.json", d1_device_boundary())
        elif freeze:
            table = freeze_path.parent / "tables" / f"{args.role}.json"
            require(digest(table) == freeze["tables"][args.role], "formal input hash differs")
        else:
            table = leaf / "input.json"
            auxiliary_command(
                input_command(python, args.config, args.model, table, spec), remaining()
            )
        raw = leaf / "raw"
        collector = leaf / "collector"
        audit = leaf / "audit.json"
        owner_path, ack = leaf / "owner.json", leaf / "ack.json"
        command = [
            python,
            "-u",
            "scripts/replay_m5.py",
            "--config",
            str(args.config),
            "--input",
            str(table),
            "--model",
            str(args.model),
            "--output",
            str(raw),
            "--mode",
            spec["mode"],
            "--tier",
            spec["tier"],
            "--phase",
            "diagnostic" if args.role == "C1" else args.phase,
            "--pair-id",
            args.role if freeze else spec["id"],
            "--sources",
            str(sources),
        ]
        if spec["mechanism"]:
            command += ["--mechanism", spec["mechanism"]]
        if d2:
            pilot = spec["id"] == "thermal-pilot"
            command += [
                "--d2-role",
                "pilot" if pilot else "measure",
                "--d2-plan",
                str(auxiliary),
                "--d2-samples",
                str(d2_samples),
            ]
            if not pilot:
                target = args.output / "thermal-pilot/raw/d2-thermal.json"
                command += ["--d2-target", str(target)]
        timeout = math.floor(remaining()) - 15
        require(timeout > 0, "M5 allocation work deadline")
        collect = [
            python,
            "scripts/run_evidence.py",
            "--task-id",
            TASK,
            "--kind",
            "performance" if args.phase == "formal" else "gpu",
            "--output",
            str(collector),
            "--timeout",
            str(timeout),
            "--seed",
            str(spec["seed"]),
            "--config",
            str(args.config),
            "--config",
            str(table),
            "--criterion",
            "independent M5 raw request and batch audit required",
            "--artifact",
            str(raw / "client.json"),
            "--artifact",
            str(raw / "sources.json"),
        ]
        collect += ["--artifact", str(raw / "engine.json")]
        if d2:
            collect += ["--config", str(auxiliary), "--artifact", str(raw / "d2-thermal.json")]
            if d2r:
                collect += ["--config", str(retained_analysis)]
            if not pilot:
                collect += ["--config", str(target)]
        collect += ["--", python, "scripts/m5_owned.py", str(owner_path), str(ack), *command]
        result = {
            **spec,
            "pair_id": args.role,
            "inputs": str(table),
            "run": str(raw),
            "collector": str(collector),
            "audit": str(audit),
            "started_monotonic_ns": time.monotonic_ns(),
            "returncode": None,
            "remaining_processes": None,
        }
        if diagnostic and not d2:
            result["diagnostic_device_before_sha256"] = digest(
                leaf / "diagnostic-device-before.json"
            )
        results.append(result)
        with (leaf / "launcher.log").open("xb") as log:
            active = subprocess.Popen(
                collect, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            result["collector_pid"] = active.pid
            atomic_write(
                args.output / "index.json",
                {"schema_version": 1, "stage": args.role, "runs": results},
            )
            owner_deadline = time.monotonic() + 10
            while (
                not owner_path.exists()
                and active.poll() is None
                and time.monotonic() < owner_deadline
            ):
                time.sleep(0.01)
            require(owner_path.exists(), "collector child ownership registration missing")
            pgid = owner_group(owner_path, job, active.pid)
            result["engine_pgid"] = pgid
            result["engine_owner_ack_ns"] = time.monotonic_ns()
            atomic_write(
                args.output / "index.json",
                {"schema_version": 1, "stage": args.role, "runs": results},
            )
            atomic_write(ack, {"job_id": job, "pgid": pgid})
            result["returncode"] = active.wait(timeout=max(1, remaining()))
            result["remaining_processes"] = owned_processes(pgid) + owned_processes(active.pid)
            result["ended_monotonic_ns"] = time.monotonic_ns()
            result["elapsed_seconds"] = (
                result["ended_monotonic_ns"] - result["started_monotonic_ns"]
            ) / 10**9
            atomic_write(
                args.output / "index.json",
                {"schema_version": 1, "stage": args.role, "runs": results},
            )
            require(result["remaining_processes"] == [], "M5 process group remained after replay")
            active = None
            owner_path = None
            require(result["returncode"] == 0, "M5 replay or collector failed")
        if d2:
            thermal = raw / "d2-thermal.json"
            result["d2_thermal_sha256"] = digest(thermal)
            thermal_status = read_json(thermal)["status"]
            expected = "target-frozen" if pilot else "admitted"
            require(thermal_status in {expected, "invalid"}, "D2 thermal replay status differs")
            result["d2_thermal_status"] = thermal_status
            if pilot or thermal_status == "invalid":
                shutil.copyfile(thermal, audit)
                result["audit_sha256"] = digest(audit)
                atomic_write(
                    args.output / "index.json",
                    {"schema_version": 1, "stage": args.role, "runs": results},
                )
                return thermal_status == expected
        check = [
            python,
            "scripts/check_m5_trace.py",
            "run",
            "--config",
            str(args.config),
            "--inputs",
            str(table),
            "--run",
            str(raw),
            "--output",
            str(audit),
            "--mode",
            spec["mode"],
        ]
        check += [
            "--pair",
            args.role if freeze else spec["id"],
            "--phase",
            "diagnostic" if args.role == "C1" else args.phase,
        ]
        auxiliary_command(check, remaining())
        result["audit_sha256"] = digest(audit)
        if diagnostic and not d2:
            after_path = leaf / "diagnostic-device-after.json"
            atomic_write(after_path, d1_device_boundary())
            result["diagnostic_device_after_sha256"] = digest(after_path)
        atomic_write(
            args.output / "index.json", {"schema_version": 1, "stage": args.role, "runs": results}
        )
        return True

    try:
        # Pinned vLLM appends /<UUID> to VLLM_RPC_BASE_PATH for UNIX sockets.
        # Keep engine IPC on the node and keep run artifacts in their original directory.
        rpc_directory = Path(tempfile.mkdtemp(prefix=f"m5-{job}-", dir="/tmp"))
        require(len(os.fsencode(rpc_directory)) + 37 <= 107, "M5 engine IPC path is too long")
        os.environ["VLLM_RPC_BASE_PATH"] = str(rpc_directory)
        source_record = collect_sources(ROOT, args.model, config)
        atomic_write(sources, {**source_record, "job_id": job, "task_id": TASK})
        for spec in specs:
            if not run_spec(spec):
                break
        if d2:
            count = sum(row.get("d2_thermal_status") == "admitted" for row in results)
            atomic_write(
                args.output / "d2-result.json",
                {
                    "status": "diagnostic-complete" if count == 4 else "diagnostic-incomplete",
                    "proposal_sha256": digest(auxiliary),
                    "input_sha256": D1_INPUT_SHA,
                    "audited_runs": count,
                    "retained_c3_sha256": C3_BLOCKED_SHA,
                    **(
                        {"role": D2R_ROLE, "retained_d2_analysis_sha256": retained_d2_sha}
                        if d2r
                        else {}
                    ),
                },
            )
        elif diagnostic:
            atomic_write(
                args.output / "d1-result.json",
                {
                    "status": "diagnostic-complete",
                    "proposal_sha256": digest(auxiliary),
                    "input_sha256": D1_INPUT_SHA,
                    "audited_runs": len(results),
                    "retained_c3_sha256": C3_BLOCKED_SHA,
                },
            )
        elif args.phase == "calibration":
            from plan_m5_calibration import evaluate_stage, serial_interval_from_pilot

            if args.role == "C6":
                pilot = next(row for row in results if row["id"] == "serial-pilot")
                interval = serial_interval_from_pilot(read_json(pilot["audit"]))
                followup = plan_stage(config, "C6", rates=rates, serial_interval_ms=interval)
                for spec in followup["runs"] if isinstance(followup, dict) else followup:
                    run_spec(spec)
            qualification = evaluate_stage(
                config,
                args.role,
                args.output / "index.json",
                rates,
                dress_reports,
                config_sha256=digest(args.config),
            )
            qualification["config_sha256"] = digest(args.config)
            atomic_write(args.output / "calibration-result.json", qualification)
            require(qualification["status"] == "qualified", "M5 calibration failed its gates")
            if args.role == "C6":
                table_specs = []
                formal_rates = {**rates, **qualification["selected_rates"]}
                for pair in config["matrix"]:
                    scenario = pair["scenario"]
                    table_specs.append(
                        {
                            "id": pair["id"],
                            "scenario": scenario,
                            "seed": pair["seed"],
                            "count": config["request_count"],
                            "rate_rps": formal_rates.get(scenario),
                            "interval_ms": interval if scenario == "serial" else None,
                        }
                    )
                batch = args.output / "formal-table-specs.json"
                atomic_write(batch, table_specs)
                auxiliary_command(
                    [
                        python,
                        "scripts/prepare_m5_inputs.py",
                        "--config",
                        str(args.config),
                        "--model",
                        str(args.model),
                        "--batch",
                        str(batch),
                        "--output",
                        str(args.output / "tables"),
                    ],
                    remaining(),
                )
                table_hashes = {
                    spec["id"]: digest(args.output / "tables" / f"{spec['id']}.json")
                    for spec in table_specs
                }
        require(remaining() >= 0, "M5 allocation work deadline")
        code = 0
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        residue = []
        if active is not None:
            try:
                if owner_path and owner_path.exists():
                    residue.extend(stop_group(owner_group(owner_path, job, active.pid)))
            except Exception as error:
                errors.append(f"engine cleanup: {error}")
            try:
                residue.extend(stop_group(active.pid, active))
            except Exception as error:
                errors.append(f"collector cleanup: {error}")
        if rpc_directory is not None:
            try:
                shutil.rmtree(rpc_directory)
            except OSError as error:
                errors.append(f"engine IPC cleanup: {error}")
        stop_sampling.set()
        sampler.join(timeout=12)
        after = snapshot()
        atomic_write(
            args.output / "allocation.json",
            {
                **binding,
                "ended_monotonic_ns": time.monotonic_ns(),
                "work_returncode": code,
                "formal_table_sha256": table_hashes,
                "node": os.uname().nodename,
                "errors": errors,
                "remaining_processes": residue,
                "gpu_before": before,
                "gpu_after": after,
                "sampler_exited": not sampler.is_alive(),
                "gpu_during_sha256": digest(gpu_samples) if gpu_samples.exists() else None,
                **(
                    {"d2_device_sha256": digest(d2_samples) if d2_samples.exists() else None}
                    if d2
                    else {}
                ),
                "rpc_directory_removed": rpc_directory is None or not rpc_directory.exists(),
                "index_sha256": digest(args.output / "index.json")
                if (args.output / "index.json").exists()
                else None,
            },
        )
        require(
            not residue and not errors and not sampler.is_alive(),
            "M5 allocation failed or cleanup incomplete",
        )


if __name__ == "__main__":
    main()
