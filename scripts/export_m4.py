"""Re-audit complete M4 raw evidence and export only public numerical allowlists."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vllm_prefix_scheduler.m4_contract import (  # noqa: E402
    audit,
    audit_resources,
    percentile,
    read_events,
    require,
    sha256,
    strict_int,
)

CPU_META = ("mode", "workload", "window", "outside_length")
CPU_PHASES = ("window", "score", "sort_exclusive", "commit", "restore_residual", "total")
CPU_CALLS = ("window", "score", "sort", "pop", "prepend", "guard_clock", "native_schedule")
CPU_VISITS = ("candidate", "queue", "hash_lookup", "pop", "prepend")
CPU_RESTORE_COUNTS = ("popleft_calls", "popleft_entries", "extendleft_calls", "extendleft_entries")


def public_cpu_costs(events):
    """Recompute numeric references/count ranges from the CPU rows already audited by main."""
    conditions = {
        (mode, workload, window, outside)
        for mode in ("native", "off", "prefix", "guarded")
        for workload in ("cold", "tied_hot", "mixed")
        for window in (8, 16, 32)
        for outside in (0, 128, 4096)
    }
    restoration_conditions = {
        key for key in conditions if key[0] in {"prefix", "guarded"} and key[1] == "mixed"
    }
    groups = {name: defaultdict(list) for name in ("normal", "restore", "unwrapped", "counter")}
    environments = []
    for event in events:
        kind = event.get("event")
        if kind == "m4_cpu_environment":
            environments.append(event)
            continue
        target = None
        if kind == "m4_cpu_sample":
            target = "normal" if event.get("variant") == "normal" else "restore"
        elif kind == "m4_cpu_unwrapped":
            target = "unwrapped"
        elif kind == "m4_cpu_restore_counter":
            target = "counter"
        if target is not None:
            key = tuple(event.get(name) for name in CPU_META)
            require(key in conditions, "CPU public metadata is outside the frozen allowlist")
            groups[target][key].append(event)
    require(len(environments) == 1, "CPU public timer environment missing/duplicated")
    require(
        set(groups["normal"]) == set(groups["unwrapped"]) == conditions
        and set(groups["restore"]) == set(groups["counter"]) == restoration_conditions,
        "CPU public reference/count matrix missing",
    )

    def meta(key):
        return dict(zip(CPU_META, key, strict=True))

    def distribution(values, count):
        require(
            len(values) == count and all(strict_int(value) for value in values),
            "CPU public duration sample count/value differs",
        )
        return {"n": count, "p50_ns": percentile(values, 0.50), "p95_ns": percentile(values, 0.95)}

    def ranges(records, field, keys, count):
        require(len(records) == count, "CPU public operation sample count differs")
        result = {}
        for name in keys:
            values = [row[field][name] for row in records]
            require(
                all(strict_int(value) for value in values), "CPU public operation count invalid"
            )
            result[name] = {"min": min(values), "max": max(values)}
        return result

    wrapper = environments[0]["empty_wrapper_ns"]
    unwrapped = []
    for key, records in sorted(groups["unwrapped"].items()):
        require(len(records) == 1, "CPU public unwrapped condition duplicated")
        unwrapped.append({**meta(key), **distribution(records[0]["samples_ns"], 100)})
    restore = [
        {
            **meta(key),
            "n": 5,
            "phases": {
                phase: distribution([row["duration_ns"][phase] for row in records], 5)
                for phase in CPU_PHASES
            },
        }
        for key, records in sorted(groups["restore"].items())
    ]
    return {
        "percentile_method": "nearest_rank",
        "duration_unit": "nanoseconds",
        "measurement_units": {
            "native_off": "one PAUSED_ALL native schedule",
            "prefix_guarded": "one waiting-window transaction",
        },
        "empty_wrapper": {
            name: distribution(wrapper[name], 1000)
            for name in ("bare_call_ns", "timed_wrapper_call_ns", "clock_pair_ns")
        },
        "unwrapped": unwrapped,
        "prepend_restore": restore,
        "normal_operation_counts": [
            {
                **meta(key),
                "n": 500,
                "calls": ranges(records, "calls", CPU_CALLS, 500),
                "visits": ranges(records, "visits", CPU_VISITS, 500),
            }
            for key, records in sorted(groups["normal"].items())
        ],
        "restore_operation_counts": [
            {
                **meta(key),
                "n": 5,
                "counts": ranges(records, "counts", CPU_RESTORE_COUNTS, 5),
            }
            for key, records in sorted(groups["counter"].items())
        ],
        "restore_counter_samples": sum(len(records) for records in groups["counter"].values()),
    }


def bind_layer_job(layer, events, sources, commit):
    """Derive one actual job from every raw row and the independently sealed runtime."""
    require(layer in {"native", "cpu", "gpu"}, "unknown raw evidence layer")
    require(
        type(sources) is dict and sources.get("project_commit") == commit,
        "layer source differs from final freeze",
    )
    require(type(events) is list and events, "layer raw events missing")
    if layer == "gpu":
        jobs = {row.get("campaign_job_id") for row in events}
        require(
            len(jobs) == 1 and all(type(job) is str and job.isdigit() for job in jobs),
            "actual GPU job binding missing/mixed",
        )
        job = next(iter(jobs))
        require(
            "job_id" not in sources or sources["job_id"] == job,
            "GPU source and raw job differ",
        )
    else:
        job = sources.get("job_id")
        require(
            type(job) is str
            and job.isdigit()
            and all(
                row.get("job_id") == job and row.get("project_commit") == commit for row in events
            ),
            "native/CPU raw and runtime job differ",
        )
    return job


def bind_final_jobs(layer_jobs, regression_jobs):
    """Only the frozen metadata+GPU two-job execution layout can form a final export."""
    require(
        type(layer_jobs) is dict
        and set(layer_jobs) == {"native", "cpu", "gpu"}
        and type(regression_jobs) is dict
        and set(regression_jobs) == {"compat", "m1", "full"},
        "complete native/CPU/GPU and regression layer bindings required",
    )
    jobs = list(layer_jobs.values()) + list(regression_jobs.values())
    require(
        all(type(job) is str and job.isdigit() for job in jobs),
        "final layer job identity missing/invalid",
    )
    metadata = layer_jobs["native"]
    require(
        layer_jobs["cpu"] == metadata
        and set(regression_jobs.values()) == {metadata}
        and layer_jobs["gpu"] != metadata,
        "final layers do not match distinct metadata/GPU roles",
    )
    return {"metadata": metadata, "gpu": layer_jobs["gpu"]}


def bind_regression_job(runtime, commit):
    """Read the existing M1/M2B runtime schema without changing historical producers."""
    require(
        type(runtime) is dict and runtime.get("project_commit") == commit,
        "baseline regression tested another freeze",
    )
    environment = runtime.get("environment")
    require(type(environment) is dict, "baseline regression environment missing")
    job = environment.get("SLURM_JOB_ID")
    require(type(job) is str and job.isdigit(), "baseline regression job binding missing")
    require(
        "job_id" not in runtime or runtime["job_id"] == job,
        "baseline regression job fields disagree",
    )
    return job


def bind_execution(commit, root=ROOT):
    names = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            "src",
            "scripts",
            "tests",
            "configs",
            "Makefile",
        ],
        text=True,
    ).splitlines()
    require(names, "empty tested source catalog")
    current = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "src", "scripts", "tests", "configs", "Makefile"],
        text=True,
    ).splitlines()
    require(set(names) == set(current), "execution source catalog differs from tested freeze")
    bound = {}
    for name in names:
        expected = hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{name}"]
            )
        ).hexdigest()
        path = root / name
        require(
            sha256(path) == expected,
            "currently executing source differs from tested freeze: " + name,
        )
        bound[str(path)] = expected
    return bound


def bind_collector(directory, commit, job, actual_files):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("project", {}).get("commit") == commit
        and manifest["project"].get("dirty") is False
        and manifest.get("source_changed_during_run") is False
        and manifest.get("config_inputs_stable") is True
        and str(manifest.get("allocation", {}).get("slurm_job_id")) == job,
        "collector freeze/job/stability differs",
    )
    bound = {str(manifest_path): sha256(manifest_path)}
    entries = manifest["files"]
    for entry in entries:
        path = directory / entry["path"]
        require(sha256(path) == entry["sha256"], "collector sealed artifact changed")
        bound[str(path)] = entry["sha256"]
    before = json.loads((directory / "source-before.json").read_text())
    after = json.loads((directory / "source-after.json").read_text())
    require(
        before == after and before.get("commit") == commit and before.get("dirty") is False,
        "collector source boundaries differ",
    )
    require(
        before.get("files_sha256") == manifest["project"].get("files_sha256"),
        "collector source fingerprint detached",
    )
    for actual in actual_files:
        digest = sha256(actual)
        matches = [
            entry
            for entry in entries
            if entry.get("sha256") == digest
            and entry.get("source", "").endswith("/" + actual.name)
            and entry["path"].startswith(("artifacts/", "configs/"))
        ]
        require(
            len(matches) == 1,
            "audited raw input is not the collector sealed artifact: " + actual.name,
        )
        bound[str(actual)] = digest
    return bound


def bind_resource_references(value, bound):
    if isinstance(value, list):
        for item in value:
            bind_resource_references(item, bound)
    elif isinstance(value, dict):
        if set(value) >= {"path", "sha256"}:
            path = Path(value["path"])
            require(sha256(path) == value["sha256"], "resource indirect original hash differs")
            if str(path) not in bound:
                bound[str(path)] = value["sha256"]
                if path.suffix == ".json":
                    bind_resource_references(json.loads(path.read_text()), bound)
        else:
            for item in value.values():
                bind_resource_references(item, bound)


def rerun_controls(config_path, data, resources_path, destination):
    """Execute fixed tested checkers; do not trust a supplied control summary."""
    destination.mkdir(parents=True, exist_ok=False)
    counts, bound = {}, {}
    for layer in ("native", "cpu", "gpu", "resources"):
        output = destination / layer
        if layer == "resources":
            command = [
                sys.executable,
                str(ROOT / "scripts/check_m4_resources_fail_closed.py"),
                "--config",
                str(config_path),
                "--resources",
                str(resources_path),
                "--output",
                str(output),
            ]
        else:
            folder = data / f"{layer}-data"
            command = [
                sys.executable,
                str(ROOT / "scripts/check_m4_fail_closed.py"),
                "--config",
                str(config_path),
                "--trace",
                str(folder / "trace.jsonl"),
                "--layer",
                layer,
                "--sources",
                str(folder / ("sources.json" if layer == "gpu" else "runtime.json")),
                "--output",
                str(output),
            ]
            if layer == "gpu":
                command += [
                    "--inputs",
                    str(data / "gpu-inputs.json"),
                    "--resources",
                    str(resources_path),
                ]
        child = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        (destination / f"{layer}-stdout.log").write_text(child.stdout)
        (destination / f"{layer}-stderr.log").write_text(child.stderr)
        (destination / f"{layer}-command.json").write_text(
            json.dumps({"command": command, "returncode": child.returncode}, indent=2) + "\n"
        )
        require(child.returncode == 0, "fresh fixed-source subprocess controls failed: " + layer)
        report = json.loads((output / "results.json").read_text())
        require(report.get("all_controls_passed") is True, "fresh controls incomplete")
        require(
            len(report["controls"]) == {"native": 17, "cpu": 5, "gpu": 18, "resources": 9}[layer],
            "fresh control matrix differs",
        )
        counts[layer] = len(report["controls"])
    for path in destination.rglob("*"):
        if path.is_file():
            bound[str(path)] = sha256(path)
    return counts, bound


def main():
    if not __debug__:
        raise RuntimeError("M4 public export requires assertions enabled")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--controls-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.config = args.config.resolve()
    args.data = args.data.resolve()
    args.resources = args.resources.resolve()
    args.controls_root = args.controls_root.resolve()
    args.output = args.output.resolve()
    require(
        any(args.controls_root.is_relative_to(ROOT / part) for part in ("runs", ".local")),
        "control replay must use the private ignored runs/.local tree",
    )
    require(
        not args.controls_root.is_relative_to(args.output)
        and not args.output.is_relative_to(args.controls_root),
        "private raw controls and public export destinations overlap",
    )
    require(not args.output.exists(), "public export output already exists")
    config = json.loads(args.config.read_text())
    resources = json.loads(args.resources.read_text())
    require(
        resources.get("config_sha256") == sha256(args.config),
        "resource catalog differs from final configuration",
    )
    reports, originals = {}, {"config": sha256(args.config), "resources": sha256(args.resources)}
    inputs = args.data / "gpu-inputs.json"
    initial_source = json.loads((args.data / "gpu-data/sources.json").read_text())
    commit = initial_source["project_commit"]
    bound = bind_execution(commit)
    bound[str(args.resources)] = sha256(args.resources)
    bind_resource_references(resources, bound)
    layer_jobs, regression_jobs = {}, {}
    cpu_costs = None
    for layer in ("native", "cpu", "gpu"):
        data = args.data / f"{layer}-data"
        source = data / ("sources.json" if layer == "gpu" else "runtime.json")
        trace = data / "trace.jsonl"
        events, source_record = read_events(trace), json.loads(source.read_text())
        job = bind_layer_job(layer, events, source_record, commit)
        layer_jobs[layer] = job
        reports[layer] = audit(
            config,
            events,
            layer=layer,
            sources=source_record,
            resources=resources if layer == "gpu" else None,
            root=ROOT,
            frozen_inputs=json.loads(inputs.read_text()) if layer == "gpu" else None,
            inputs_sha256=sha256(inputs) if layer == "gpu" else None,
        )
        require(reports[layer]["accepted"] is True, "M4 actual layer incomplete")
        if layer == "cpu":
            cpu_costs = public_cpu_costs(events)
        originals[f"{layer}_trace"] = sha256(trace)
        originals[f"{layer}_sources"] = sha256(source)
        bound.update(
            bind_collector(
                args.data / layer,
                reports[layer]["project_commit"],
                job,
                [source, trace, args.config, *([inputs] if layer == "gpu" else [])],
            )
        )
        # Schema2 command success/integrity is distinct from the behavioral audit above.
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_run_evidence.py"),
                str(args.data / layer),
                "--require-command-success",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    commits = {report["project_commit"] for report in reports.values()}
    require(len(commits) == 1, "M4 source freezes were mixed")
    require(commits == {commit}, "M4 layers differ from final execution freeze")
    regression = args.data / "regressions"
    for name, wanted in (("compat", 432), ("m1", 12), ("full", 636)):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_run_evidence.py"),
                str(regression / name),
                "--require-command-success",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        runtime = json.loads((regression / f"{name}-data/runtime.json").read_text())
        job = bind_regression_job(runtime, commit)
        regression_jobs[name] = job
        bound.update(
            bind_collector(
                regression / name,
                commit,
                job,
                [regression / f"{name}-data/runtime.json", regression / f"{name}-data/trace.jsonl"],
            )
        )
        originals[f"regression_{name}_runtime"] = sha256(regression / f"{name}-data/runtime.json")
        originals[f"regression_{name}_trace"] = sha256(regression / f"{name}-data/trace.jsonl")
        if name != "m1":
            checked = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tests/native/m2b_trace_check.py"),
                    "--config",
                    str(ROOT / "configs/native-m2b.json"),
                    "--trace",
                    str(regression / f"{name}-data/trace.jsonl"),
                    "--suite",
                    name,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            checked = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tests/native/test_native_replay.py"),
                    "--config",
                    str(ROOT / "configs/native-bc-replay.json"),
                    "--trace",
                    str(regression / "m1-data/trace.jsonl"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        outcome = json.loads(checked.stdout)
        require(
            outcome.get("audited_native_cases" if name == "m1" else "case_count") == wanted,
            "baseline regression raw case count differs",
        )
    selected_jobs = bind_final_jobs(layer_jobs, regression_jobs)
    reports["resources"] = audit_resources(
        resources, config, selected_commit=commit, selected_jobs=selected_jobs
    )
    control_counts, control_bound = rerun_controls(
        args.config, args.data, args.resources, args.controls_root
    )
    bound.update(control_bound)
    for layer in control_counts:
        originals[f"{layer}_controls"] = sha256(args.controls_root / layer / "results.json")
    summary = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "accepted": True,
        "baseline_commit": config["baseline_commit"],
        "tested_commit": commit,
        "upstream_commit": config["upstream_commit"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "platform": {
            "gpu": "RTX 3090",
            "memory_gib": 24,
            "dtype": "bfloat16",
            "vllm": "0.32.0",
            "python": "3.12.16",
        },
        "scope": "M4 scheduler stability and local CPU transaction costs; not M5",
        "native_cases": reports["native"]["case_count"],
        "gpu_cases": reports["gpu"]["case_count"],
        "native_long_rounds": 1200,
        "regression_cases": {"compat": 432, "m1": 12, "full": 636},
        "gpu_cases_summary": {
            key: {
                field: row[field]
                for field in ("requests", "batches", "empty_batches", "guard_registrations")
            }
            for key, row in reports["gpu"]["details"].items()
        },
        "cpu_groups": reports["cpu"]["details"]["groups"],
        "cpu_costs": cpu_costs,
        "control_counts": control_counts,
        "budget": {
            "allocations_used": len(resources["allocations"]),
            "allocation_limit": 3,
            "requested_minutes": reports["resources"]["requested_minutes"],
            "requested_minutes_limit": 60,
        },
        "limitations": [
            "CPU wrapper overhead included; restore residual includes exception bookkeeping",
            "native/off paused native schedule and prefix/guarded window transaction are different units",
            "no TTFT, throughput, natural-arrival fairness or speedup conclusion",
            "same-ID overlapping generations remain unsupported",
            "historical M3 formal-02 remains UNVERIFIED_PURGED; original full-history gate remains blocked",
        ],
    }
    provenance = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "tested_commit": commit,
        "raw_sha256": originals,
        "recomputed_audit_sha256": {
            layer: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            for layer, value in reports.items()
        },
        "private_data_exported": False,
    }
    for path, expected in bound.items():
        require(sha256(path) == expected, "bound input changed during export")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary", summary), ("provenance", provenance)):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print("Complete M4 public numerical allowlist exported")


if __name__ == "__main__":
    main()
