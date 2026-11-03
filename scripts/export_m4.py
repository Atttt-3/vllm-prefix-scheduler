"""Re-audit complete M4 raw evidence and export only public numerical allowlists."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vllm_prefix_scheduler.m4_contract import audit, read_events, require, sha256  # noqa: E402


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
    reports, originals = {}, {"config": sha256(args.config), "resources": sha256(args.resources)}
    inputs = args.data / "gpu-inputs.json"
    initial_source = json.loads((args.data / "gpu-data/sources.json").read_text())
    bound = bind_execution(initial_source["project_commit"])
    bound[str(args.resources)] = sha256(args.resources)
    bind_resource_references(resources, bound)
    gpu_rows = read_events(args.data / "gpu-data/trace.jsonl")
    jobs = {str(row.get("campaign_job_id")) for row in gpu_rows}
    require(len(jobs) == 1 and next(iter(jobs)).isdigit(), "actual GPU job binding missing")
    job = next(iter(jobs))
    for layer in ("native", "cpu", "gpu"):
        data = args.data / f"{layer}-data"
        source = data / ("sources.json" if layer == "gpu" else "runtime.json")
        trace = data / "trace.jsonl"
        reports[layer] = audit(
            config,
            read_events(trace),
            layer=layer,
            sources=json.loads(source.read_text()),
            resources=resources if layer == "gpu" else None,
            root=ROOT,
            frozen_inputs=json.loads(inputs.read_text()) if layer == "gpu" else None,
            inputs_sha256=sha256(inputs) if layer == "gpu" else None,
        )
        require(reports[layer]["accepted"] is True, "M4 actual layer incomplete")
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
    commit = next(iter(commits))
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
        require(
            runtime.get("project_commit") == commit, "baseline regression tested another freeze"
        )
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
        "control_counts": control_counts,
        "budget": {
            "allocations_used": len(resources["allocations"]),
            "allocation_limit": 3,
            "requested_minutes": reports["gpu"]["resources"]["requested_minutes"],
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
