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


def controls(path, expected_scope, original_hash):
    result = json.loads(path.read_text())
    require(result.get("scope") == expected_scope, "control scope differs from actual raw layer")
    records = result.get("controls")
    require(type(records) is list and records, "empty controls")
    operations = [row["operation"] for row in records]
    from check_m4_fail_closed import COMMON

    required = {"positive", "optimized"}
    if expected_scope == "resource_contract_controls":
        required |= {
            "missing_terminal",
            "omitted_job",
            "wrong_exit",
            "active_job",
            "watcher_missing",
            "watcher_started_late",
            "omitted_attempt",
        }
    elif expected_scope == "cpu":
        required |= {"cpu_scan", "cpu_io", "cpu_samples"}
    else:
        required |= set(COMMON)
        required |= (
            {"insufficient_rounds", "weakref_retained"}
            if expected_scope == "native"
            else {"probe_side_effect", "missing_terminal", "omitted_allocation"}
        )
    require(set(operations) == required, "required real subprocess negative controls missing")
    require(
        len(operations) == len(set(operations))
        and "positive" in operations
        and "optimized" in operations,
        "control catalog incomplete/duplicate",
    )
    if expected_scope == "resource_contract_controls":
        require(
            result.get("original_sha256") == original_hash,
            "resource controls use different raw index",
        )
    else:
        require(
            original_hash in result.get("originals", {}).values(),
            "controls use different raw trace",
        )
    for row in records:
        directory = path.parent / row["operation"]
        stdout = (directory / "stdout.log").read_text()
        stderr = (directory / "stderr.log").read_text()
        code = row.get("returncode")
        if row["operation"] == "positive":
            require(code == 0, "positive actual subprocess failed")
            if expected_scope != "resource_contract_controls":
                require(
                    json.loads(stdout)["accepted"] is True,
                    "positive control was only a fixture/partial",
                )
        else:
            require(
                type(code) is int and code != 0 and row["expected_error"] in stderr,
                "negative subprocess failed for an unintended reason",
            )
        for item in row.get("derived_inputs", []):
            require(sha256(item["path"]) == item["sha256"], "derived subprocess input hash differs")
        command = row["command"]
        for flag in ("--trace", "--resources"):
            if flag in command:
                require(
                    Path(command[command.index(flag) + 1]).is_file(),
                    "subprocess input was not retained",
                )
    return len(records)


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
    config = json.loads(args.config.read_text())
    resources = json.loads(args.resources.read_text())
    reports, originals = {}, {"config": sha256(args.config), "resources": sha256(args.resources)}
    inputs = args.data / "gpu-inputs.json"
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
    control_counts = {}
    for layer in ("native", "cpu", "gpu", "resources"):
        path = args.controls_root / layer / "results.json"
        control_counts[layer] = controls(
            path,
            "resource_contract_controls" if layer == "resources" else layer,
            originals["resources" if layer == "resources" else f"{layer}_trace"],
        )
        originals[f"{layer}_controls"] = sha256(path)
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
            "requested_minutes": len(resources["allocations"]) * 20,
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
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary", summary), ("provenance", provenance)):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print("Complete M4 public numerical allowlist exported")


if __name__ == "__main__":
    main()
