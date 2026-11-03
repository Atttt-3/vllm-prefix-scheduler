"""Run real child-process controls over an unchanged complete M4 raw trace.

Contract fixtures remain explicitly local. Each mutation is a retained derived
copy; original trace, configuration and resource records are never overwritten.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m4_contract import (  # noqa: E402
    canonical,
    read_events,
    require,
    require_assertions,
    sha256,
)


def mutate(events, operation):
    data = copy.deepcopy(events)
    if operation == "empty":
        return []
    if operation in {"missing_case", "insufficient_repeats"}:
        begins = [e for e in data if e["event"] == "m4_case_begin"]
        require(begins, "case control has no applicable case")
        if operation == "missing_case":
            return [e for e in data if e.get("case_id") != begins[-1]["case_id"]]
        return [e for e in data if e.get("repeat") != 2]
    if operation == "duplicate_case":
        target = next(e for e in data if e["event"] == "m4_case_begin")
        duplicate = copy.deepcopy(target)
        duplicate["seq"] = (
            max(e["seq"] for e in data if e.get("engine_id") == target.get("engine_id")) + 1
        )
        duplicate["monotonic_ns"] = max(e.get("monotonic_ns", 0) for e in data) + 1
        data.append(duplicate)
    elif operation == "mislabeled_case":
        next(e for e in data if e["event"] == "m4_case_begin")["seed"] = 999
    elif operation == "no_chunk":
        for event in data:
            if event["event"] == "m4_schedule":
                for state in event.get("states", []):
                    state["num_computed_tokens"] = state["prompt_length"]
    elif operation == "no_preempt":
        data = [e for e in data if e["event"] != "m4_preempt_success"]
    elif operation == "no_pending_cancel":
        for event in data:
            if event["event"] == "m4_cancel_start":
                event["pending"] = []
                for request in event.get("requests", []):
                    request["num_in_flight_tokens"] = request["num_output_placeholders"] = 0
    elif operation == "unsafe_id_reuse":
        data = [e for e in data if e["event"] != "m4_generation_drained"]
    elif operation == "probe_side_effect":
        next(e for e in data if e["event"] == "m4_probe_after")["snapshot"][
            "unexpected_mutation"
        ] = 1
    elif operation == "resource_residue":
        next(e for e in data if e["event"] == "m4_generation_drained")["resources"][
            "non_null_ref_sum"
        ] = 1
    elif operation == "lost_output":
        target = next(e for e in data if e["event"] == "m4_consume_success")
        data.remove(target)
    elif operation == "lost_empty_notification":
        target = next(e for e in data if e["event"] == "m4_schedule" and not e["intervals"])
        batch = target["batch_id"]
        data = [
            e
            for e in data
            if not (
                e.get("batch_id") == batch
                and e["event"] in {"m4_consume_start", "m4_consume_success"}
            )
        ]
    elif operation == "guard_refresh":
        target = next(e for e in data if e["event"] == "m4_schedule" and e.get("guard_before"))
        target["guard_before"][0]["enqueued_ns"] += 1
    elif operation == "insufficient_rounds":
        data = [e for e in data if e.get("round") != 99]
    elif operation == "weakref_retained":
        next(e for e in data if e["event"] == "m4_round_end")["weak_refs_alive"] = 1
    elif operation == "cpu_scan":
        target = next(
            e for e in data if e["event"] == "m4_cpu_sample" and e["outside_length"] == 4096
        )
        target["visits"]["queue"] += 4096
    elif operation == "cpu_io":
        next(e for e in data if e["event"] == "m4_cpu_sample")["observer_io"] = True
    elif operation == "cpu_samples":
        data.remove(next(e for e in data if e["event"] == "m4_cpu_sample"))
    else:
        raise ValueError(f"unknown mutation {operation}")
    return data


COMMON = {
    "empty": "empty raw event set",
    "missing_case": "entire unique frozen matrix",
    "duplicate_case": "duplicate case",
    "mislabeled_case": "case metadata mislabeled",
    "insufficient_repeats": "entire unique frozen matrix",
    "no_chunk": "multi-chunk",
    "no_preempt": "preemption/resume",
    "no_pending_cancel": "pending",
    "unsafe_id_reuse": "ID reused before",
    "resource_residue": "native resource",
    "lost_output": "causal drain",
    "lost_empty_notification": "causal drain",
    "guard_refresh": "guard before",
}


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layer", choices=("native", "gpu", "cpu"), required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-fixture", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    events = [
        {k: v for k, v in event.items() if not k.startswith("_")}
        for event in read_events(args.trace)
    ]
    original_paths = [args.trace] if args.trace.is_file() else sorted(args.trace.rglob("*.jsonl"))
    originals = {str(path): sha256(path) for path in original_paths}
    negatives = (
        {"cpu_scan": "candidate accesses", "cpu_io": "IO", "cpu_samples": "sample count"}
        if args.layer == "cpu"
        else dict(COMMON)
    )
    if args.layer == "native":
        negatives.update(insufficient_rounds="100 long-loop", weakref_retained="weak references")
    if args.layer == "gpu":
        negatives["probe_side_effect"] = "probe changed"
    plans = [("positive", None), *negatives.items(), ("optimized", "assertions enabled")]
    if args.resources and args.layer == "gpu":
        plans += [
            ("missing_terminal", "terminal/checkpoint"),
            ("omitted_allocation", "accepted job"),
        ]
    records = []
    for operation, expected_error in plans:
        directory = args.output / operation
        directory.mkdir()
        unchanged = operation in {"positive", "optimized", "missing_terminal", "omitted_allocation"}
        data = events if unchanged else mutate(events, operation)
        derived = directory / "derived"
        derived.mkdir(exist_ok=False)
        trace = derived / "trace.jsonl"
        with trace.open("x") as stream:
            stream.write("".join(canonical(event) + "\n" for event in data))
        command = [
            sys.executable,
            *(["-O"] if operation == "optimized" else []),
            str(ROOT / "scripts/check_m4_trace.py"),
            "--config",
            str(args.config.resolve()),
            "--trace",
            str(trace),
            "--layer",
            args.layer,
            "--output",
            str(directory / "audit.json"),
        ]
        if args.contract_fixture:
            command += ["--contract-fixture"]
        if args.inputs:
            command += ["--inputs", str(args.inputs.resolve())]
        if args.sources:
            command += ["--sources", str(args.sources.resolve())]
        if args.resources:
            resources_path = args.resources.resolve()
            if operation in {"missing_terminal", "omitted_allocation"}:
                resource = json.loads(resources_path.read_text())
                if operation == "missing_terminal":
                    resource["allocations"][0]["evidence"].pop("terminal")
                else:
                    resource["allocations"].pop()
                resources_path = derived / "resources.json"
                with resources_path.open("x") as stream:
                    stream.write(canonical(resource))
            command += ["--resources", str(resources_path)]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONOPTIMIZE": "0"},
        )
        (directory / "stdout.log").write_text(result.stdout)
        (directory / "stderr.log").write_text(result.stderr)
        record = {
            "operation": operation,
            "command": command,
            "returncode": result.returncode,
            "expected_error": expected_error,
            "mutation_rows": len(data),
            "derived_trace_sha256": sha256(trace),
            "derived_inputs": [
                {"path": str(path), "sha256": sha256(path)}
                for path in sorted(derived.iterdir())
                if path.is_file()
            ],
            "working_directory": str(ROOT),
            "success": result.returncode == 0
            if operation == "positive"
            else result.returncode != 0 and expected_error in result.stderr,
        }
        records.append(record)
        (args.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        require(
            record["success"],
            f"real subprocess control did not reach intended outcome: {operation}",
        )
    require(
        all(sha256(path) == wanted for path, wanted in originals.items()),
        "control changed original trace",
    )
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "task_id": "M4-scheduler-stability",
                "scope": "contract_fixture" if args.contract_fixture else args.layer,
                "originals": originals,
                "controls": records,
                "all_controls_passed": True,
                "complete_m4_acceptance": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"controls": len(records), "all_controls_passed": True}))


if __name__ == "__main__":
    main()
