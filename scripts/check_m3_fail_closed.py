"""Real subprocess positive/negative controls over a complete M3 raw trace.

This never creates GPU evidence. Supply either real archived evidence or an
explicit contract-test fixture, and preserve that distinction in the report.
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

from vllm_prefix_scheduler.m3_contract import (  # noqa: E402
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
    if operation in {"missing_case", "missing_repeat", "missing_mode"}:

        def drop(event):
            case = event.get("case_id", "")
            if operation == "missing_case":
                return case == "cancel:guarded"
            if operation == "missing_repeat":
                return case.startswith("mechanism:") and case.endswith(":2")
            return event.get("engine_id") == "smoke-prefix" or case == "smoke:prefix"

        return [event for event in data if not drop(event)]
    if operation == "duplicate_case":
        event = next(e for e in data if e["event"] == "case_begin")
        duplicate = copy.deepcopy(event)
        duplicate["seq"] = max(e["seq"] for e in data if e["pid"] == event["pid"]) + 1
        duplicate["monotonic_ns"] = (
            max(e["monotonic_ns"] for e in data if e["pid"] == event["pid"]) + 1
        )
        data.append(duplicate)
    elif operation == "mislabeled_case":
        next(e for e in data if e["event"] == "case_begin")["variant"] = "wrong-label"
    elif operation == "missing_sources":
        next(e for e in data if e["event"] == "engine_ready")["sources"] = {}
    elif operation == "missing_return":
        target = next(e for e in data if e["event"] == "model_output_materialized")
        data.remove(target)
    elif operation == "orphan_batch":
        next(e for e in data if e["event"] == "runner_execute_success")["batch_id"] = "nonexistent"
    elif operation == "fifo_reordered":
        targets = [e for e in data if e["event"] == "update_from_output_start"]
        require(len(targets) > 1, "FIFO control requires at least two outputs")
        targets[0]["batch_id"], targets[1]["batch_id"] = (
            targets[1]["batch_id"],
            targets[0]["batch_id"],
        )
    elif operation == "wrong_output":
        target = next(e for e in data if e["event"] == "client_done")
        target["token_ids"][0] += 1
    elif operation == "unfinished_request":
        data.remove(next(e for e in data if e["event"] == "client_done"))
    elif operation == "native_resource_leak":
        target = next(e for e in data if e["event"] == "scheduler_shutdown_before")
        target["resources"]["non_null_ref_sum"] = 1
    elif operation == "process_resource_leak":
        next(e for e in data if e["event"] == "process_exit")["owned_pids_remaining"] = [987654]
    elif operation == "refreshed_guard_age":
        target = next(e for e in data if e["event"] == "decision" and e.get("now_ns") is not None)
        target["records"][target["before"][0]]["enqueued_ns"] += 1
    elif operation == "observer_error":
        next(e for e in data if e["event"] == "engine_shutdown")["observation_errors"] = [
            "lost event"
        ]
    else:
        raise ValueError("unknown mutation")
    return data


NEGATIVES = {
    "empty": "empty raw event set",
    "missing_case": "entire unique frozen matrix",
    "missing_repeat": "entire unique frozen matrix",
    "missing_mode": "entire unique frozen matrix",
    "duplicate_case": "entire unique frozen matrix",
    "mislabeled_case": "case metadata mislabeled",
    "missing_sources": "source checkout was not clean",
    "missing_return": "exactly one model_output_materialized",
    "orphan_batch": "exactly one runner_execute_success",
    "fifo_reordered": "batch execution and consumption ordering broken",
    "wrong_output": "final async output differs",
    "unfinished_request": "client completions missing",
    "native_resource_leak": "native resource not zero",
    "process_resource_leak": "owned processes did not exit",
    "refreshed_guard_age": "guard enqueue age or sequence was refreshed",
    "observer_error": "output processor/observer not drained",
}


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract-fixture",
        action="store_true",
        help="Explicitly label synthetic contract-test input; never GPU acceptance",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    events = read_events(args.trace_dir)
    clean = [{k: v for k, v in e.items() if not k.startswith("_")} for e in events]
    config, resources = json.loads(args.config.read_text()), json.loads(args.resources.read_text())
    source_paths = (
        [args.trace_dir] if args.trace_dir.is_file() else sorted(args.trace_dir.rglob("*.jsonl"))
    )
    originals = {str(path): sha256(path) for path in source_paths}
    records = []
    plans = [
        ("positive", None),
        *NEGATIVES.items(),
        ("optimized", "assertions enabled"),
        ("reduced_repeats", "exactly three independent repeats"),
        ("active_job", "final valid owned-job query"),
    ]
    for name, fragment in plans:
        directory = args.output / name
        directory.mkdir()
        data = (
            clean
            if name in {"positive", "optimized", "reduced_repeats", "active_job"}
            else mutate(clean, name)
        )
        trace = directory / "trace.jsonl"
        trace.write_text("".join(canonical(e) + "\n" for e in data))
        child_config, child_resources = copy.deepcopy(config), copy.deepcopy(resources)
        if name == "reduced_repeats":
            child_config["repeats"] = 1
        if name == "active_job":
            child_resources["active_owned_jobs"] = ["123"]
        (directory / "config.json").write_bytes(
            canonical(child_config).encode()
            if name == "reduced_repeats"
            else args.config.read_bytes()
        )
        (directory / "resources.json").write_text(canonical(child_resources))
        command = [
            sys.executable,
            *(["-O"] if name == "optimized" else []),
            str(ROOT / "scripts/check_m3_trace.py"),
            "--config",
            str(directory / "config.json"),
            "--trace-dir",
            str(directory),
            "--resources",
            str(directory / "resources.json"),
        ]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONOPTIMIZE"}
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
        (directory / "stdout.txt").write_text(result.stdout)
        (directory / "stderr.txt").write_text(result.stderr)
        accepted = (
            result.returncode == 0
            if name == "positive"
            else result.returncode != 0 and fragment in result.stderr
        )
        records.append(
            {
                "name": name,
                "argv": command,
                "returncode": result.returncode,
                "expected_error": fragment,
                "passed": accepted,
                "trace_sha256": sha256(trace),
                "stdout_sha256": sha256(directory / "stdout.txt"),
                "stderr_sha256": sha256(directory / "stderr.txt"),
            }
        )
        (args.output / "results.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_kind": "synthetic_contract_fixture"
                    if args.contract_fixture
                    else "archived_real_trace",
                    "source_sha256": originals,
                    "auditor_sha256": sha256(ROOT / "src/vllm_prefix_scheduler/m3_contract.py"),
                    "results": records,
                },
                indent=2,
            )
            + "\n"
        )
        require(accepted, f"subprocess control {name} did not meet its specific expected result")
    require(
        originals == {str(path): sha256(path) for path in source_paths},
        "control source trace changed",
    )
    print(
        json.dumps(
            {
                "controls": len(records),
                "passed": True,
                "input_kind": "synthetic_contract_fixture"
                if args.contract_fixture
                else "archived_real_trace",
            }
        )
    )


if __name__ == "__main__":
    main()
