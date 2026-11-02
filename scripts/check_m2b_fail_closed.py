"""Real subprocess negatives derived from a complete independently accepted trace.

Small original-case excerpts are retained for primitive tampering to avoid copying
the complete large native trace for every negative. They are negative tests only;
the baseline must first pass the complete M2B audit and cannot be replaced by an
excerpt or summary. No source/environment/lock mutation is performed.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/native"))

from vllm_prefix_scheduler.m2b_contract import (  # noqa: E402
    expected_cases,
    require,
    require_assertions,
    verify_result_cases,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def child_check(kind, payload, config):
    """Called in a fresh process; never reports native-matrix acceptance."""
    from m2b_trace_check import audit_leaf

    expected = expected_cases(config)
    if kind == "results":
        verify_result_cases(payload, config)
    elif kind == "config":
        expected_cases(payload)
    elif kind == "leaf":
        case = payload[0].get("case") if payload else None
        require(case in expected, "negative excerpt has unexpected case")
        meta = expected[case]
        require(
            payload[0]["event"] == "case_begin" and payload[-1]["event"] == "case_end",
            "leaf boundaries missing",
        )
        require(
            sum(e["event"] == "case_begin" for e in payload)
            == sum(e["event"] == "case_end" for e in payload)
            == 1,
            "duplicate leaf boundaries",
        )
        for event in payload:
            require(
                event.get("case") == case and all(event.get(k) == v for k, v in meta.items()),
                "excerpt case mislabeled",
            )
            require(type(event.get("repeat")) is int, "repeat must be strict integer")
        audit_leaf(payload, meta, config)
    else:
        raise ValueError("unknown negative child operation")


def primitive_definitions():
    """Each corruption changes a raw fact or required identity, never a summary label."""
    return [
        (
            "raw_clock_type",
            lambda e: e["event"] == "clock" and type(e.get("value")) is int,
            lambda es, i: es[i].update(value=True),
        ),
        (
            "raw_clock_value",
            lambda e: e["event"] == "clock" and e.get("purpose") == "enqueue",
            lambda es, i: es[i].update(value=es[i]["value"] + 1),
        ),
        (
            "enqueue_sequence",
            lambda e: bool(e.get("guard", {}).get("records")),
            lambda es, i: es[i]["guard"]["records"][0].update(sequence=999999),
        ),
        (
            "enqueue_time",
            lambda e: bool(e.get("guard", {}).get("records")),
            lambda es, i: es[i]["guard"]["records"][0].update(enqueued_ns=999999),
        ),
        (
            "object_binding",
            lambda e: e["event"] == "native_add_return" and e.get("actual_object") is not None,
            lambda es, i: es[i].update(actual_object="object:missing"),
        ),
        (
            "object_generation",
            lambda e: e["event"] == "binding",
            lambda es, i: es[i].update(generation=es[i]["generation"] + 1),
        ),
        (
            "decision_order",
            lambda e: e["event"] == "decision_after" and len(e.get("objects", [])) > 1,
            lambda es, i: es[i].update(objects=list(reversed(es[i]["objects"]))),
        ),
        (
            "window_suffix",
            lambda e: e["event"] == "window_after" and bool(e.get("after")),
            lambda es, i: es[i]["after"].append("injected_suffix"),
        ),
        (
            "native_dispatch_tokens",
            lambda e: e["event"] == "native_schedule_return" and bool(e.get("scheduled")),
            lambda es, i: es[i]["scheduled"].update({next(iter(es[i]["scheduled"])): 99999}),
        ),
        (
            "dispatch_interval",
            lambda e: e["event"] == "dispatch" and bool(e.get("intervals")),
            lambda es, i: es[i]["intervals"][0].update(end=es[i]["intervals"][0]["end"] + 1),
        ),
        (
            "cleanup_reason",
            lambda e: e["event"] == "guard_after" and e.get("operation") == "discard",
            lambda es, i: es[i].update(reason="invented"),
        ),
        (
            "cleanup_record_retained",
            lambda e: (
                e["event"] == "guard_after"
                and e.get("operation") == "discard"
                and e.get("result") is True
            ),
            lambda es, i: es[i].update(result=False),
        ),
        (
            "cleanup_completion_missing",
            lambda e: e["event"] == "guard_after" and e.get("operation") == "discard",
            lambda es, i: es.pop(i),
        ),
        ("native_return_missing", lambda e: e["event"] == "return", lambda es, i: es.pop(i)),
        (
            "native_dispatch_return_missing",
            lambda e: e["event"] == "native_schedule_return",
            lambda es, i: es.pop(i),
        ),
        (
            "native_source_hash",
            lambda e: e["event"] == "native_objects",
            lambda es, i: es[i]["sources"][
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
            ].update(sha256="0" * 64),
        ),
        (
            "parallel_fields_missing",
            lambda e: e["event"] == "native_objects",
            lambda es, i: es[i]["effective_config"].update(parallel={}),
        ),
        (
            "false_fault",
            lambda e: e.get("guard", {}).get("present") is True,
            lambda es, i: es[i]["guard"].update(fault_code="native_schedule"),
        ),
        (
            "wrong_case_label",
            lambda e: e["event"] == "case_begin",
            lambda es, i: es[i].update(variant="invented"),
        ),
    ]


PRIMITIVE_ERRORS = {
    "raw_clock_type": "invalid clock was accepted",
    "raw_clock_value": "last clock differs from raw reads",
    "enqueue_sequence": "live records differ from native/clock oracle",
    "enqueue_time": "live records differ from native/clock oracle",
    "object_binding": "object has no successful native binding",
    "object_generation": "native generation not independently increasing",
    "decision_order": "actual decision differs from independent protection/FIFO/prefix order",
    "window_suffix": "window suffix/order changed outside policy",
    "native_dispatch_tokens": "native dispatch token map changed after return",
    "dispatch_interval": "scheduled interval size mismatch",
    "cleanup_reason": "guard call/completion mismatch",
    "cleanup_record_retained": "cleanup return disagrees with actual live state",
    "cleanup_completion_missing": "positive dispatch retained protection",
    "native_return_missing": "unfinished native call",
    "native_dispatch_return_missing": "actual native return/dispatch/outer output steps differ",
    "native_source_hash": "unverified native class source",
    "parallel_fields_missing": "effective native parallel semantics changed",
    "false_fault": "fault state differs from causal events",
    "wrong_case_label": "excerpt case mislabeled",
}


def select_primitive_sources(trace, expected):
    """Retain only the few original leaves needed by all primitive negatives."""
    from m2b_trace_check import iter_case_groups

    definitions = primitive_definitions()
    selected = {}
    for case, events in iter_case_groups(trace, expected):
        if any(e["event"] in {"expected_error", "fault_injection"} for e in events):
            continue
        for label, predicate, _ in definitions:
            if label in selected:
                continue
            for index, event in enumerate(events):
                if predicate(event):
                    selected[label] = (case, events, index)
                    break
        if len(selected) == len(definitions):
            break
    require(
        set(selected) == {label for label, _, _ in definitions},
        "required negative primitive absent from accepted trace",
    )
    return selected


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native-m2b.json")
    parser.add_argument("--child-check", choices=("results", "config", "leaf"))
    parser.add_argument("--payload", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.child_check:
        require(args.payload is not None, "child payload is required")
        child_check(args.child_check, json.loads(args.payload.read_text()), config)
        return
    require(sys.platform == "linux", "real missing-native gate requires Linux")
    require(
        args.output is not None and args.trace is not None,
        "--output and complete --trace are required",
    )
    from m2b_trace_check import assert_trace

    accepted = assert_trace(args.trace, config)
    require(accepted["matrix_verified"] is True, "negative baseline not fully accepted")
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "baseline.json",
        {
            "trace_sha256": fingerprint(args.trace),
            "case_count": accepted["case_count"],
            "matrix_verified": True,
            "note": "Primitive excerpts below are only negative tests, never full-matrix evidence.",
        },
    )
    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    environment.update(
        PYTHONHASHSEED=str(config["seed"]), VLLM_USE_V2_MODEL_RUNNER="1", HF_HUB_OFFLINE="1"
    )
    commands = []
    this = Path(__file__).resolve()

    def child(label, kind, payload, fragment=None):
        path = args.output / f"{label}.json"
        write(path, payload)
        commands.append(
            (
                label,
                [
                    sys.executable,
                    str(this),
                    "--child-check",
                    kind,
                    "--payload",
                    str(path),
                    "--config",
                    str(args.config),
                ],
                fragment,
                {},
            )
        )

    expected = expected_cases(config)
    result_rows = [{"case": case, **meta, "status": "passed"} for case, meta in expected.items()]
    child("positive_config", "config", config)
    child("positive_result_catalog", "results", result_rows)
    for label, payload, fragment in (
        ("empty_results", [], "results must be a nonempty list"),
        ("missing_result", result_rows[:-1], "result matrix is incomplete"),
        ("duplicate_result", result_rows + result_rows[:1], "duplicate result case"),
        ("extra_result", result_rows + [{"case": "extra"}], "unexpected result case"),
    ):
        child(label, "results", payload, fragment)
    for label, field, value in (
        ("mislabeled_result", "variant", "invented"),
        ("skipped_result", "status", "skipped"),
    ):
        altered = copy.deepcopy(result_rows)
        altered[0][field] = value
        child(
            label,
            "results",
            altered,
            "result identity mismatch"
            if field == "variant"
            else "result failed, skipped or incomplete",
        )
    for label, mutate, fragment in (
        (
            "empty_matrix",
            lambda c: c.update(scenarios={}),
            "complete 50-combination scenario catalog differs",
        ),
        ("insufficient_repeats", lambda c: c.update(repeats=2), "repeats must be an integer >= 3"),
        ("missing_leaf", lambda c: c["leaves"].pop(), "named leaf catalog is missing"),
        (
            "duplicate_leaf",
            lambda c: c["leaves"].append(c["leaves"][0]),
            "named leaf catalog is missing",
        ),
        (
            "extra_leaf",
            lambda c: c["leaves"].append(
                {"scenario": "invented", "mode": "guarded", "variant": "default"}
            ),
            "named leaf catalog is missing",
        ),
        (
            "invalid_config",
            lambda c: c.update(max_wait_ms=True),
            "frozen native baseline/input/source configuration drift",
        ),
        (
            "clock_plan_drift",
            lambda c: c["clock_plans"]["exact"].update(decision_ns=1),
            "frozen clock plan differs",
        ),
    ):
        altered = copy.deepcopy(config)
        mutate(altered)
        child(label, "config", altered, fragment)
    mutation_index = []
    selected = select_primitive_sources(args.trace, expected)
    positive_sources = {}
    for label, _, mutate in primitive_definitions():
        case, original, index = selected[label]
        if case not in positive_sources:
            positive = f"positive_native_leaf_{len(positive_sources)}"
            positive_sources[case] = positive
            child(positive, "leaf", original)
        payload = copy.deepcopy(original)
        mutate(payload, index)
        child(label, "leaf", payload, PRIMITIVE_ERRORS[label])
        mutation_index.append(
            {
                "negative": label,
                "source_case": case,
                "positive_control": positive_sources[case],
                "target_error": PRIMITIVE_ERRORS[label],
            }
        )
    write(args.output / "primitive-index.json", mutation_index)
    empty = args.output / "empty.jsonl"
    empty.write_text("")
    commands.append(
        (
            "empty_trace",
            [sys.executable, str(ROOT / "tests/native/m2b_trace_check.py"), "--trace", str(empty)],
            "nonempty",
            {},
        )
    )
    commands.append(
        (
            "missing_native",
            [
                sys.executable,
                "-S",
                str(ROOT / "scripts/replay_m2b.py"),
                "--model-config",
                str(args.output),
                "--output",
                str(args.output / "missing-native-data"),
            ],
            "No module named 'vllm'",
            {},
        )
    )
    for script in ("scripts/replay_m2b.py", "tests/native/m2b_trace_check.py"):
        name = Path(script).stem
        for flag in ("-O", "-OO"):
            commands.append(
                (
                    name + flag,
                    [sys.executable, flag, str(ROOT / script), "--help"],
                    "assertions enabled",
                    {},
                )
            )
        for level in ("1", "2"):
            commands.append(
                (
                    name + "-env" + level,
                    [sys.executable, str(ROOT / script), "--help"],
                    "assertions enabled",
                    {"PYTHONOPTIMIZE": level},
                )
            )
    results = []
    for label, command, fragment, changes in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, env={**environment, **changes}, timeout=60
        )
        (args.output / f"{label}.stdout").write_text(result.stdout)
        (args.output / f"{label}.stderr").write_text(result.stderr)
        passed = (
            result.returncode == 0
            if fragment is None
            else result.returncode != 0 and fragment in result.stderr
        )
        results.append(
            {
                "case": label,
                "argv": command,
                "returncode": result.returncode,
                "positive_control": fragment is None,
                "expected_error": fragment,
                "passed": passed,
                "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            }
        )
        write(args.output / "results.json", results)
        require(passed, f"subprocess {label} did not satisfy its exact positive/negative control")
    print(
        json.dumps(
            {
                "fail_closed_cases": sum(not r["positive_control"] for r in results),
                "positive_controls": sum(r["positive_control"] for r in results),
                "all_passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
