"""Local contract controls. These tests do not run vLLM or a GPU."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from m4_trace_fixture import Trace, gpu_fixture

from vllm_prefix_scheduler.m4_contract import (
    audit,
    audit_lifecycle,
    audit_probe,
    expected_cases,
    matrix_gate,
    source_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/m4-stability.json").read_text())


def test_frozen_matrix_cannot_be_shrunk():
    c = config()
    assert {layer: len(expected_cases(c, layer)) for layer in ("native", "gpu", "cpu")} == {
        "native": 105,
        "gpu": 60,
        "cpu": 540,
    }
    for key, value in (
        ("repeats", 1),
        ("native_rounds_per_seed", 99),
        ("seeds", [17]),
        ("window", True),
    ):
        changed = copy.deepcopy(c)
        changed[key] = value
        with pytest.raises(ValueError, match="frozen configuration"):
            expected_cases(changed, "gpu")
    changed = copy.deepcopy(c)
    changed["gpu_matrix"].pop()
    with pytest.raises(ValueError, match="GPU matrix"):
        expected_cases(changed, "gpu")


def test_complete_synthetic_gpu_bridge_is_only_contract_evidence():
    c = config()
    result = audit(c, gpu_fixture(c), layer="gpu", contract_fixture=True)
    assert result["case_count"] == 60
    assert result["scope"] == "contract_fixture"
    assert result["accepted"] is False
    assert result["complete_m4_acceptance"] is False


@pytest.mark.parametrize("operation", ["empty", "missing", "duplicate", "mislabeled"])
def test_matrix_boundaries_fail_closed(operation):
    c = config()
    events = gpu_fixture(c)
    if operation == "empty":
        events = []
    elif operation == "missing":
        case = events[0]["case_id"]
        events = [e for e in events if e["case_id"] != case]
    elif operation == "duplicate":
        events.append(copy.deepcopy(events[0]))
    else:
        events[0]["mode"] = "guarded"
    with pytest.raises(ValueError):
        matrix_gate(c, events, "gpu")


def test_safe_reuse_has_independent_first_dispatch_and_age():
    meta = {
        "mode": "guarded",
        "scenario": "long_cycle",
        "variant": "complete_cancel_reuse",
        "repeat": 0,
        "seed": 17,
    }
    trace = Trace(meta, "safe")
    trace.complete("same-id")
    trace.complete("intervening-id")
    trace.complete("same-id")
    result = audit_lifecycle(trace.events, "guarded")
    assert result["guard_registrations"] == 3
    assert len(result["bindings"]) == 3
    assert all(r["repeated_prompt_tokens"] == 0 for r in result["dispatch"].values())
    bad = [e for e in copy.deepcopy(trace.events) if e["event"] != "m4_generation_drained"]
    with pytest.raises(ValueError, match="ID reused before"):
        audit_lifecycle(bad, "guarded")


def test_probe_equality_is_recomputed_from_raw_storage():
    events = [
        {
            "event": kind,
            "seq": seq,
            "probe_id": "probe",
            "score": 0,
            "flags": {"full": True, "events": True, "stats": True, "metrics": True},
            "snapshot": {"free": [1, 2], "stats": {"hits": 3}},
        }
        for seq, kind in enumerate(("m4_probe_before", "m4_probe_after"), 1)
    ]
    audit_probe(events)
    disabled = copy.deepcopy(events)
    disabled[0]["flags"]["events"] = False
    with pytest.raises(ValueError, match="probe profile not active"):
        audit_probe(disabled)
    events[1]["snapshot"]["stats"]["hits"] += 1
    with pytest.raises(ValueError, match="probe changed"):
        audit_probe(events)


def test_missing_runtime_rejected_without_gpu_dependency_import():
    with pytest.raises(ValueError, match="source/runtime"):
        source_gate(None, config(), "gpu")
    assert "vllm" not in sys.modules
    assert "vllm_prefix_scheduler.scheduler" not in sys.modules


@pytest.mark.parametrize(
    "args,environment", [(["-O"], {}), (["-OO"], {}), ([], {"PYTHONOPTIMIZE": "1"})]
)
def test_optimized_child_refuses_before_argument_or_dependency_work(args, environment):
    child = subprocess.run(
        [sys.executable, *args, str(ROOT / "scripts/check_m4_trace.py")],
        text=True,
        capture_output=True,
        env={**os.environ, **environment},
    )
    assert child.returncode != 0
    assert "assertions enabled" in child.stderr


def test_resource_audit_requires_each_original_and_full_terminal(tmp_path):
    from m4_trace_fixture import resource_fixture

    from vllm_prefix_scheduler.m4_contract import audit_resources

    resources = resource_fixture(tmp_path / "originals")
    result = audit_resources(resources, config(), selected_commit="a" * 40, selected_job="12345")
    assert result["allocations"] == 1 and result["requested_minutes"] == 20
    bad = copy.deepcopy(resources)
    bad["allocations"][0]["evidence"].pop("terminal")
    with pytest.raises(ValueError, match="terminal/checkpoint"):
        audit_resources(bad, config())
    bad = copy.deepcopy(resources)
    bad["allocations"][0]["limit_minutes"] = 21
    with pytest.raises(ValueError, match="20 minute"):
        audit_resources(bad, config())
    bad = copy.deepcopy(resources)
    bad["allocations"] *= 4
    with pytest.raises(ValueError, match="budget"):
        audit_resources(bad, config())
    receipt = Path(resources["allocations"][0]["evidence"]["terminal"]["path"])
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="original missing/changed"):
        audit_resources(resources, config())


def test_unregistered_failed_submission_is_not_dropped(tmp_path):
    from m4_trace_fixture import resource_fixture

    from vllm_prefix_scheduler.m4_contract import audit_resources

    resources = resource_fixture(tmp_path / "originals")
    extra = Path(resources["submission_catalog_dir"]) / "attempt-02"
    extra.mkdir()
    (extra / "submission.json").write_text(json.dumps({"returncode": 2, "stdout": ""}))
    with pytest.raises(ValueError, match="omitted/unregistered"):
        audit_resources(resources, config())


@pytest.mark.parametrize("mode", ["prefix", "guarded"])
def test_window_tail_requires_actual_entry_of_original_outside_objects(mode):
    from vllm_prefix_scheduler.m4_contract import audit_barrier

    requests = [
        {"request_id": f"q{i}", "object_id": f"object:{i}", "generation": i} for i in range(18)
    ]
    records = (
        [{**request, "enqueued_ns": i + 1, "sequence": i} for i, request in enumerate(requests)]
        if mode == "guarded"
        else []
    )
    tail_records = records[16:] if records else []
    events = [
        {
            "event": "m4_barrier_check",
            "seq": 1,
            "before": requests,
            "after": requests,
            "candidates": requests[:16],
            "guard_before": records,
            "guard_after": records,
        },
        {"event": "m4_window", "seq": 2, "waiting": requests[16:], "candidates": requests[16:]},
        {
            "event": "m4_tail_entered",
            "seq": 3,
            "records_before": tail_records,
            "records_after": tail_records,
        },
    ]
    audit_barrier(events, "window_tail_age")
    with pytest.raises(ValueError, match="exactly one m4_tail_entered"):
        audit_barrier(events[:-1], "window_tail_age")
    wrong = copy.deepcopy(events)
    wrong[1]["waiting"] = requests[:2]
    wrong[1]["candidates"] = requests[:2]
    with pytest.raises(ValueError, match="earlier outside-window objects"):
        audit_barrier(wrong, "window_tail_age")
