"""Local contract and independent oracle negatives, never native acceptance."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m2b_contract import (
    SCENARIOS,
    expected_cases,
    leaf_catalog,
    verify_result_cases,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/native-m2b.json").read_text())


@pytest.fixture(scope="module")
def auditor():
    spec = importlib.util.spec_from_file_location(
        "m2b_oracle_tests", ROOT / "tests/native/m2b_trace_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def negative_runner():
    spec = importlib.util.spec_from_file_location(
        "m2b_negative_tests", ROOT / "scripts/check_m2b_fail_closed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_catalog_expands_every_leaf_three_times(config):
    cases = expected_cases(config)
    assert sum(map(len, SCENARIOS.values())) == 50
    assert len(leaf_catalog()) == 212
    assert len(cases) == 636
    assert len(expected_cases(config, suite="compat")) == 432
    assert all(
        sum(meta == {"repeat": r, **leaf} for meta in cases.values()) == 1
        for leaf in leaf_catalog()
        for r in range(3)
    )
    copy_catalog = leaf_catalog()
    copy_catalog[0]["mode"] = "bad"
    assert leaf_catalog()[0]["mode"] == "native"


@pytest.mark.parametrize("repeats", [None, False, True, -1, 0, 1, 2, 3.0, "3"])
def test_insufficient_or_invalid_repeats_fail(config, repeats):
    config["repeats"] = repeats
    with pytest.raises(ValueError, match="repeats"):
        expected_cases(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(scenarios={}),
        lambda c: c["scenarios"].pop("preemption"),
        lambda c: c["scenarios"].update(extra=["guarded"]),
        lambda c: c["scenarios"]["hot_flow"].append("guarded"),
        lambda c: c.update(leaves=[]),
        lambda c: c["leaves"].pop(),
        lambda c: c["leaves"].append(c["leaves"][0]),
        lambda c: c["leaves"][0].update(variant="wrong"),
        lambda c: c["leaves"].reverse(),
        lambda c: c.update(seed=43),
        lambda c: c.update(window=True),
        lambda c: c.update(vllm_commit="main"),
        lambda c: c["source_sha256"].update({"vllm/v1/request.py": "0" * 64}),
        lambda c: c["requests"][0]["tokens"].append(42),
        lambda c: c["clock_plans"]["exact"].update(decision_ns=499999999),
    ],
)
def test_matrix_inputs_cannot_be_silently_changed(config, mutation):
    mutation(config)
    with pytest.raises(ValueError):
        expected_cases(config)


def test_result_identity_and_completion_are_complete(config):
    cases = expected_cases(config)
    results = [{"case": case, **meta, "status": "passed"} for case, meta in cases.items()]
    assert verify_result_cases(results, config) == cases
    for bad in ([], results[:-1], results + results[:1], results + [{"case": "extra"}]):
        with pytest.raises(ValueError):
            verify_result_cases(bad, config)
    for field, value in (
        ("mode", "bogus"),
        ("variant", "bogus"),
        ("repeat", True),
        ("status", "skipped"),
    ):
        changed = copy.deepcopy(results)
        changed[0][field] = value
        with pytest.raises(ValueError):
            verify_result_cases(changed, config)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, 0),
        (-0.0, 0),
        (0.5, 500000),
        (0.1, 100001),
        (5e-324, 1),
        (10**400, 10**406),
        (sys.float_info.max, int(sys.float_info.max) * 1000000),
    ],
)
def test_fraction_oracle_extreme_numeric_contract(auditor, value, expected):
    assert auditor.threshold_oracle(value) == expected


@pytest.mark.parametrize("value", [True, False, None, "500", -1, float("nan"), float("inf")])
def test_fraction_oracle_rejects_invalid_wait(auditor, value):
    with pytest.raises(ValueError):
        auditor.threshold_oracle(value)


@pytest.mark.parametrize(
    "events",
    [
        [{"event": "call", "call": 1, "method": "free", "owner": "r"}],
        [{"event": "raise", "call": 1, "method": "free", "owner": "r", "error_type": "ValueError"}],
        [
            {"event": "call", "call": 1, "method": "free", "owner": "r"},
            {"event": "return", "call": 1, "method": "allocate_slots", "owner": "r"},
        ],
    ],
)
def test_call_completion_cannot_be_missing_or_mislabeled(auditor, events):
    with pytest.raises(ValueError):
        auditor.audit_calls(events)


def test_native_raise_is_a_real_completion_not_a_missing_return(auditor):
    events = [
        {"event": "call", "call": 1, "method": "free", "owner": "r"},
        {"event": "raise", "call": 1, "method": "free", "owner": "r", "error_type": "RuntimeError"},
    ]
    assert len(auditor.audit_calls(events)) == 1


def test_summary_only_labels_cannot_satisfy_native_matrix(auditor, config, tmp_path):
    events = []
    for case, meta in expected_cases(config).items():
        events += [
            {"event": "case_begin", "case": case, **meta, "seed": config["seed"]},
            {"event": "case_end", "case": case, **meta, "result": {"status": "passed"}},
        ]
    trace = tmp_path / "claims.jsonl"
    trace.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="native_objects"):
        auditor.assert_trace(trace, config)


@pytest.mark.parametrize("script", ["scripts/replay_m2b.py", "tests/native/m2b_trace_check.py"])
@pytest.mark.parametrize("flags,optimize", [(["-O"], None), (["-OO"], None), ([], "1"), ([], "2")])
def test_optimized_entrypoints_fail_before_import(script, flags, optimize):
    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    if optimize is not None:
        environment["PYTHONOPTIMIZE"] = optimize
    result = subprocess.run(
        [sys.executable, *flags, str(ROOT / script), "--help"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )
    assert result.returncode != 0
    assert "M2B validation requires assertions enabled" in result.stderr


def test_empty_trace_rejected_by_real_process(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/native/m2b_trace_check.py"), "--trace", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "nonempty" in result.stderr


def test_streaming_reader_rejects_interleaved_and_reopened_leaves(auditor, config, tmp_path):
    expected = expected_cases(config)
    cases = list(expected)[:2]
    records = [
        {"event": "case_begin", "case": cases[0], **expected[cases[0]]},
        {"event": "case_begin", "case": cases[1], **expected[cases[1]]},
        {"event": "case_end", "case": cases[0], **expected[cases[0]]},
    ]
    path = tmp_path / "interleaved.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in records))
    with pytest.raises(ValueError, match="duplicated or interleaved"):
        list(auditor.iter_case_groups(path, expected))


def test_compat_suite_cannot_be_used_to_shrink_frozen_config(config):
    expected = expected_cases(config, suite="compat")
    assert all(meta["scenario"] != "hot_flow" for meta in expected.values())
    assert any(meta["mode"] == "guarded" for meta in expected.values())
    config["scenarios"].pop("hot_flow")
    with pytest.raises(ValueError, match="50-combination"):
        expected_cases(config, suite="compat")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.clear(),
        lambda value: value.pop("data_parallel_size"),
        lambda value: value.update(unreviewed_parallel_size=1),
        lambda value: value.update(tensor_parallel_size=True),
        lambda value: value.update(pipeline_parallel_size=1.0),
        lambda value: value.update(prefill_context_parallel_size=2),
    ],
)
def test_native_parallel_observation_requires_exact_nonempty_fields(auditor, config, mutation):
    """A valid read-only identity record is the control for each raw-field corruption."""
    native = {
        "event": "native_objects",
        "class": auditor.legacy.NATIVE_CLASS,
        "mro": [auditor.legacy.NATIVE_CLASS, "vllm.v1.core.sched.scheduler.Scheduler"],
        "async_scheduling": True,
        "use_v2_model_runner": True,
        "log_stats": True,
        "kv_events": True,
        "kv_metrics": True,
        "apc": True,
        "block_size": 16,
        "publisher": "NullEventPublisher",
        "max_concurrent_batches": 2,
        "capacity": 10,
        "effective_config": {
            "budget": 64,
            "active": 2,
            "parallel": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "decode_context_parallel_size": 1,
                "prefill_context_parallel_size": 1,
            },
        },
        "sources": {
            auditor.legacy.NATIVE_CLASS: {
                "sha256": config["source_sha256"]["vllm/v1/core/sched/async_scheduler.py"]
            },
            "vllm.v1.core.sched.scheduler.Scheduler": {
                "sha256": config["source_sha256"]["vllm/v1/core/sched/scheduler.py"]
            },
        },
        "instance_fields": [],
    }
    metadata = {"scenario": "bc_ample", "mode": "native", "variant": "default"}
    assert auditor.audit_identity([native], metadata, config) is native
    mutation(native["effective_config"]["parallel"])
    with pytest.raises(ValueError, match="effective native parallel semantics changed"):
        auditor.audit_identity([native], metadata, config)


def rejected_attempts(state):
    """Local oracle controls only; these are not native scheduler evidence."""
    guard = {
        "present": True,
        "fault_code": "clock_type" if state == "fault" else None,
        "closed": state == "closed",
        "records": [],
    }
    return [
        {"event": "schedule_before", "step": 3, "phase": "retry", "guard": copy.deepcopy(guard)},
        {
            "event": "schedule_raise",
            "step": 3,
            "phase": "retry",
            "error_type": "WaitGuardError",
            "guard": copy.deepcopy(guard),
        },
        {
            "event": "enqueue_call",
            "step": 3,
            "request_id": "blocked",
            "object": "object:3",
            "guard": copy.deepcopy(guard),
        },
        {
            "event": "enqueue_raise",
            "step": 3,
            "request_id": "blocked",
            "object": "object:3",
            "error_type": "WaitGuardError",
            "guard": copy.deepcopy(guard),
        },
    ]


@pytest.mark.parametrize("state", ["fault", "closed"])
@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda es: es.__delitem__(slice(0, 2)), "guard retry schedule attempt missing"),
        (lambda es: es.__delitem__(slice(2, 4)), "guard retry add attempt missing"),
        (lambda es: es.insert(0, copy.deepcopy(es[0])), "guard retry schedule attempt duplicated"),
        (lambda es: es.insert(2, copy.deepcopy(es[2])), "guard retry add attempt duplicated"),
        (lambda es: es.pop(1), "guard retry schedule did not raise WaitGuardError"),
        (lambda es: es.pop(3), "guard retry add did not raise WaitGuardError"),
        (
            lambda es: es[1].update(error_type="RuntimeError"),
            "guard retry schedule did not raise WaitGuardError",
        ),
        (
            lambda es: es[3].update(error_type="RuntimeError"),
            "guard retry add did not raise WaitGuardError",
        ),
        (lambda es: es[1].update(step=2), "guard retry schedule did not raise WaitGuardError"),
        (
            lambda es: es[3].update(object="object:other"),
            "guard retry add did not raise WaitGuardError",
        ),
        (lambda es: es[3].update(step=2), "guard retry add did not raise WaitGuardError"),
        (
            lambda es: es[0]["guard"].update(fault_code=None, closed=False),
            "retry schedule did not occur after fault/close",
        ),
        (
            lambda es: es[2]["guard"].update(fault_code=None, closed=False),
            "retry add did not occur after fault/close",
        ),
        (
            lambda es: es.insert(1, {"event": "native_schedule_enter"}),
            "rejected schedule entered protected/native work",
        ),
        (
            lambda es: es.insert(3, {"event": "native_add_call"}),
            "rejected add entered protected/native work",
        ),
    ],
)
def test_rejected_attempts_need_actual_calls_and_matching_errors(auditor, state, mutate, fragment):
    events = rejected_attempts(state)
    args = {"schedule_phase": "retry", "request_id": "blocked", "state": state}
    assert auditor.audit_rejected_retries(events, **args) == {
        "schedule_attempts": 1,
        "add_attempts": 1,
    }
    mutate(events)
    with pytest.raises(ValueError, match=fragment):
        auditor.audit_rejected_retries(events, **args)


def empty_reuse_boundary():
    return {
        "event": "reuse_boundary",
        "pending_outputs": 0,
        "guard": {"present": True, "records": []},
        "state": {
            "waiting": [],
            "kv_holding_waiting": [],
            "deferred_waiting": [],
            "running": [],
            "active_requests": [],
            "requests": {},
            "free_queue": [1],
            "blocks": [{"id": 1, "null": False, "ref_cnt": 0}],
            "request_blocks": [{}],
            "full": {
                "scheduler_lifecycle": {
                    "_inflight_prefills": {"set": []},
                    "deferred_frees": [],
                    "finished_req_ids": {"set": []},
                    "reset_preempted_req_ids": {"set": []},
                }
            },
        },
    }


@pytest.mark.parametrize("field", ["finished_req_ids", "reset_preempted_req_ids"])
@pytest.mark.parametrize("value", [{"set": ["old_generation"]}, {}, [], None])
def test_reuse_requires_exact_empty_native_notifications(auditor, field, value):
    event = empty_reuse_boundary()
    auditor.audit_reuse_boundary(event)
    event["state"]["full"]["scheduler_lifecycle"][field] = value
    with pytest.raises(
        ValueError, match="ID reuse retains native terminal/preemption notifications"
    ):
        auditor.audit_reuse_boundary(event)


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda e: e.update(pending_outputs=1), "ID reuse retains pending native outputs"),
        (
            lambda e: e["guard"]["records"].append({"object": "old"}),
            "ID reuse retains protection records",
        ),
        (
            lambda e: e["state"]["full"]["scheduler_lifecycle"].update(
                _inflight_prefills={"set": ["old"]}
            ),
            "native in-flight Request references remain",
        ),
        (lambda e: e["state"]["blocks"][0].update(ref_cnt=1), "KV references remain"),
        (
            lambda e: e["state"].update(request_blocks=[{"old": []}]),
            "native request block mappings retained historical IDs",
        ),
    ],
)
def test_reuse_also_requires_native_ownership_to_be_drained(auditor, mutate, fragment):
    event = empty_reuse_boundary()
    auditor.audit_reuse_boundary(event)
    mutate(event)
    with pytest.raises(ValueError, match=fragment):
        auditor.audit_reuse_boundary(event)


@pytest.mark.parametrize("label", ["fifo_output_out_of_order", "fifo_output_duplicate"])
def test_fifo_negative_mutations_target_raw_consumption_order(auditor, negative_runner, label):
    events = [
        {"event": "schedule_after", "step": 0, "pending_outputs": 1},
        {"event": "schedule_after", "step": 1, "pending_outputs": 2},
        {"event": "output_after", "output_step": 0, "pending_outputs": 1},
        {"event": "output_after", "output_step": 1, "pending_outputs": 0},
        {"event": "drained", "pending_outputs": 0},
    ]
    for event in events:
        event.update(scenario="bc_pressure", mode="native")
    assert auditor.legacy.audit_fifo(events) == 2
    _, predicate, mutate = next(
        definition
        for definition in negative_runner.primitive_definitions()
        if definition[0] == label
    )
    index = next(i for i, event in enumerate(events) if predicate(event))
    mutate(events, index)
    with pytest.raises(ValueError, match=negative_runner.PRIMITIVE_ERRORS[label]):
        auditor.legacy.audit_fifo(events)


def test_resource_negative_mutation_targets_final_nonnull_block(auditor, negative_runner):
    event = empty_reuse_boundary()
    event["event"] = "fixture_drained"
    event["state"]["blocks"].insert(0, {"id": 0, "null": True, "ref_cnt": 1})
    auditor.audit_resources(event["state"])
    label = "native_kv_reference_retained"
    _, predicate, mutate = next(
        definition
        for definition in negative_runner.primitive_definitions()
        if definition[0] == label
    )
    assert predicate(event)
    mutate([event], 0)
    assert event["state"]["blocks"][0]["ref_cnt"] == 1
    assert event["state"]["blocks"][1]["ref_cnt"] == 1
    with pytest.raises(ValueError, match=negative_runner.PRIMITIVE_ERRORS[label]):
        auditor.audit_resources(event["state"])


@pytest.mark.parametrize(
    "operation",
    [
        "missing_final_case",
        "missing_final_repeat",
        "duplicate_reopened_case",
        "extra_case",
        "mislabeled_case",
        "skipped_event",
    ],
)
def test_full_trace_mutators_reach_case_stream_gate(auditor, negative_runner, tmp_path, operation):
    """Local stream-format controls; complete native leaf semantics run on Linux."""
    expected = {
        f"{repeat}:local:guarded:{variant}": {
            "repeat": repeat,
            "scenario": "local",
            "mode": "guarded",
            "variant": variant,
        }
        for repeat in range(3)
        for variant in ("first", "last")
    }
    events = [
        {"case": case, **metadata, "event": event}
        for case, metadata in expected.items()
        for event in ("case_begin", "case_end")
    ]
    source = tmp_path / "original.jsonl"
    source.write_text("".join(json.dumps(event) + "\n" for event in events))
    original = source.read_bytes()
    assert len(list(auditor.iter_case_groups(source, expected))) == 6
    output = tmp_path / "mutated.jsonl"
    observed = negative_runner.write_mutated_trace(source, output, operation, expected)
    assert source.read_bytes() == original
    assert observed["base_sha256"] == hashlib.sha256(original).hexdigest()
    if operation == "missing_final_case":
        assert observed["removed_records"] == 2
        assert observed["affected_cases"] == [list(expected)[-1]]
    elif operation == "missing_final_repeat":
        assert observed["removed_records"] == 4
        assert len(observed["affected_cases"]) == 2
    else:
        assert observed["changed_records"] == 1
    with pytest.raises(ValueError, match=negative_runner.RAW_TRACE_ERRORS[operation]):
        list(auditor.iter_case_groups(output, expected))
