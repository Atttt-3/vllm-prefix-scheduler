"""Synthetic contract fixtures, never inference or GPU acceptance evidence."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m3_contract import (
    audit_batches,
    audit_drained,
    canonical,
    case_catalog,
    expected_cases,
    read_events,
    validate_case_identity,
    validate_envelopes,
)

ROOT = Path(__file__).resolve().parents[1]


def envelope(event, seq, **fields):
    return {
        "schema": 1,
        "event": event,
        "pid": 10,
        "seq": seq,
        "monotonic_ns": seq * 10,
        "clock_domain": "monotonic_ns:contract-fixture",
        "engine_id": "diagnostic-prefix",
        "mode": "prefix",
        "phase": "diagnostic",
        **fields,
    }


def batch_fixture():
    rid = "mechanism:prefix:0::C"
    descriptor = {
        "num_scheduled_tokens": {rid: 32},
        "total_num_scheduled_tokens": 32,
        "new_req_ids": [rid],
        "cached_req_ids": [],
        "finished_req_ids": [],
    }
    descriptor["output_sha256"] = hashlib.sha256(canonical(descriptor).encode()).hexdigest()
    common = {"batch_id": "10:1"}
    output = {
        "output_type": "vllm.v1.outputs.ModelRunnerOutput",
        "req_ids": [rid],
        "sampled_token_ids": [[7]],
    }
    return [
        envelope(
            "schedule_success",
            1,
            **common,
            **descriptor,
            intervals=[
                {
                    "request_id": rid,
                    "object_id": "fixture:1",
                    "start": 0,
                    "end": 32,
                    "prompt_length": 32,
                }
            ],
        ),
        envelope("runner_execute_start", 2, **common, **descriptor),
        envelope(
            "runner_execute_success",
            3,
            **common,
            result_kind="none",
            forward_observed=True,
            forward_tokens=32,
            forward_tokens_padded=32,
            forward_req_ids=[rid],
        ),
        envelope("runner_sample_start", 4, **common),
        envelope("runner_sample_success", 5, **common, result_kind="async"),
        envelope("model_output_materialized", 6, **common, **output),
        envelope("update_from_output_start", 7, **common, **output),
        envelope("update_from_output_success", 8, **common, **output),
    ]


def drain_fixture():
    from vllm_prefix_scheduler.m3_contract import EMPTY_LIST_RESOURCES, ZERO_RESOURCES

    return {
        **{name: [] for name in EMPTY_LIST_RESOURCES},
        **{name: 0 for name in ZERO_RESOURCES},
        "total_non_null_blocks": 100,
        "free_count": 100,
        "free_unique_count": 100,
        "free_traversed_count": 100,
        "guard_fault_code": None,
        "cached_block_count": 20,
    }


def test_exact_matrix_and_frozen_config():
    config = json.loads((ROOT / "configs/m3-real-model.json").read_text())
    expected = expected_cases(config)
    assert len(expected) == 20
    assert len([k for k in expected if k.startswith("smoke:")]) == 4
    assert len([k for k in expected if k.startswith("mechanism:")]) == 15
    for mutation in (1, True, 0, 4):
        bad = copy.deepcopy(config)
        bad["repeats"] = mutation
        with pytest.raises(ValueError, match="exactly three"):
            expected_cases(bad)
    bad = copy.deepcopy(config)
    bad["functional"]["async_scheduling"] = False
    with pytest.raises(ValueError, match="only change max_num_seqs"):
        expected_cases(bad)


def test_matrix_boundary_identity_cannot_be_empty_missing_or_mislabeled():
    config = json.loads((ROOT / "configs/m3-real-model.json").read_text())
    events = []
    for case, metadata in case_catalog().items():
        events.extend(
            [
                envelope("case_begin", len(events) + 1, case_id=case, **metadata),
                envelope("case_end", len(events) + 2, case_id=case, status="passed", **metadata),
            ]
        )
    assert len(validate_case_identity(events, config)) == 20
    for mutated in ([], events[:-1], events + [events[0]]):
        with pytest.raises(ValueError, match="entire unique"):
            validate_case_identity(mutated, config)
    events[0]["variant"] = "mislabeled"
    with pytest.raises(ValueError, match="metadata mislabeled"):
        validate_case_identity(events, config)


def test_batch_fixture_is_only_contract_positive():
    events = batch_fixture()
    validate_envelopes(events)
    result = audit_batches(events)
    assert result["batches"] == result["positive_batches"] == 1
    assert result["runner_forward_tokens"] == 32


@pytest.mark.parametrize(
    "kind",
    [
        "runner_execute_start",
        "runner_execute_success",
        "runner_sample_start",
        "runner_sample_success",
        "model_output_materialized",
        "update_from_output_start",
        "update_from_output_success",
    ],
)
def test_batch_chain_requires_each_real_stage(kind):
    with pytest.raises(ValueError, match=f"exactly one {kind}"):
        audit_batches([e for e in batch_fixture() if e["event"] != kind])


@pytest.mark.parametrize(
    "kind,field,value,error",
    [
        ("runner_execute_start", "output_sha256", "0" * 64, "runner descriptor"),
        ("runner_execute_success", "forward_observed", False, "real execute/forward"),
        ("runner_execute_success", "forward_tokens", 31, "forward token count"),
        ("runner_sample_success", "result_kind", "none", "async sampling"),
        ("model_output_materialized", "req_ids", [], "consumed output differs"),
        ("update_from_output_success", "sampled_token_ids", [[8]], "consumed output differs"),
        ("update_from_output_start", "batch_id", "wrong", "exactly one update_from_output_start"),
    ],
)
def test_batch_chain_rejects_forged_links(kind, field, value, error):
    events = batch_fixture()
    next(e for e in events if e["event"] == kind)[field] = value
    with pytest.raises(ValueError, match=error):
        audit_batches(events)


def test_resource_drain_keeps_apc_hashes_but_rejects_active_references():
    resources = drain_fixture()
    audit_drained(resources)
    resources["non_null_ref_sum"] = 1
    with pytest.raises(ValueError, match="non_null_ref_sum"):
        audit_drained(resources)
    resources = drain_fixture()
    resources["free_unique_count"] -= 1
    with pytest.raises(ValueError, match="free queue"):
        audit_drained(resources)


def test_trace_file_preferred_without_recursive_duplicates(tmp_path):
    source = tmp_path / "trace.jsonl"
    source.write_text("".join(canonical(e) + "\n" for e in batch_fixture()))
    nested = tmp_path / "engines"
    nested.mkdir()
    (nested / "copy.jsonl").write_text(source.read_text())
    (tmp_path / "gpu-samples.jsonl").write_text('{"not":"event"}\n')
    assert len(read_events(tmp_path)) == len(read_events(source)) == 8


def test_real_subprocess_contract_positive_and_negatives(tmp_path):
    """Fresh interpreters audit a labeled synthetic batch; no GPU claim."""
    program = "from vllm_prefix_scheduler.m3_contract import require_assertions,audit_batches; import json,sys; require_assertions(); audit_batches(json.load(open(sys.argv[1])))"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("PYTHONOPTIMIZE", None)
    for name in ("positive", "missing_sample", "bad_forward", "optimized"):
        events = batch_fixture()
        if name == "missing_sample":
            events = [e for e in events if e["event"] != "runner_sample_success"]
        if name == "bad_forward":
            events[2]["forward_observed"] = False
        payload = tmp_path / f"{name}.json"
        payload.write_text(json.dumps(events))
        command = [
            sys.executable,
            *(["-O"] if name == "optimized" else []),
            "-c",
            program,
            str(payload),
        ]
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=10)
        if name == "positive":
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0
            expected = {
                "missing_sample": "exactly one runner_sample_success",
                "bad_forward": "real execute/forward",
                "optimized": "assertions enabled",
            }[name]
            assert expected in result.stderr
