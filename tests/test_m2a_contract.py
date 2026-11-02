"""Local gate regressions; these are not native scheduler evidence."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m2a_contract import (
    SCENARIOS,
    expected_cases,
    verify_result_cases,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/native-m2a.json").read_text())


def test_complete_frozen_matrix_is_nonempty(config):
    cases = expected_cases(config)
    assert len(cases) == 72
    assert len(SCENARIOS) == 11
    assert set(cases) == {
        f"{repeat}:{scenario}:{mode}"
        for repeat in range(3)
        for scenario, modes in SCENARIOS.items()
        for mode in modes
    }


@pytest.mark.parametrize("repeats", [None, False, True, -1, 0, 1, 2, 3.0, "3"])
def test_repeats_fail_closed(config, repeats):
    config["repeats"] = repeats
    with pytest.raises(ValueError, match="repeats"):
        expected_cases(config)


@pytest.mark.parametrize("scenarios", [{}, [], None, {"bc_pressure": ["native"]}])
def test_incomplete_scenario_catalog_fails(config, scenarios):
    config["scenarios"] = scenarios
    with pytest.raises(ValueError, match="scenario"):
        expected_cases(config)


@pytest.mark.parametrize(
    "modes", [[], ["native"], ["native", "off", "off"], ["native", "off", "guarded"], "native"]
)
def test_missing_duplicate_or_extra_mode_fails(config, modes):
    config["scenarios"]["bc_pressure"] = modes
    with pytest.raises(ValueError, match="scenario"):
        expected_cases(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("block_size", True),
        ("seed", 43),
        ("window", 1),
        ("pending_output_depth", 0),
        ("watermark", True),
        ("enable_chunked_prefill", 1),
        ("vllm_commit", "main"),
    ],
)
def test_baseline_drift_is_rejected(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        expected_cases(config)


def test_input_and_source_manifest_are_required(config):
    changed = copy.deepcopy(config)
    changed["requests"][2]["tokens"][0] += 1
    with pytest.raises(ValueError, match="input drift"):
        expected_cases(changed)
    del config["source_sha256"]["vllm/v1/core/kv_cache_metrics.py"]
    with pytest.raises(ValueError, match="manifest"):
        expected_cases(config)


def test_result_identity_requires_exact_complete_set(config):
    expected = expected_cases(config)
    results = [
        {"case": case, **metadata, "status": "passed"} for case, metadata in expected.items()
    ]
    assert verify_result_cases(results, config) == expected
    for bad in ([], results[:-1], [*results, results[0]], [*results, {"case": "extra"}]):
        with pytest.raises(ValueError):
            verify_result_cases(bad, config)
    for key, value in [
        ("status", "skipped"),
        ("repeat", True),
        ("mode", "guarded"),
        ("scenario", "wrong"),
    ]:
        changed = copy.deepcopy(results)
        changed[0][key] = value
        with pytest.raises(ValueError):
            verify_result_cases(changed, config)


@pytest.mark.parametrize("script", ["scripts/replay_m2a.py", "tests/native/m2a_trace_check.py"])
@pytest.mark.parametrize("flags,optimize", [(["-O"], None), (["-OO"], None), ([], "1"), ([], "2")])
def test_optimized_entrypoint_fails_before_native_import(script, flags, optimize):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    if optimize is not None:
        env["PYTHONOPTIMIZE"] = optimize
    result = subprocess.run(
        [sys.executable, *flags, str(ROOT / script), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert result.returncode != 0
    assert "M2A validation requires assertions enabled" in result.stderr


def test_empty_trace_and_invalid_config_never_pass(tmp_path, config):
    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    config_path = tmp_path / "config.json"
    for repeats in (3, 0, 1):
        config["repeats"] = repeats
        config_path.write_text(json.dumps(config))
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/native/m2a_trace_check.py"),
                "--trace",
                str(trace),
                "--config",
                str(config_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0
        assert 'verified": true' not in result.stdout


@pytest.fixture(scope="module")
def auditor():
    spec = importlib.util.spec_from_file_location(
        "m2a_audit_gate_tests", ROOT / "tests/native/m2a_trace_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clock_normalization_preserves_aliases_and_nonclock_content(auditor):
    def event(lifetime, aliases):
        return {
            "type": "vllm.v1.metrics.stats.KVCacheEvictionEvent",
            "fields": {
                "lifetime_seconds": lifetime,
                "idle_seconds": lifetime,
                "reuse_gaps_seconds": {"ref": aliases},
            },
        }

    assert auditor.normalized(event(1.5, 12)) == auditor.normalized(event(2.5, 12))
    assert auditor.normalized(event(1.5, 12)) != auditor.normalized(event(1.5, 13))
    assert auditor.normalized({"timestamp": 1}) != auditor.normalized({"timestamp": 2})
    with pytest.raises(ValueError, match="alias"):
        auditor.normalized(event(1.5, -1))
    enum_field = {"type": {"enum": "vllm.v1.engine.EngineCoreEventType", "name": "QUEUED"}}
    assert auditor.normalized(enum_field) == enum_field


def test_identity_normalization_preserves_alias_relationships(auditor):
    first = {"identities": {"request": 10, "view": 10}}
    same = {"identities": {"request": 20, "view": 20}}
    changed = {"identities": {"request": 20, "view": 30}}
    assert auditor.normalized(first) == auditor.normalized(same)
    assert auditor.normalized(first) != auditor.normalized(changed)


@pytest.mark.parametrize(
    "events",
    [
        [{"event": "output_after", "output_step": 1, "pending_outputs": 0}],
        [{"event": "schedule_after", "step": 1, "pending_outputs": 1}],
        [
            {"event": "schedule_after", "step": 1, "pending_outputs": 1},
            {"event": "output_after", "output_step": 2, "pending_outputs": 0},
        ],
    ],
)
def test_fifo_gate_rejects_orphan_pending_and_reordered_outputs(auditor, events):
    with pytest.raises(ValueError):
        auditor.audit_fifo(events)


def test_case_labels_and_success_flags_cannot_replace_native_trace(tmp_path, config, auditor):
    events = []
    for case, metadata in expected_cases(config).items():
        events.append({"case": case, **metadata, "event": "case_begin"})
        events.append(
            {"case": case, **metadata, "event": "case_end", "result": {"status": "passed"}}
        )
    trace = tmp_path / "claims-only.jsonl"
    trace.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="native_objects"):
        auditor.assert_trace(trace, config)
