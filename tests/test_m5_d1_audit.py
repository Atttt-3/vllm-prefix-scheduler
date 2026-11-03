"""Exercise the D1 independent decision boundaries."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATES = {"overhead_ttft_ns": 2_000_000, "overhead_ttft_rel": 0.05, "overhead_throughput_rel": 0.05}


@pytest.fixture
def d1(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("check_m5_d1")


def metrics(ttft, requests=100, tokens=1000):
    return {
        "strata": {"all": {"ttft_ns": {"p95": ttft}}},
        "throughput": {
            "last_terminal_requests_per_s": requests,
            "last_terminal_tokens_per_s": tokens,
        },
    }


@pytest.mark.parametrize(
    ("reference", "observed", "expected"),
    [
        (100_000_000, 95_000_000, True),
        (100_000_000, 94_999_999, False),
        (100_000_000, 105_000_000, True),
        (100_000_000, 105_000_001, False),
        (10_000_000, 12_000_000, True),
        (10_000_000, 12_000_001, False),
    ],
)
def test_ttft_cutoff_is_inclusive_in_both_directions(d1, reference, observed, expected):
    result = d1.compare_pair(metrics(reference), metrics(observed), GATES)
    assert result["ttft_delta_ns"] == observed - reference
    assert result["ttft_within_tolerance"] is expected
    assert result["within_tolerance"] is expected


def test_both_throughput_quantities_are_required(d1):
    reference = metrics(100_000_000)
    result = d1.compare_pair(reference, metrics(100_000_000, requests=104, tokens=1051), GATES)
    assert result["ttft_within_tolerance"]
    assert not result["throughput_within_tolerance"]
    assert not result["within_tolerance"]


def test_decision_distinguishes_same_tier_instability_and_consistent_cross_tier(d1):
    reference = metrics(100_000_000)
    stable = d1.compare_pair(reference, reference, GATES)
    slower = d1.compare_pair(reference, metrics(106_000_000), GATES)
    faster = d1.compare_pair(reference, metrics(94_000_000), GATES)
    passing = {"E/E": [stable] * 2, "L/L": [stable] * 2, "E/L": [stable] * 4}
    assert d1.qualification_decision(passing)["enter_qualification"]
    blocked = {**passing, "E/L": [slower] * 4}
    assert d1.qualification_decision(blocked)["findings"] == ["repeatable_tier_ttft_difference"]
    unstable = {**blocked, "E/E": [stable, faster], "L/L": [slower, stable]}
    assert d1.qualification_decision(unstable)["findings"] == [
        "endpoint_repeat_instability",
        "light_repeat_instability",
    ]
    mixed = {**passing, "E/L": [slower, faster] * 2}
    assert d1.qualification_decision(mixed)["findings"] == ["cross_tier_difference_not_resolved"]
    with pytest.raises(ValueError, match="incomplete D1 pair inventory"):
        d1.qualification_decision({**passing, "E/L": [stable] * 3})


def test_telemetry_requires_device_and_affinity_binding(d1, tmp_path):
    path = tmp_path / "device.json"
    snapshot = {
        "monotonic_ns": 31,
        "cpu_affinity": [2, 3],
        "gpu": {"returncode": 0, "stdout": "GPU-aaa, 210, 405, 38, [N/A]\n", "stderr": ""},
    }
    path.write_text(json.dumps(snapshot))
    result = d1.telemetry(path, "GPU-aaa")
    assert result["sm_clock_mhz"] == 210
    assert result["power_w"] is None
    with pytest.raises(ValueError, match="device identity"):
        d1.telemetry(path, "GPU-bbb")
    snapshot["cpu_affinity"].reverse()
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="CPU affinity"):
        d1.telemetry(path, "GPU-aaa")
