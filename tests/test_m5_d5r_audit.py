"""Independent D5-R thermal selection and fail-closed allocation controls."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIMITS = json.loads((ROOT / "configs/m5-d3.json").read_text())["conditioning"]
BAND = (64, 66)
NS = 1_000_000_000
WORKLOAD = {"requests": [{"request_id": "r0", "offset_ns": 0}], "sampling": {"max_tokens": 1}}


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("check_m5_d2")


def telemetry():
    return [
        {
            "monotonic_ns": second * NS,
            "gpu_uuid": "GPU-test",
            "temperature_c": 62 if second < 46 else 65,
            "sm_clock_mhz": 1875,
            "memory_clock_mhz": 9501,
            "gpu_utilization_percent": 90,
        }
        for second in range(1, 100)
    ]


def raw_pilot(*, observed=66, reset=66, plateau=None):
    reset_ns = reset * NS
    drain_ns = reset_ns - 1
    return {
        "role": "pilot",
        "schema_version": 2,
        "status": "target-frozen" if plateau else "invalid",
        "reason": None,
        "conditioning_start_ns": NS,
        "conditioning_cutoff_ns": 151 * NS,
        "conditioning_end_ns": reset_ns + 2,
        "plateau": plateau,
        "waves": [
            {
                "wave": 1,
                "started_ns": 2 * NS,
                "drained_at_ns": drain_ns,
                "reset_after_ns": reset_ns,
                "sample_observed_through_ns": observed * NS,
                "plateau_decision_ns": reset_ns + 1,
                "requests": [
                    {
                        "request_id": "d2-preheat-1-r0",
                        "planned_ns": 2 * NS,
                        "submit_ns": 2 * NS,
                        "first_token_ns": 3 * NS,
                        "end_ns": 4 * NS,
                        "status": "completed",
                        "token_ids": [1],
                    }
                ],
            }
        ],
        "preheat_boundaries": [
            {
                "label": "d2-preheat-1",
                "at_ns": drain_ns,
                "runner_request_ids": [],
                "runner_pending": False,
            }
        ],
    }


def test_d5r_independent_pilot_chooses_earliest_in_band_window(audit):
    samples = telemetry()
    legacy = audit.thermal_window(samples, NS, 90 * NS, LIMITS)
    selected = audit.thermal_window(samples, NS, 90 * NS, LIMITS, pilot_temperature_range=BAND)
    assert (legacy["end_ns"], legacy["temperature_c"]) == (31 * NS, 62)
    assert (selected["end_ns"], selected["temperature_c"]) == (66 * NS, 65)
    result = audit.prelude_summary(
        raw_pilot(plateau=selected), WORKLOAD, samples, LIMITS, None, pilot_temperature_range=BAND
    )
    assert result["verified"]
    assert result["conditioned_after_ns"] == 66 * NS + 1


@pytest.mark.parametrize("temperature", [64, 66])
def test_d5r_pilot_band_bounds_are_inclusive(audit, temperature):
    samples = [{**row, "temperature_c": temperature} for row in telemetry()]
    selected = audit.thermal_window(samples, NS, 60 * NS, LIMITS, pilot_temperature_range=BAND)
    assert selected["end_ns"] == 31 * NS
    assert selected["temperature_c"] == temperature


def test_d5r_does_not_admit_samples_after_exact_observed_cutoff(audit):
    result = audit.prelude_summary(
        raw_pilot(observed=65), WORKLOAD, telemetry(), LIMITS, None, pilot_temperature_range=BAND
    )
    assert result["selected_plateau"] is None
    assert result["issues"] == ["no_qualifying_prelude_window"]


def test_d5r_rejects_a_self_consistent_legacy_out_of_band_pilot(audit):
    samples = telemetry()
    legacy = audit.thermal_window(samples, NS, 65 * NS, LIMITS)
    raw = raw_pilot(observed=65, plateau=legacy)
    assert audit.prelude_summary(raw, WORKLOAD, samples, LIMITS, None)["verified"]
    result = audit.prelude_summary(
        raw, WORKLOAD, samples, LIMITS, None, pilot_temperature_range=BAND
    )
    assert not result["verified"]
    assert "reported_plateau_differs_from_recomputation" in result["issues"]


def test_d5r_rejects_a_false_invalid_status_when_complete_checkpoint_qualifies(audit):
    samples = telemetry()
    raw = {**raw_pilot(), "target": None, "target_gpu_uuid": None, "pre_t0": None}
    result = audit.prelude_summary(
        raw, WORKLOAD, samples, LIMITS, None, pilot_temperature_range=BAND
    )
    assert result["selected_plateau"]["temperature_c"] == 65
    with pytest.raises(ValueError, match="differs from independent in-band selection"):
        audit.verified_pilot_target(raw, result, strict=True)

    raw["waves"][0]["sample_observed_through_ns"] = 65 * NS
    result = audit.prelude_summary(
        raw, WORKLOAD, samples, LIMITS, None, pilot_temperature_range=BAND
    )
    assert audit.verified_pilot_target(raw, result, strict=True) is None


@pytest.mark.parametrize(
    ("mutation", "issue"),
    [
        ("unobserved", "preheat_sample_visibility_boundary_invalid"),
        ("incomplete", "preheat_wave_incomplete"),
        ("after_deadline", "preheat_plateau_decision_outside_conditioning_deadline"),
    ],
)
def test_d5r_requires_complete_wave_and_bound_decision(audit, mutation, issue):
    raw = raw_pilot(reset=151 if mutation == "after_deadline" else 66)
    wave = raw["waves"][0]
    if mutation == "unobserved":
        wave["sample_observed_through_ns"] -= 1
    elif mutation == "incomplete":
        wave["requests"][0]["status"] = "timeout"
    result = audit.prelude_summary(
        raw, WORKLOAD, telemetry(), LIMITS, None, pilot_temperature_range=BAND
    )
    assert not result["verified"]
    assert issue in result["issues"]


def test_d5r_pilot_filter_cannot_be_applied_to_measured_leaves(audit):
    raw = {**raw_pilot(), "role": "measure"}
    with pytest.raises(ValueError, match="only to the independent pilot"):
        audit.prelude_summary(
            raw, WORKLOAD, telemetry(), LIMITS, None, pilot_temperature_range=BAND
        )


def test_d5r_completed_receipt_requires_pilot_and_stops_after_invalid_leaf(audit):
    with pytest.raises(ValueError, match="lacks an independently reviewable pilot"):
        audit.verify_thermal_run_order([], prospective=True, require_reviewable_pilot=True)
    leaves = [
        {"id": name, "returncode": 0, "audit_sha256": "a" * 64, "d2_thermal_status": status}
        for name, status in (
            ("thermal-pilot", "target-frozen"),
            ("E1", "invalid"),
            ("E2", "admitted"),
        )
    ]
    audit.verify_thermal_run_order(leaves[:2], prospective=True, require_reviewable_pilot=True)
    with pytest.raises(ValueError, match="continued after an invalid"):
        audit.verify_thermal_run_order(leaves, prospective=True, require_reviewable_pilot=True)
