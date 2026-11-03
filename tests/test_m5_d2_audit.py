"""Exercise D2's independent thermal and performance evidence boundaries."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NS = 1_000_000_000
LIMITS = {
    "minimum_conditioning_s": 30,
    "pilot_cap_s": 150,
    "per_measured_engine_cap_s": 75,
    "read_only_side_sampler": {"max_gap_s": 1.5},
    "pilot_plateau": {
        "window_s": 20,
        "min_total_samples": 18,
        "min_active_samples_per_half": 6,
        "max_temp_span_c": 2,
        "max_sm_median_relative_change_between_halves": 0.1,
        "min_active_memory_clock_mode_share": 0.9,
    },
    "per_measured_engine": {
        "preheat_temp_median_from_target_c": 2,
        "preheat_active_sm_median_relative_from_target": 0.1,
        "actual_pre_t0_temp_from_target_c": 2,
        "preheat_active_memory_mode_must_equal_target": True,
        "max_age_of_actual_pre_t0_device_sample_s": 1.5,
    },
    "in_window_comparability": {
        "min_total_samples_per_round": 6,
        "min_active_samples_per_round": 4,
        "max_temp_median_difference_per_pair_c": 2,
        "max_active_sm_median_relative_difference_per_pair": 0.1,
        "same_active_memory_clock_mode_per_pair": True,
    },
}


@pytest.fixture
def d2(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("check_m5_d2")


def sample(seconds, *, temp=50, sm=1000, memory=5000, util=80):
    return {
        "monotonic_ns": int(seconds * NS),
        "temperature_c": temp,
        "sm_clock_mhz": sm,
        "memory_clock_mhz": memory,
        "gpu_utilization_percent": util,
        "gpu_uuid": "GPU-test",
    }


def test_first_plateau_obeys_minimum_and_frozen_target(d2):
    samples = [sample(second) for second in range(1, 40)]
    assert d2.thermal_window(samples, NS, 30 * NS, LIMITS) is None
    selected = d2.thermal_window(samples, NS, 39 * NS, LIMITS)
    assert selected["end_ns"] == 31 * NS
    assert selected["sample_count"] == 21
    assert (
        d2.thermal_window(
            samples,
            NS,
            39 * NS,
            LIMITS,
            {"temperature_c": 53, "sm_clock_mhz": 1000, "memory_clock_mhz": 5000},
        )
        is None
    )


def test_gaps_activity_and_clock_drift_independently_reject_plateau(d2):
    samples = [sample(second) for second in range(1, 32)]
    assert d2.thermal_window(samples[:20] + samples[21:], NS, 31 * NS, LIMITS) is None
    active_sparse = [sample(second, util=int(second > 26)) for second in range(1, 32)]
    assert d2.thermal_window(active_sparse, NS, 31 * NS, LIMITS) is None
    changed_sm = [sample(second, sm=1000 if second < 21 else 1200) for second in range(1, 32)]
    assert d2.thermal_window(changed_sm, NS, 31 * NS, LIMITS) is None


def test_pre_t0_requires_journal_row_and_tolerates_only_one_unflushed_sample(d2):
    target = {"temperature_c": 50}
    chosen = sample(10)
    phases = {"warmup_end_ns": 11 * NS, "measure_start_ns": int(11.3 * NS)}
    raw = {
        "conditioning_end_ns": 9 * NS,
        "pre_t0": {
            "sample": chosen,
            "decision_ns": int(11.1 * NS),
            "measure_start_ns": int(11.3 * NS),
        },
    }
    assert d2.pre_t0_ok(raw, [chosen, sample(11)], phases, target, LIMITS)
    assert not d2.pre_t0_ok(raw, [sample(10, temp=53), sample(11)], phases, target, LIMITS)
    assert not d2.pre_t0_ok(raw, [chosen, sample(10.5), sample(11)], phases, target, LIMITS)
    assert not d2.pre_t0_ok(raw, [chosen], phases, {"temperature_c": 53}, LIMITS)


def test_window_comparison_checks_boundaries_and_uses_active_clocks(d2):
    first = d2.window_profile([sample(s) for s in range(10, 18)], 10 * NS, 17 * NS, LIMITS)
    missing_start = d2.window_profile([sample(s) for s in range(12, 18)], 10 * NS, 17 * NS, LIMITS)
    assert first["complete"]
    assert not missing_start["complete"]
    exact = d2.window_profile(
        [sample(s, temp=52, sm=1100) for s in range(20, 28)], 20 * NS, 27 * NS, LIMITS
    )
    outside = {**exact, "sm_median_mhz": 1101}
    assert d2.compare_thermal(first, exact, LIMITS)["comparable"]
    assert not d2.compare_thermal(first, outside, LIMITS)["comparable"]
    assert not d2.compare_thermal(first, missing_start, LIMITS)["comparable"]
