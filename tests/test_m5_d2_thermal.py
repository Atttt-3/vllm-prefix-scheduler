import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SECOND = 1_000_000_000


@pytest.fixture
def thermal(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import m5_d2_thermal

    return m5_d2_thermal


@pytest.fixture
def limits():
    return {
        "minimum_conditioning_s": 30,
        "read_only_side_sampler": {"max_gap_s": 1.5},
        "pilot_plateau": {
            "window_s": 20,
            "min_total_samples": 18,
            "min_active_samples_per_half": 6,
            "max_temp_span_c": 2,
            "max_sm_median_relative_change_between_halves": 0.10,
            "min_active_memory_clock_mode_share": 0.90,
        },
        "per_measured_engine": {
            "preheat_temp_median_from_target_c": 2,
            "preheat_active_sm_median_relative_from_target": 0.10,
            "preheat_active_memory_mode_must_equal_target": True,
            "max_age_of_actual_pre_t0_device_sample_s": 1.5,
            "actual_pre_t0_temp_from_target_c": 2,
        },
    }


def sample(second, **changes):
    return {
        "monotonic_ns": second * SECOND,
        "gpu_uuid": "GPU-d2",
        "temperature_c": 50,
        "sm_clock_mhz": 1500,
        "memory_clock_mhz": 9501,
        "gpu_utilization_percent": 80,
        **changes,
    }


def test_first_eligible_window_and_frozen_target(thermal, limits):
    samples = [sample(i) for i in range(1, 61)]
    selected = thermal.first_plateau(samples, SECOND, limits)
    assert (selected["start_ns"], selected["end_ns"], selected["sample_count"]) == (
        11 * SECOND,
        31 * SECOND,
        21,
    )
    target = thermal.target_from_plateau(selected)
    assert target == {"temperature_c": 50, "sm_clock_mhz": 1500, "memory_clock_mhz": 9501}
    assert thermal.first_plateau(samples, SECOND, limits, target) == selected
    for change in (
        {"temperature_c": 47.99},
        {"sm_clock_mhz": 1700},
        {"memory_clock_mhz": 5001},
    ):
        assert thermal.first_plateau(samples, SECOND, limits, target | change) is None


@pytest.mark.parametrize(
    ("sample_changes", "drop"),
    [
        ({29: {"temperature_c": 53}}, set()),
        ({i: {"gpu_utilization_percent": 0} for i in range(11, 18)}, set()),
        ({i: {"sm_clock_mhz": 1200} for i in range(11, 21)}, set()),
        ({i: {"memory_clock_mhz": 5001} for i in (12, 17, 20)}, set()),
        ({}, {25}),
    ],
)
def test_invalid_thermal_windows(thermal, limits, sample_changes, drop):
    samples = [sample(i, **sample_changes.get(i, {})) for i in range(1, 32) if i not in drop]
    assert thermal.first_plateau(samples, SECOND, limits) is None


def test_retains_only_completed_sample_records(thermal, tmp_path):
    path = tmp_path / "device.jsonl"
    one = sample(1)
    path.write_text(json.dumps(one) + "\n" + '{"monotonic_ns": 2000')
    assert thermal.read_samples(path, 2 * SECOND) == [one]
    assert thermal.read_samples(path, SECOND - 1) == []
    with path.open("a") as stream:
        stream.write("}\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        thermal.read_samples(path, 2 * SECOND)


def test_pre_t0_uses_actual_measure_start(thermal, limits):
    gate = limits["per_measured_engine"]
    target = {"temperature_c": 50}
    assert thermal.pre_t0_admitted(sample(10, temperature_c=52), 11_500_000_000, target, gate)
    assert not thermal.pre_t0_admitted(sample(10), 11_500_000_001, target, gate)
    assert not thermal.pre_t0_admitted(sample(10, temperature_c=52.1), 11 * SECOND, target, gate)
    assert not thermal.pre_t0_admitted(sample(10), 9 * SECOND, target, gate)
    assert not thermal.pre_t0_admitted(None, 11 * SECOND, target, gate)
