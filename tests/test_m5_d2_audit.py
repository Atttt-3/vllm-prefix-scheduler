"""Exercise D2's independent thermal and performance evidence boundaries."""

import importlib
import json
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_d2 import d2r110_plan, d2r_plan

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


def test_d2r_plan_retains_original_thermal_and_performance_thresholds():
    plan = json.loads((ROOT / "configs/m5-d2r.json").read_text())
    assert d2r_plan(plan) is plan
    plan["stability_gate"]["ttft_p95_relative_margin"] = 0.06
    with pytest.raises(
        ValueError, match="original input, thermal checks, stability gate or budget"
    ):
        d2r_plan(plan)


def test_d2r110_plan_retains_original_thermal_performance_and_shadow():
    plan = json.loads((ROOT / "configs/m5-d2r110.json").read_text())
    assert d2r110_plan(plan) is plan
    with pytest.raises(ValueError, match="cap, shadow or budget differs"):
        d2r110_plan({**plan, "shadow_conditioning_cap_s": 96})
    with pytest.raises(ValueError, match="retained D2-R analysis differs"):
        d2r110_plan({**plan, "retained_D2R": {**plan["retained_D2R"], "analysis_sha256": "b" * 64}})
    plan["conditioning"]["pilot_plateau"]["max_temp_span_c"] += 1
    with pytest.raises(
        ValueError, match="original input, thermal checks, stability gate or budget"
    ):
        d2r110_plan(plan)


def d2r_prelude(d2):
    limits = {**LIMITS, "per_measured_engine_cap_s": 95}
    target = {"temperature_c": 50, "sm_clock_mhz": 1000, "memory_clock_mhz": 5000}
    samples = [sample(i, temp=54 if i < 75 else 50) for i in range(1, 97)]
    workload = {"requests": [{"request_id": "r0", "offset_ns": 0}], "sampling": {"max_tokens": 1}}
    raw = {
        "schema_version": 2,
        "role": "measure",
        "status": "admitted",
        "reason": None,
        "conditioning_start_ns": NS,
        "conditioning_cutoff_ns": 96 * NS,
        "conditioning_end_ns": 96 * NS + 100,
        "plateau": d2.thermal_window(samples, NS, 95 * NS, limits, target),
        "preheat_boundaries": [
            {
                "label": "d2-preheat-1",
                "at_ns": 94 * NS,
                "runner_request_ids": [],
                "runner_pending": False,
            }
        ],
        "waves": [
            {
                "wave": 1,
                "started_ns": 75 * NS,
                "requests": [
                    {
                        "request_id": "d2-preheat-1-r0",
                        "planned_ns": 75 * NS,
                        "submit_ns": 75 * NS,
                        "first_token_ns": 77 * NS,
                        "end_ns": 78 * NS,
                        "status": "completed",
                        "token_ids": [1],
                    }
                ],
                "drained_at_ns": 94 * NS,
                "reset_after_ns": 95 * NS,
                "sample_observed_through_ns": 95 * NS,
                "plateau_decision_ns": 96 * NS - 1,
            }
        ],
    }
    return raw, workload, samples, limits, target


def test_d2r_separates_pre_deadline_decision_from_actual_unwind(d2):
    raw, workload, samples, limits, target = d2r_prelude(d2)
    report = d2.prelude_summary(raw, workload, samples, limits, target)
    assert report["verified"], report["issues"]
    assert report["conditioned_after_ns"] == 96 * NS - 1
    assert report["conditioning_seconds_within_cap"] == 95
    assert report["cleanup_after_cutoff_seconds"] == 100 / NS
    assert report["selected_plateau"]["end_ns"] == 95 * NS

    legacy = {**raw, "schema_version": 1}
    report = d2.prelude_summary(legacy, workload, samples, LIMITS, target)
    assert not report["verified"]
    assert "conditioning_clock_or_cap_invalid" in report["issues"]
    assert "cleanup_after_cutoff_seconds" not in report


def test_d2r_rejects_late_decision_and_extended_reported_cutoff(d2):
    raw, workload, samples, limits, target = d2r_prelude(d2)
    raw["conditioning_cutoff_ns"] += 1
    raw["waves"][0]["plateau_decision_ns"] = 96 * NS + 1
    report = d2.prelude_summary(raw, workload, samples, limits, target)
    assert not report["verified"]
    assert "conditioning_fixed_cutoff_differs" in report["issues"]
    assert "preheat_plateau_decision_outside_conditioning_deadline" in report["issues"]
    assert report["selected_plateau"] is None


@pytest.mark.parametrize(
    ("reset", "decision", "status", "checked"),
    [
        (95 * NS, 96 * NS, "eligible", 1),
        (95 * NS, 96 * NS + 1, "not_eligible", 0),
        (97 * NS, 98 * NS, "not_eligible", 0),
    ],
)
def test_d2r110_shadow_requires_full_reset_and_decision_by_95s(
    d2, reset, decision, status, checked
):
    raw, workload, samples, limits, target = d2r_prelude(d2)
    limits["per_measured_engine_cap_s"] = 110
    raw["conditioning_cutoff_ns"] = 111 * NS
    raw["conditioning_end_ns"] = decision + 1
    raw["waves"][0].update(reset_after_ns=reset, plateau_decision_ns=decision)
    prelude = d2.prelude_summary(raw, workload, samples, limits, target)
    assert prelude["verified"], prelude["issues"]
    shadow = d2.conditioning_shadow(raw, prelude, 95)
    assert shadow["cutoff_ns"] == 96 * NS
    assert shadow["status"] == status
    assert shadow["completed_checked_waves"] == checked
    assert shadow["qualifying_checkpoint_ns"] == (decision if status == "eligible" else None)


def test_d2r110_shadow_preserves_missing_and_expired_evidence(d2):
    raw, workload, samples, limits, target = d2r_prelude(d2)
    limits["per_measured_engine_cap_s"] = 110
    raw.update(conditioning_cutoff_ns=111 * NS, conditioning_end_ns=111 * NS, plateau=None)
    raw["waves"][0].update(sample_observed_through_ns=75 * NS)
    prelude = d2.prelude_summary(raw, workload, samples, limits, target)
    assert not prelude["verified"]
    assert d2.conditioning_shadow(raw, prelude, 95)["status"] == "not_eligible"

    raw["conditioning_end_ns"] = 95 * NS
    prelude = d2.prelude_summary(raw, workload, samples, limits, target)
    assert d2.conditioning_shadow(raw, prelude, 95)["status"] == "unverified"


def test_d2r110_shadow_does_not_decide_post_primer_admission(d2):
    raw, workload, samples, limits, target = d2r_prelude(d2)
    limits["per_measured_engine_cap_s"] = 110
    raw.update(
        conditioning_cutoff_ns=111 * NS,
        status="invalid",
        reason="pre-t0 temperature, freshness or GPU identity outside limits",
    )
    prelude = d2.prelude_summary(raw, workload, samples, limits, target)
    assert prelude["producer_status"] == "invalid"
    assert d2.conditioning_shadow(raw, prelude, 95)["status"] == "eligible"
