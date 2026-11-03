"""Classify D4's frozen rate contrast from independently verified timings."""

from itertools import combinations

from prepare_m5_d4 import admission_summary

from vllm_prefix_scheduler.m5_campaign import require

MEASURED = ("E1", "E2", "E3", "E4")
SAME_RATE_PAIRS = (("E1", "E4"), ("E2", "E3"))


def assess_d4(runs, target, tail_count, limits, compare_thermal, *, released):
    rates = dict(zip(MEASURED, target["measure_order_rps"], strict=True))
    summaries = {}
    for identity, rate in rates.items():
        run = runs.get(identity)
        timing = run.get("native_timing", {}) if run is not None else {}
        if run is not None:
            require(run["rate_rps"] == rate, "D4 measured rate differs from ABBA order")
        admission = None
        complete = (
            run is not None
            and run.get("endpoint_evidence_complete") is True
            and timing.get("status") == "complete"
            and timing.get("clock", {}).get("verified") is True
            and timing["planned_count"]
            == timing["first_decomposed_count"]
            == timing["terminal_decomposed_count"]
            == target["count_per_engine"]
        )
        if complete:
            admission = admission_summary(timing, target["max_num_seqs"], tail_count)
        summaries[identity] = {
            "rate_rps": rate,
            "status": run["status"] if run is not None else "unattempted",
            "request_evidence_complete": complete,
            "admission": admission,
        }

    thermal_pairs = []
    for first, second in combinations(MEASURED, 2):
        reference = runs.get(first, {}).get("in_window")
        observed = runs.get(second, {}).get("in_window")
        thermal_pairs.append(
            {
                "reference": first,
                "observed": second,
                "thermal": compare_thermal(reference, observed, limits)
                if reference is not None and observed is not None
                else None,
            }
        )
    windows_comparable = all(
        pair["thermal"] is not None and pair["thermal"]["comparable"] is True
        for pair in thermal_pairs
    )
    pre_t0_verified = all(
        runs.get(name, {}).get("pre_t0_verified") is True
        and runs[name]["prelude"]["verified"] is True
        for name in MEASURED
    )
    evidence_complete = all(summary["request_evidence_complete"] for summary in summaries.values())
    signal = "unverified"
    if released and evidence_complete and pre_t0_verified and windows_comparable:
        low = [row["admission"] for row in summaries.values() if row["rate_rps"] == 6]
        high = [row["admission"] for row in summaries.values() if row["rate_rps"] == 8]
        require(len(low) == len(high) == 2, "D4 rate contrast inventory differs")
        signal = (
            "supported"
            if max(row["tail_wait_median_ns"] for row in low)
            < min(row["tail_wait_median_ns"] for row in high)
            and max(row["peak_waiting"] for row in low) <= min(row["peak_waiting"] for row in high)
            else "not_supported"
        )
    return {
        "request_evidence_complete": evidence_complete,
        "pre_t0_verified": pre_t0_verified,
        "mutually_thermally_comparable": windows_comparable,
        "all_window_thermal_pairs": thermal_pairs,
        "runs": summaries,
        "pressure_signal": signal,
        "occupancy_scope": "Overlap with native first-scheduled to last-output intervals approximates active request occupancy.",
        "qualification_started": False,
    }
