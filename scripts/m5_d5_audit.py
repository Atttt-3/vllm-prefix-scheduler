"""Assess the frozen D5 admission-cap contrast from validated native intervals."""

from fractions import Fraction
from itertools import combinations

from m5_d4_audit import MEASURED
from prepare_m5_d4 import admission_summary

from vllm_prefix_scheduler.m5_campaign import require

THROUGHPUT_FIELDS = ("requests", "tokens")
THROUGHPUT_COUNTS = {"requests": "completed_requests", "tokens": "generated_tokens"}


def throughput_guard(reference, observed, loss):
    base_window = reference["duration_ns"]["last_terminal_from_start"]
    test_window = observed["duration_ns"]["last_terminal_from_start"]
    require(
        all(type(value) is int and value > 0 for value in (base_window, test_window)),
        "D5 endpoint throughput denominator invalid",
    )
    passes = {}
    for name, count in THROUGHPUT_COUNTS.items():
        base = reference["throughput"][count]
        test = observed["throughput"][count]
        require(
            all(type(value) is int and value > 0 for value in (base, test)),
            "D5 endpoint throughput numerator invalid",
        )
        passes[name] = Fraction(test * base_window, base * test_window) >= 1 - Fraction(str(loss))
    return passes


def assess_d5(runs, pairs, target, criteria, limits, compare_thermal, *, released):
    caps = dict(zip(MEASURED, target["measure_order_max_num_active_seqs"], strict=True))
    summaries = {}
    for identity, cap in caps.items():
        run = runs.get(identity, {})
        timing = run.get("native_timing", {})
        if run:
            require(run["max_num_active_seqs"] == cap, "D5 measured cap differs from ABBA order")
        complete = (
            run.get("endpoint_evidence_complete") is True
            and timing.get("status") == "complete"
            and timing.get("clock", {}).get("verified") is True
            and timing["planned_count"]
            == timing["first_decomposed_count"]
            == timing["terminal_decomposed_count"]
            == target["count_per_engine"]
        )
        endpoint = run.get("metrics")
        summaries[identity] = {
            "max_num_active_seqs": cap,
            "status": run.get("status", "unattempted"),
            "request_evidence_complete": complete,
            "admission": admission_summary(timing, cap, criteria["tail_request_count"])
            if complete
            else None,
            "ttft_p95_ns": endpoint["strata"]["all"]["ttft_ns"]["p95"] if endpoint else None,
            "last_terminal_throughput": {
                name: endpoint["throughput"][f"last_terminal_{name}_per_s"]
                for name in THROUGHPUT_FIELDS
            }
            if endpoint
            else None,
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
        row["thermal"] is not None and row["thermal"]["comparable"] is True for row in thermal_pairs
    )
    pre_t0_verified = all(
        runs.get(name, {}).get("pre_t0_verified") is True
        and runs[name]["prelude"]["verified"] is True
        for name in MEASURED
    )
    evidence_complete = all(row["request_evidence_complete"] for row in summaries.values())
    cap4 = [name for name, cap in caps.items() if cap == 4]
    cap5 = [name for name, cap in caps.items() if cap == 5]
    require(len(cap4) == len(cap5) == 2, "D5 cap contrast inventory differs")
    require(
        [(row["reference"], row["observed"]) for row in pairs] == [tuple(cap4), tuple(cap5)],
        "D5 same-cap pair inventory differs",
    )
    same_cap_pass = all(
        row["thermal"] is not None
        and row["thermal"]["comparable"] is True
        and row["performance"] is not None
        and row["performance"]["within_tolerance"] is True
        for row in pairs
    )
    backlog = None
    if all(summaries[name]["admission"] is not None for name in cap4):
        backlog = all(
            summaries[name]["admission"]["tail_wait_median_ns"]
            >= criteria["cap4_tail_wait_median_min_ns"]
            and summaries[name]["admission"]["peak_waiting"] >= criteria["cap4_peak_waiting_min"]
            for name in cap4
        )
    contrasts = []
    for first in cap4:
        for second in cap5:
            a, b = summaries[first], summaries[second]
            tail_delta = ttft_delta = throughput_delta = throughput_pass = None
            if a["admission"] is not None and b["admission"] is not None:
                tail_delta = (
                    b["admission"]["tail_wait_median_ns"] - a["admission"]["tail_wait_median_ns"]
                )
                ttft_delta = b["ttft_p95_ns"] - a["ttft_p95_ns"]
                base, treatment = a["last_terminal_throughput"], b["last_terminal_throughput"]
                require(
                    all(base[name] > 0 for name in THROUGHPUT_FIELDS),
                    "D5 reference throughput missing",
                )
                throughput_delta = {
                    name: (treatment[name] - base[name]) / base[name] for name in THROUGHPUT_FIELDS
                }
                throughput_pass = throughput_guard(
                    runs[first]["metrics"],
                    runs[second]["metrics"],
                    criteria["cross_cap_throughput_max_loss_rel"],
                )
            contrasts.append(
                {
                    "reference": first,
                    "observed": second,
                    "tail_wait_median_delta_ns": tail_delta,
                    "ttft_p95_delta_ns": ttft_delta,
                    "relative_last_terminal_throughput_delta": throughput_delta,
                    "throughput_within_loss_limit": throughput_pass,
                }
            )

    wait_and_ttft = evidence_complete and all(
        row["tail_wait_median_delta_ns"] < 0 and row["ttft_p95_delta_ns"] < 0 for row in contrasts
    )
    throughput = evidence_complete and all(
        passed for row in contrasts for passed in row["throughput_within_loss_limit"].values()
    )
    signal = "unverified"
    if (
        released
        and evidence_complete
        and pre_t0_verified
        and windows_comparable
        and same_cap_pass
        and backlog
    ):
        signal = "supported" if wait_and_ttft and throughput else "not_supported"
    return {
        "request_evidence_complete": evidence_complete,
        "pre_t0_verified": pre_t0_verified,
        "mutually_thermally_comparable": windows_comparable,
        "all_window_thermal_pairs": thermal_pairs,
        "runs": summaries,
        "cap4_backlog_reproduced": backlog,
        "same_cap_gates_passed": same_cap_pass,
        "cross_cap_contrasts": contrasts,
        "strict_tail_wait_and_ttft_improvement": wait_and_ttft if evidence_complete else None,
        "cross_cap_throughput_gate_passed": throughput if evidence_complete else None,
        "optimization_signal": signal,
        "occupancy_scope": "Overlap with native first-scheduled to last-output intervals approximates active request occupancy.",
        "qualification_started": False,
    }
