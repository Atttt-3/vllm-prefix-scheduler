"""Independently review M5 thermal and endpoint diagnostic evidence."""

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from check_m5_d1 import compare_pair, matched_requests
from m5_d3_audit import assess_d3, timing_leaf, timing_pair, verify_native_sources
from m5_d4_audit import SAME_RATE_PAIRS, assess_d4
from m5_d5_audit import assess_d5
from m5_seal import ROOT, git_identity, sealed_leaf, verify_analysis

from vllm_prefix_scheduler.m5_campaign import (
    D2_ROLE,
    D2R110_OFFICIAL_ANALYSIS_SHA256,
    D2R110_ROLE,
    D2R_OFFICIAL_ANALYSIS_SHA256,
    D2R_ROLE,
    D3_ROLE,
    D4_ROLE,
    D5_ROLE,
    digest,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_contract import audit_run
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA
from vllm_prefix_scheduler.m5_d2 import (
    D2_ANALYSIS_SHA,
    D2_PROPOSAL_SHA,
    D2R110_PROPOSAL_NAME,
    D2R_PROPOSAL_NAME,
    D3_PROPOSAL_NAME,
    d2r110_plan,
    d2r_plan,
    d3_plan,
)
from vllm_prefix_scheduler.m5_d4 import (
    D4_INPUT_8_NAME,
    D4_PREVIOUS_ANALYSIS,
    D4_PROPOSAL_NAME,
    d4_plan,
)
from vllm_prefix_scheduler.m5_d5 import D5_PREVIOUS_ANALYSIS, D5_PROPOSAL_NAME, d5_plan, d5_profile
from vllm_prefix_scheduler.m5_resources import audit_history, device

ROLE = D2_ROLE
PROPOSAL_SHA = D2_PROPOSAL_SHA
RUNS = ("thermal-pilot", "E1", "E2", "E3", "E4")
PAIRS = (("E1", "E2"), ("E3", "E4"), ("E1", "E4"))
SECOND = 1_000_000_000


def near_relative(reference, observed, fraction):
    return reference > 0 and abs(observed - reference) <= fraction * reference


def memory_mode(active):
    return Counter(sample["memory_clock_mhz"] for sample in active).most_common(1)[0]


def thermal_window(samples, start, end, limits, target=None):
    """Find the first eligible trailing window using independently read samples."""
    period = limits["pilot_plateau"]["window_s"] * SECOND
    max_gap = limits["read_only_side_sampler"]["max_gap_s"] * SECOND
    minimum = start + limits["minimum_conditioning_s"] * SECOND
    plate = limits["pilot_plateau"]
    per_engine = limits["per_measured_engine"]
    candidates = [sample for sample in samples if start <= sample["monotonic_ns"] <= end]
    for last in candidates:
        right = last["monotonic_ns"]
        if right < minimum:
            continue
        left = right - period
        block = [sample for sample in candidates if left <= sample["monotonic_ns"] <= right]
        if len(block) < plate["min_total_samples"]:
            continue
        times = [sample["monotonic_ns"] for sample in block]
        if times[0] - left > max_gap or any(
            second - first > max_gap for first, second in zip(times, times[1:], strict=False)
        ):
            continue
        active = [sample for sample in block if sample["gpu_utilization_percent"] > 0]
        middle = right - period // 2
        halves = (
            [sample["sm_clock_mhz"] for sample in active if sample["monotonic_ns"] < middle],
            [sample["sm_clock_mhz"] for sample in active if sample["monotonic_ns"] >= middle],
        )
        if min(map(len, halves)) < plate["min_active_samples_per_half"]:
            continue
        temperatures = [sample["temperature_c"] for sample in block]
        if max(temperatures) - min(temperatures) > plate["max_temp_span_c"] or not near_relative(
            median(halves[0]),
            median(halves[1]),
            plate["max_sm_median_relative_change_between_halves"],
        ):
            continue
        clock, occurrences = memory_mode(active)
        if occurrences / len(active) < plate["min_active_memory_clock_mode_share"]:
            continue
        profile = {
            "start_ns": left,
            "end_ns": right,
            "sample_count": len(block),
            "active_sample_count": len(active),
            "temperature_c": median(temperatures),
            "sm_clock_mhz": median(sample["sm_clock_mhz"] for sample in active),
            "memory_clock_mhz": clock,
            "gpu_uuid": last["gpu_uuid"],
        }
        if target is None or (
            abs(profile["temperature_c"] - target["temperature_c"])
            <= per_engine["preheat_temp_median_from_target_c"]
            and near_relative(
                target["sm_clock_mhz"],
                profile["sm_clock_mhz"],
                per_engine["preheat_active_sm_median_relative_from_target"],
            )
            and profile["memory_clock_mhz"] == target["memory_clock_mhz"]
        ):
            return profile
    return None


def window_profile(samples, start, end, limits):
    """Summarize the whole measurement window without discarding a slow interval."""
    block = [sample for sample in samples if start <= sample["monotonic_ns"] <= end]
    active = [sample for sample in block if sample["gpu_utilization_percent"] > 0]
    rule = limits["in_window_comparability"]
    max_gap = limits["read_only_side_sampler"]["max_gap_s"] * SECOND
    points = [start, *(sample["monotonic_ns"] for sample in block), end]
    complete = (
        len(block) >= rule["min_total_samples_per_round"]
        and len(active) >= rule["min_active_samples_per_round"]
        and all(
            second - first <= max_gap for first, second in zip(points, points[1:], strict=False)
        )
    )
    return {
        "complete": complete,
        "sample_count": len(block),
        "active_sample_count": len(active),
        "temperature_median_c": median(sample["temperature_c"] for sample in block)
        if block
        else None,
        "sm_median_mhz": median(sample["sm_clock_mhz"] for sample in active) if active else None,
        "memory_mode_mhz": memory_mode(active)[0] if active else None,
    }


def compare_thermal(reference, observed, limits):
    if not reference["complete"] or not observed["complete"]:
        return {"comparable": False, "reason": "incomplete_in_window_telemetry"}
    rule = limits["in_window_comparability"]
    temp_difference = observed["temperature_median_c"] - reference["temperature_median_c"]
    sm_difference = observed["sm_median_mhz"] - reference["sm_median_mhz"]
    memory_equal = reference["memory_mode_mhz"] == observed["memory_mode_mhz"]
    return {
        "comparable": (
            abs(temp_difference) <= rule["max_temp_median_difference_per_pair_c"]
            and near_relative(
                reference["sm_median_mhz"],
                observed["sm_median_mhz"],
                rule["max_active_sm_median_relative_difference_per_pair"],
            )
            and memory_equal
        ),
        "temperature_median_delta_c": temp_difference,
        "active_sm_median_delta_mhz": sm_difference,
        "active_memory_modes_equal": memory_equal,
    }


def device_samples(path, gpu):
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    previous = 0
    affinity = None
    require(samples, "D2 telemetry empty")
    for sample in samples:
        at = sample["monotonic_ns"]
        require(
            type(at) is int and at > previous and sample["gpu_uuid"] == gpu,
            "D2 telemetry clock or GPU identity differs",
        )
        previous = at
        mask = sample["cpu_affinity"]
        require(
            type(mask) is list
            and mask
            and all(type(core) is int and core >= 0 for core in mask)
            and mask == sorted(set(mask))
            and (affinity is None or mask == affinity),
            "D2 telemetry CPU affinity differs",
        )
        affinity = mask
        for key in (
            "temperature_c",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "gpu_utilization_percent",
            "memory_used_mib",
            "power_w",
        ):
            value = sample[key]
            require(
                (key == "power_w" and value is None)
                or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
                f"D2 invalid device value {key}",
            )
        require(
            sample["gpu_utilization_percent"] <= 100
            and isinstance(sample["p_state"], str)
            and bool(sample["p_state"]),
            "D2 active state evidence malformed",
        )
        apps = sample["gpu_apps"]
        require(
            type(apps) is list
            and all(
                type(app["pid"]) is int
                and app["pid"] > 1
                and (app["pgid"] is None or type(app["pgid"]) is int and app["pgid"] > 1)
                for app in apps
            ),
            "D2 device process evidence malformed",
        )
    return samples


def bind_thermal_artifact(
    leaf, row, proposal_sha=PROPOSAL_SHA, *, role=ROLE, d4_configs=None, d5_configs=None
):
    path = leaf / "raw/d2-thermal.json"
    want = digest(path)
    require(want == row["d2_thermal_sha256"], "D2 thermal artifact differs from index")
    entries = read_json(leaf / "collector/manifest.json")["files"]
    proposal_name = {
        ROLE: "proposed-d2.json",
        D2R_ROLE: D2R_PROPOSAL_NAME,
        D2R110_ROLE: D2R110_PROPOSAL_NAME,
        D3_ROLE: D3_PROPOSAL_NAME,
        D4_ROLE: D4_PROPOSAL_NAME,
        D5_ROLE: D5_PROPOSAL_NAME,
    }[role]
    require(
        sum(
            item["path"].startswith("configs/")
            and item["source"].endswith(f"/{proposal_name}")
            and item["sha256"] == proposal_sha
            for item in entries
        )
        == 1,
        "D2 reviewed proposal is not in the collector",
    )
    previous = []
    if role in (D4_ROLE, D5_ROLE):
        staged = d4_configs if role == D4_ROLE else d5_configs
        require(type(staged) is dict and staged, f"{role} collector inputs missing")
        previous.extend(staged.items())
    elif role != ROLE:
        previous.append(("previous-d2-analysis.json", D2_ANALYSIS_SHA))
    if role in (D2R110_ROLE, D3_ROLE):
        previous.append(("previous-d2-r-analysis.json", D2R_OFFICIAL_ANALYSIS_SHA256))
    if role == D3_ROLE:
        previous.append(("previous-d2-r110-analysis.json", D2R110_OFFICIAL_ANALYSIS_SHA256))
    for name, sha in previous:
        require(
            sum(
                item["path"].startswith("configs/")
                and (
                    item["source"] == name
                    if name.startswith("/")
                    else item["source"].endswith(f"/{name}")
                )
                and item["sha256"] == sha
                for item in entries
            )
            == 1,
            f"D2 retained analysis {name} is not in the collector",
        )
    require(
        sum(
            item["path"].startswith("artifacts/")
            and item["source"] == f"{row['run']}/d2-thermal.json"
            and item["sha256"] == want
            for item in entries
        )
        == 1,
        "D2 thermal artifact is not in the collector",
    )
    return read_json(path)


def completed_wave(wave, number, workload, raw_boundaries, earliest):
    input_rows = workload["requests"]
    start = wave["started_ns"]
    requests = wave["requests"]
    drain, reset = wave["drained_at_ns"], wave["reset_after_ns"]
    if (
        wave["wave"] != number
        or type(start) is not int
        or start < earliest
        or type(drain) is not int
        or type(reset) is not int
        or not start < drain <= reset
        or len(requests) != len(input_rows)
    ):
        return False
    for row, planned in zip(requests, input_rows, strict=True):
        if (
            row["request_id"] != f"d2-preheat-{number}-{planned['request_id']}"
            or row["planned_ns"] != start + planned["offset_ns"]
            or row["status"] != "completed"
            or len(row["token_ids"]) != workload["sampling"]["max_tokens"]
            or any(type(row[key]) is not int for key in ("submit_ns", "first_token_ns", "end_ns"))
            or not row["planned_ns"]
            <= row["submit_ns"]
            <= row["first_token_ns"]
            <= row["end_ns"]
            <= drain
        ):
            return False
    boundaries = [point for point in raw_boundaries if point["label"] == f"d2-preheat-{number}"]
    return (
        len(boundaries) == 1
        and boundaries[0]["at_ns"] == drain
        and boundaries[0]["runner_request_ids"] == []
        and boundaries[0]["runner_pending"] is False
    )


def prelude_summary(raw, workload, samples, limits, expected_target):
    start, end = raw["conditioning_start_ns"], raw["conditioning_end_ns"]
    role = raw["role"]
    prospective = raw.get("schema_version", 1) == 2
    cap = limits["pilot_cap_s" if role == "pilot" else "per_measured_engine_cap_s"] * SECOND
    issues = []
    selected = None
    selected_after = None
    previous_observed = 0
    complete_waves = 0
    start_ok = type(start) is int and start > 0
    cutoff = start + cap if start_ok else None
    clock_ok = start_ok and type(end) is int and end > start
    if prospective:
        if (
            type(raw.get("conditioning_cutoff_ns")) is not int
            or raw["conditioning_cutoff_ns"] != cutoff
        ):
            issues.append("conditioning_fixed_cutoff_differs")
    elif clock_ok:
        clock_ok = end <= cutoff
    if not clock_ok:
        issues.append("conditioning_clock_or_cap_invalid")
    if start_ok:
        previous = start
        for index, wave in enumerate(raw["waves"], start=1):
            if not completed_wave(wave, index, workload, raw["preheat_boundaries"], previous):
                issues.append("preheat_wave_incomplete")
                break
            previous = wave["reset_after_ns"]
            if (
                previous > cutoff
                or type(end) is not int
                or previous > end
                or (prospective and wave["started_ns"] >= cutoff)
            ):
                issues.append("preheat_wave_after_conditioning_deadline")
                break
            complete_waves += 1
            if selected is None:
                decision = wave.get("plateau_decision_ns") if prospective else previous
                if prospective and (
                    type(decision) is not int or not previous <= decision <= min(cutoff, end)
                ):
                    issues.append("preheat_plateau_decision_outside_conditioning_deadline")
                    break
                observed = wave["sample_observed_through_ns"]
                if observed is None:
                    if prospective:
                        previous = decision
                    continue
                if (
                    type(observed) is not int
                    or not previous_observed <= observed <= previous
                    or not any(sample["monotonic_ns"] == observed for sample in samples)
                ):
                    issues.append("preheat_sample_visibility_boundary_invalid")
                    break
                previous_observed = observed
                selected = thermal_window(samples, start, observed, limits, expected_target)
                if selected is not None:
                    selected_after = decision
                if prospective:
                    previous = decision
            else:
                issues.append("continued_after_first_qualifying_wave")
    if selected is None:
        issues.append("no_qualifying_prelude_window")
    if raw["plateau"] != selected:
        issues.append("reported_plateau_differs_from_recomputation")
    if selected is not None and not selected_after <= end:
        issues.append("conditioning_did_not_end_after_first_qualifying_wave")
    report = {
        "verified": not issues,
        "issues": issues,
        "producer_status": raw["status"],
        "producer_reason": raw["reason"],
        "complete_preheat_waves": complete_waves,
        "conditioning_seconds": (end - start) / SECOND if start_ok and type(end) is int else None,
        "selected_plateau": selected,
        "conditioned_after_ns": selected_after,
    }
    if prospective:
        report.update(
            conditioning_cutoff_ns=raw.get("conditioning_cutoff_ns"),
            conditioning_seconds_within_cap=(min(end, cutoff) - start) / SECOND
            if clock_ok
            else None,
            cleanup_after_cutoff_seconds=max(0, end - cutoff) / SECOND if clock_ok else None,
        )
    return report


def conditioning_shadow(raw, prelude, cap_s):
    """Determine which fully checked thermal checkpoint this shorter cap could admit."""
    start, end = raw["conditioning_start_ns"], raw["conditioning_end_ns"]
    cutoff = start + cap_s * SECOND if type(start) is int and start > 0 else None
    selected = prelude["conditioned_after_ns"]
    complete = raw["waves"][: prelude["complete_preheat_waves"]]
    checkpoints = [
        wave
        for wave in complete
        if cutoff is not None
        and type(wave.get("plateau_decision_ns")) is int
        and wave["reset_after_ns"] <= wave["plateau_decision_ns"] <= cutoff
    ]
    result = {
        "cap_seconds": cap_s,
        "cutoff_ns": cutoff,
        "status": "unverified",
        "reason": "conditioning_evidence_does_not_cover_shadow_cutoff",
        "completed_checked_waves": len(checkpoints),
        "qualifying_checkpoint_ns": None,
    }
    ordinary_stop = {
        "no_qualifying_prelude_window",
        "preheat_wave_incomplete",
        "preheat_wave_after_conditioning_deadline",
        "preheat_plateau_decision_outside_conditioning_deadline",
    }
    if cutoff is None or type(end) is not int or set(prelude["issues"]) - ordinary_stop:
        return result
    if prelude["verified"] and any(wave["plateau_decision_ns"] == selected for wave in checkpoints):
        result.update(status="eligible", reason=None, qualifying_checkpoint_ns=selected)
    elif end >= cutoff:
        result.update(status="not_eligible", reason="no_qualifying_checkpoint_by_cutoff")
    return result


def pre_t0_ok(raw, samples, phases, target, limits):
    detail = raw["pre_t0"]
    if not detail or target is None:
        return False
    decision = detail["decision_ns"]
    t0 = phases["measure_start_ns"]
    sample = detail["sample"]
    if not sample or not isinstance(decision, int):
        return False
    seen = sample["monotonic_ns"]
    # The GPU query timestamp precedes the process query and journal flush.
    # One newer row can have been sampled by the writer but not visible to the replay.
    newer = [row for row in samples if seen < row["monotonic_ns"] <= decision]
    rule = limits["per_measured_engine"]
    return (
        raw["conditioning_end_ns"] <= phases["warmup_end_ns"] <= decision <= t0
        and detail["measure_start_ns"] == t0
        and sample in samples
        and seen <= decision
        and len(newer) <= 1
        and 0 <= t0 - seen <= rule["max_age_of_actual_pre_t0_device_sample_s"] * SECOND
        and abs(sample["temperature_c"] - target["temperature_c"])
        <= rule["actual_pre_t0_temp_from_target_c"]
    )


def thermal_target(plateau):
    if plateau is None:
        return None
    return {name: plateau[name] for name in ("temperature_c", "sm_clock_mhz", "memory_clock_mhz")}


def review(attempt, proposal_path):
    plan = read_json(proposal_path)
    role = plan["role"]
    latest = role == D2R110_ROLE
    d3 = role == D3_ROLE
    d4 = role == D4_ROLE
    d5 = role == D5_ROLE
    native_timing = d3 or d4 or d5
    prospective = role in (D2R_ROLE, D2R110_ROLE, D3_ROLE, D4_ROLE, D5_ROLE)
    proposal_sha = digest(proposal_path)
    thermal_plan = plan
    if d5:
        plan, thermal_plan = d5_plan(proposal_path, ROOT / "configs/m5-d3.json")
    elif d4:
        plan, thermal_plan = d4_plan(proposal_path, ROOT / "configs/m5-d3.json")
    elif d3:
        d3_plan(plan)
    elif latest:
        d2r110_plan(plan)
    elif prospective:
        d2r_plan(plan)
    else:
        require(
            role == ROLE and proposal_sha == PROPOSAL_SHA,
            "D2 plan differs from the authorized proposal",
        )
    limits = thermal_plan["conditioning"]
    expected = plan["target"]
    pairs_to_review = (
        tuple(tuple(pair) for pair in expected["same_cap_pairs"])
        if d5
        else SAME_RATE_PAIRS
        if d4
        else PAIRS
    )
    thermal_target_plan = thermal_plan["target"]
    require(
        thermal_target_plan["input_sha256"] == D1_INPUT_SHA
        and (
            tuple(thermal_target_plan["fresh_engines"]),
            tuple(tuple(p) for p in thermal_target_plan["primary_pairs"]),
        )
        == (RUNS, PAIRS[:2])
        and tuple(thermal_target_plan["order_drift_sentinel"]) == PAIRS[-1],
        "D2 frozen engine or pair inventory differs",
    )
    c3 = attempt.parent / "c3-01/node/calibration-result.json"
    require(
        digest(c3) == C3_BLOCKED_SHA and read_json(c3)["status"] == "blocked",
        "D2 original C3 blocked outcome differs",
    )
    count = {ROLE: 7, D2R_ROLE: 8, D2R110_ROLE: 9, D3_ROLE: 10, D4_ROLE: 11, D5_ROLE: 12}[role]
    history = audit_history(attempt.parent)[:count]
    job = read_json(attempt / "submission.json")["stdout"].strip().split(";")[0]
    auth = read_json(attempt / "authorization.json")
    require(
        len(history) == count
        and sum(row["requested_minutes"] for row in history) == 20 * count
        and sum(row["role"] == role for row in history) == 1
        and history[-1]["role"] == role
        and history[-1]["job_id"] == job
        and history[-1]["phase"] == "calibration"
        and history[-1]["requested_minutes"] == 20
        and history[-1]["max_attempts"] == count
        and auth["roles"] == [role],
        "D2 resource receipt or one-allocation authorization differs",
    )
    if prospective and not (d4 or d5):
        prior = attempt.parent.parent / "d2-independent-review/analysis.json"
        auth_plan = {
            D2R_ROLE: "d2r_plan_sha256",
            D2R110_ROLE: "d2r110_plan_sha256",
            D3_ROLE: "d3_plan_sha256",
        }
        require(
            auth[auth_plan[role]] == proposal_sha
            and history[6]["role"] == ROLE
            and digest(prior) == D2_ANALYSIS_SHA
            and read_json(prior)["status"] == "diagnostic-incomplete",
            "D2-R authorization or previous D2 evidence differs",
        )
    if latest or d3:
        prior_retry = attempt.parent.parent / "d2-r-review/analysis.json"
        require(
            history[7]["role"] == D2R_ROLE
            and digest(prior_retry) == D2R_OFFICIAL_ANALYSIS_SHA256
            and read_json(prior_retry)["status"] == "diagnostic-incomplete",
            "D2-R110 previous D2-R evidence differs",
        )
    if d3:
        prior_110 = attempt.parent.parent / "d2-r110-review/analysis.json"
        require(
            history[8]["role"] == D2R110_ROLE
            and history[8]["complete"]
            and digest(prior_110) == D2R110_OFFICIAL_ANALYSIS_SHA256
            and read_json(prior_110)["status"] == "diagnostic-complete",
            "D3 retained D2-R110 result differs",
        )
    if d4:
        prior_d3 = attempt.parent.parent / "d3-review/analysis.json"
        predecessor = read_json(prior_d3)
        preheat_path = c3.parent / "mixed-2-endpoint/input.json"
        preheat_sha = expected["arrival_tables"][str(expected["preheat_rate_rps"])]["sha256"]
        require(
            auth["d4_plan_sha256"] == proposal_sha
            and history[9]["role"] == D3_ROLE
            and history[9]["complete"]
            and digest(prior_d3) == plan["D3_analysis_sha256"]
            and predecessor["role"] == D3_ROLE
            and predecessor["status"] == "diagnostic-complete"
            and predecessor["proposal_sha256"] == plan["D3_plan_sha256"]
            and predecessor["resource_release"] == json.loads(json.dumps(history[9], default=str))
            and digest(preheat_path) == preheat_sha == D1_INPUT_SHA,
            "D4 authorization, retained D3 or preheat source differs",
        )
        preheat_workload = read_json(preheat_path)
        rates = dict(
            zip(RUNS, (expected["preheat_rate_rps"], *expected["measure_order_rps"]), strict=True)
        )
    if d5:
        prior_d4 = attempt.parent.parent / "d4-review/analysis.json"
        predecessor = read_json(prior_d4)
        require(
            auth["d5_plan_sha256"] == proposal_sha
            and history[10]["role"] == D4_ROLE
            and history[10]["complete"]
            and digest(prior_d4) == plan["D4_analysis_sha256"]
            and predecessor["role"] == D4_ROLE
            and predecessor["status"] == "diagnostic-complete"
            and predecessor["proposal_sha256"] == plan["D4_plan_sha256"]
            and predecessor["d4_assessment"]["pressure_signal"] == "supported"
            and predecessor["resource_release"] == json.loads(json.dumps(history[10], default=str)),
            "D5 authorization or retained D4 evidence differs",
        )
    resource = history[-1]
    planned = read_json(attempt / "planned.json")
    config_path = ROOT / "configs/m5-performance.json"
    config = read_json(config_path)
    gates = config["gates"]
    require(
        planned["config_sha256"] == digest(config_path)
        and planned["auxiliary_sha256"] == proposal_sha
        and (
            gates["overhead_ttft_ns"],
            gates["overhead_ttft_rel"],
            gates["overhead_throughput_rel"],
        )
        == (2_000_000, 0.05, 0.05),
        "D2 original equivalence gate or config source differs",
    )
    commit = planned["execution_commit"]
    verify_analysis(commit, commit)
    git_files = git_identity(commit)
    node = attempt / "node"
    allocation = read_json(node / "allocation.json")
    index_path = node / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        require(index["stage"] == role and index["schema_version"] == 1, "D2 index scope differs")
        rows = index["runs"]
    else:
        rows = []
    require(
        tuple(row["id"] for row in rows) == RUNS[: len(rows)], "D2 executed engine order differs"
    )
    if prospective:
        require(
            all(
                row.get("returncode") == 0
                and row.get("audit_sha256")
                and row.get("d2_thermal_status") == ("target-frozen" if i == 0 else "admitted")
                for i, row in enumerate(rows[:-1])
            ),
            "D2-R continued after an invalid or failed engine",
        )
    samples_path = node / "d2-device.jsonl"
    samples = []
    if samples_path.is_file():
        require(
            digest(samples_path) == allocation["d2_device_sha256"],
            "D2 device journal differs from allocation",
        )
        if samples_path.stat().st_size:
            samples = device_samples(samples_path, device(allocation["gpu_before"])[0])
        else:
            require(not resource["complete"], "Completed D2 has an empty device journal")
    elif resource["complete"]:
        raise ValueError("Completed D2 lacks the device journal")
    snapshots = [json.loads(line) for line in (node / "gpu-during.jsonl").read_text().splitlines()]
    require(
        samples == [row["thermal"] for row in snapshots if "thermal" in row],
        "D2 device journal differs from the allocator snapshot journal",
    )
    runs = {}
    target = None
    client_pids, engine_pids, environments = set(), set(), set()
    valid_metrics, requests = {}, {}
    for row in rows:
        identity = row["id"]
        pilot = identity == RUNS[0]
        rate = rates[identity] if d4 else expected["arrival_rate_rps"]
        d5_expected_profile = d5_profile(plan, identity) if d5 else None
        if d5:
            require(
                all(
                    type(row.get(name)) is int and row[name] == value
                    for name, value in d5_expected_profile.items()
                ),
                f"D5 indexed capacity or runner settings differ for {identity}",
            )
        require(
            row["mode"] == expected["mode"]
            and row["tier"] == expected["tier"]
            and row["seed"] == expected["seed"]
            and row["scenario"] == expected["scenario"]
            and row["count"] == expected["count_per_engine"]
            and row["rate_rps"] == rate,
            f"D2 run parameters differ for {identity}",
        )
        if row.get("returncode") != 0 or not row.get("audit_sha256"):
            runs[identity] = {"status": "incomplete", "returncode": row.get("returncode")}
            if d4:
                runs[identity]["rate_rps"] = rate
            if d5:
                runs[identity]["max_num_active_seqs"] = d5_expected_profile["max_num_active_seqs"]
            if latest and not pilot:
                runs[identity]["conditioning_95s_shadow"] = {
                    "cap_seconds": plan["shadow_conditioning_cap_s"],
                    "cutoff_ns": None,
                    "status": "unverified",
                    "reason": "sealed_thermal_evidence_unavailable",
                    "completed_checked_waves": 0,
                    "qualifying_checkpoint_ns": None,
                }
            continue
        leaf = node / identity
        table = leaf / "input.json"
        input_sha = expected["arrival_tables"][str(rate)]["sha256"] if d4 else D1_INPUT_SHA
        require(digest(table) == input_sha, f"D2 input differs for {identity}")
        bound = sealed_leaf(
            attempt,
            planned["remote_attempt"],
            row,
            table,
            config_path,
            config,
            job,
            git_files,
            commit,
            **({"d5_expected_profile": d5_expected_profile} if d5 else {}),
        )
        if native_timing:
            verify_native_sources(thermal_plan, read_json(leaf / "raw/sources.json"))
        d4_configs = None
        d5_configs = None
        if d4:
            d4_configs = {
                D3_PROPOSAL_NAME: plan["D3_plan_sha256"],
                D4_INPUT_8_NAME: preheat_sha,
                D4_PREVIOUS_ANALYSIS: plan["D3_analysis_sha256"],
                "original-c3-result.json": C3_BLOCKED_SHA,
            }
            if not pilot:
                pilot_thermal = f"{planned['remote_attempt']}/thermal-pilot/raw/d2-thermal.json"
                d4_configs[pilot_thermal] = digest(node / "thermal-pilot/raw/d2-thermal.json")
        if d5:
            d5_configs = {
                D3_PROPOSAL_NAME: plan["D3_plan_sha256"],
                D5_PREVIOUS_ANALYSIS: plan["D4_analysis_sha256"],
                "original-c3-result.json": C3_BLOCKED_SHA,
            }
            if not pilot:
                pilot_thermal = f"{planned['remote_attempt']}/thermal-pilot/raw/d2-thermal.json"
                d5_configs[pilot_thermal] = digest(node / "thermal-pilot/raw/d2-thermal.json")
        thermal = bind_thermal_artifact(
            leaf, row, proposal_sha, role=role, d4_configs=d4_configs, d5_configs=d5_configs
        )
        require(
            thermal["schema_version"] == (2 if prospective else 1)
            and thermal["plan_sha256"] == proposal_sha
            and thermal["role"] == ("pilot" if pilot else "measure")
            and thermal["status"] == row["d2_thermal_status"]
            and thermal["status"]
            in ({"target-frozen", "invalid"} if pilot else {"admitted", "invalid"}),
            f"D2 thermal artifact scope differs for {identity}",
        )
        if prospective and thermal["conditioning_start_ns"] is not None:
            require(
                row["started_monotonic_ns"]
                <= thermal["conditioning_start_ns"]
                <= thermal["conditioning_end_ns"]
                <= row["ended_monotonic_ns"],
                f"D2-R conditioning and cleanup lie outside the owned engine for {identity}",
            )
        client, engine = (read_json(leaf / "raw" / f"{name}.json") for name in ("client", "engine"))
        if d4:
            profile = engine["profile"]
            require(
                profile["max_num_seqs"] == expected["max_num_seqs"]
                and profile["max_num_active_seqs"] is None,
                "D4 runtime active sequence capacity differs",
            )
        phases = client["phases"]
        if pilot or thermal["status"] == "invalid":
            require(
                read_json(leaf / "audit.json") == thermal
                and not client["requests"]
                and "measure_start_ns" not in phases,
                f"D2 untimed run contains measured data for {identity}",
            )
        clock = client["clock"]
        require(
            clock["pid"] not in client_pids and engine["clock"]["pid"] not in engine_pids,
            f"D2 engine was not fresh for {identity}",
        )
        client_pids.add(clock["pid"])
        engine_pids.add(engine["clock"]["pid"])
        environments.add(
            (
                clock["host"],
                clock["boot_id"],
                clock["time_namespace"],
                engine["profile"]["num_gpu_blocks"],
            )
        )
        leaf_samples = [
            sample
            for sample in samples
            if row["started_monotonic_ns"] <= sample["monotonic_ns"] <= row["ended_monotonic_ns"]
        ]
        workload = read_json(table)
        prelude = prelude_summary(
            thermal, preheat_workload if d4 else workload, leaf_samples, limits, target
        )
        owned_windows = [
            (thermal["conditioning_start_ns"], thermal["conditioning_end_ns"]),
            (phases.get("measure_start_ns"), phases.get("drain_end_ns")),
        ]
        owned_samples = [
            sample
            for sample in leaf_samples
            if any(
                start is not None and end is not None and start <= sample["monotonic_ns"] <= end
                for start, end in owned_windows
            )
        ]
        foreign = {app["pgid"] for sample in owned_samples for app in sample["gpu_apps"]} - {
            row["engine_pgid"]
        }
        has_owner = any(sample["gpu_apps"] for sample in owned_samples)
        if foreign or not has_owner:
            prelude["issues"].append("incomplete_or_foreign_gpu_process_evidence")
            prelude["verified"] = False
        runs[identity] = {
            "status": thermal["status"],
            "evidence": bound,
            "thermal_artifact_sha256": row["d2_thermal_sha256"],
            "prelude": prelude,
        }
        if d4:
            runs[identity]["rate_rps"] = rate
        if d5:
            runs[identity]["max_num_active_seqs"] = engine["profile"]["max_num_active_seqs"]
        if latest and not pilot:
            runs[identity]["conditioning_95s_shadow"] = conditioning_shadow(
                thermal, prelude, plan["shadow_conditioning_cap_s"]
            )
        if pilot:
            candidate = thermal_target(prelude["selected_plateau"])
            if (
                thermal["status"] == "target-frozen"
                and prelude["verified"]
                and thermal["target"] == candidate
                and thermal["target_gpu_uuid"] == prelude["selected_plateau"]["gpu_uuid"]
                and thermal["pre_t0"] is None
            ):
                target = candidate
            else:
                prelude["verified"] = False
                prelude["issues"].append("pilot_target_unverified")
            continue
        if thermal["status"] != "admitted":
            continue
        pre_t0 = (
            thermal["target"] == target
            and thermal["target_gpu_uuid"] == device(allocation["gpu_before"])[0]
            and pre_t0_ok(thermal, leaf_samples, phases, target, limits)
        )
        measured = window_profile(
            leaf_samples, phases["measure_start_ns"], phases["drain_end_ns"], limits
        )
        run = audit_run(
            config_path,
            table,
            leaf / "raw",
            pair_id=identity,
            mode=expected["mode"],
            phase="calibration",
        )
        require(run == read_json(leaf / "audit.json"), f"D2 on-node audit differs for {identity}")
        metrics = run["metrics"]
        counts = metrics["strata"]["all"]["counts"]
        complete = all(
            counts[key] == expected["count_per_engine"]
            for key in ("planned", "submitted", "completed", "timely_first_token")
        )
        runs[identity].update(
            pre_t0_verified=pre_t0,
            in_window=measured,
            counts=counts,
            metrics=metrics,
            endpoint_evidence_complete=complete,
        )
        if native_timing:
            runs[identity]["native_timing"] = timing_leaf(
                client, engine, workload, gates, leaf_end_ns=row["ended_monotonic_ns"]
            )
        if complete:
            valid_metrics[identity] = metrics
            requests[identity] = {row["request_id"]: row for row in client["requests"]}
    require(len(environments) <= 1, "D2 node clock domain or engine profile varied")

    if d5:
        node_path = node / "d5-result.json"
        if node_path.exists():
            audited = sum("metrics" in run for run in runs.values())
            require(
                read_json(node_path)
                == {
                    "status": "diagnostic-complete"
                    if audited == len(RUNS) - 1
                    else "diagnostic-incomplete",
                    "role": D5_ROLE,
                    "proposal_sha256": proposal_sha,
                    "inherited_d3_proposal_sha256": plan["D3_plan_sha256"],
                    "retained_d4_analysis_sha256": plan["D4_analysis_sha256"],
                    "input_sha256": D1_INPUT_SHA,
                    "audited_runs": audited,
                    "retained_c3_sha256": C3_BLOCKED_SHA,
                },
                "D5 node result differs from independent evidence",
            )
        elif resource["complete"]:
            raise ValueError("Completed D5 lacks the node result")
    elif d4:
        node_path = node / "d4-result.json"
        measurement_hashes = {
            rate: entry["sha256"] for rate, entry in expected["arrival_tables"].items()
        }
        if node_path.exists():
            audited = sum("metrics" in run for run in runs.values())
            node_result = read_json(node_path)
            require(
                node_result
                == {
                    "status": "diagnostic-complete"
                    if audited == len(RUNS) - 1
                    else "diagnostic-incomplete",
                    "role": D4_ROLE,
                    "proposal_sha256": proposal_sha,
                    "inherited_d3_proposal_sha256": plan["D3_plan_sha256"],
                    "retained_d3_analysis_sha256": plan["D3_analysis_sha256"],
                    "preheat_input_sha256": preheat_sha,
                    "measurement_input_sha256": measurement_hashes,
                    "measure_order_rps": expected["measure_order_rps"],
                    "audited_runs": audited,
                    "retained_c3_sha256": C3_BLOCKED_SHA,
                },
                "D4 node result differs from independent evidence",
            )
        elif resource["complete"]:
            raise ValueError("Completed D4 lacks the node result")
    elif (node / "d2-result.json").exists():
        node_result = read_json(node / "d2-result.json")
        require(
            node_result["status"] in {"diagnostic-complete", "diagnostic-incomplete"}
            and node_result["proposal_sha256"] == proposal_sha
            and node_result["input_sha256"] == D1_INPUT_SHA
            and node_result["retained_c3_sha256"] == C3_BLOCKED_SHA
            and node_result["audited_runs"] == sum("metrics" in run for run in runs.values()),
            "D2 node result differs from independent evidence",
        )
        if prospective:
            require(
                node_result["role"] == role
                and node_result["retained_d2_analysis_sha256"] == D2_ANALYSIS_SHA,
                "D2-R node result does not bind the retained D2 analysis",
            )
        if latest or d3:
            require(
                node_result["retained_d2r_analysis_sha256"] == D2R_OFFICIAL_ANALYSIS_SHA256,
                "D2-R110 node result does not bind the retained D2-R analysis",
            )
        if d3:
            require(
                node_result["retained_d2r110_analysis_sha256"] == D2R110_OFFICIAL_ANALYSIS_SHA256,
                "D3 node result does not bind the retained D2-R110 analysis",
            )
    elif resource["complete"]:
        raise ValueError("Completed D2 lacks the node result")

    pairs = []
    for first, second in pairs_to_review:
        result = {"reference": first, "observed": second, "thermal": None, "performance": None}
        if d4:
            require(rates[first] == rates[second], "D4 comparison joins different rates")
            result["rate_rps"] = rates[first]
        if d5:
            first_cap = d5_profile(plan, first)["max_num_active_seqs"]
            require(
                first_cap == d5_profile(plan, second)["max_num_active_seqs"],
                "D5 comparison joins different caps",
            )
            result["max_num_active_seqs"] = first_cap
        if all("in_window" in runs.get(key, {}) for key in (first, second)):
            result["thermal"] = compare_thermal(
                runs[first]["in_window"], runs[second]["in_window"], limits
            )
        if first in valid_metrics and second in valid_metrics:
            result["performance"] = {
                **compare_pair(valid_metrics[first], valid_metrics[second], gates),
                **matched_requests(requests[first], requests[second]),
            }
        if native_timing:
            native = None
            if all("native_timing" in runs.get(key, {}) for key in (first, second)):
                native = timing_pair(runs[first]["native_timing"], runs[second]["native_timing"])
                if result["performance"] is not None:
                    require(
                        native["p95_anchors"]["ttft_p95_difference_ns"]
                        == result["performance"]["ttft_delta_ns"],
                        "D3 same-cohort native summary differs from original endpoint TTFT",
                    )
            result["native_timing"] = native
        pairs.append(result)

    assessment = None
    if d5:
        assessment = assess_d5(
            runs,
            pairs,
            expected,
            plan["assessment"],
            limits,
            compare_thermal,
            released=resource["complete"],
        )
    elif d4:
        assessment = assess_d4(
            runs,
            expected,
            plan["assessment"]["tail_request_count"],
            limits,
            compare_thermal,
            released=resource["complete"],
        )
    heat_verified = (
        resource["complete"]
        and target is not None
        and len(runs) == len(RUNS)
        and all(
            runs[identity].get("pre_t0_verified") is True and runs[identity]["prelude"]["verified"]
            for identity in RUNS[1:]
        )
        and (
            assessment["mutually_thermally_comparable"]
            if d4 or d5
            else all(
                pair["thermal"] is not None and pair["thermal"]["comparable"] for pair in pairs
            )
        )
    )
    outcomes = [
        pair["performance"]["within_tolerance"] for pair in pairs if pair["performance"] is not None
    ]
    endpoint = (
        "unstable"
        if False in outcomes
        else "stable"
        if len(outcomes) == len(pairs_to_review)
        else "unverified"
    )
    return {
        "status": "diagnostic-complete"
        if resource["complete"]
        and len(outcomes) == len(pairs_to_review)
        and (not (d4 or d5) or assessment["request_evidence_complete"])
        else "diagnostic-incomplete",
        "retained_original_C3": {"status": "blocked", "stage_result_sha256": C3_BLOCKED_SHA},
        "proposal_sha256": proposal_sha,
        **(
            {"role": role, "retained_original_D2": plan["retained_D2"]}
            if prospective and not (d4 or d5)
            else {}
        ),
        **(
            {
                "retained_original_D2R": plan["retained_D2R"],
                **(
                    {"shadow_conditioning_cap_s": plan["shadow_conditioning_cap_s"]}
                    if latest
                    else {}
                ),
            }
            if latest or d3
            else {}
        ),
        **(
            {
                "retained_original_D2R110": plan["retained_D2R110"],
                "d3_assessment": assess_d3(runs, pairs, gates),
            }
            if d3
            else {}
        ),
        **(
            {
                "role": role,
                "inherited_d3_proposal_sha256": plan["D3_plan_sha256"],
                "retained_d3_analysis_sha256": plan["D3_analysis_sha256"],
                "preheat_input_sha256": preheat_sha,
                "measurement_input_sha256": measurement_hashes,
                "d4_assessment": assessment,
            }
            if d4
            else {"input_sha256": D1_INPUT_SHA}
        ),
        **(
            {
                "role": role,
                "inherited_d3_proposal_sha256": plan["D3_plan_sha256"],
                "retained_d4_analysis_sha256": plan["D4_analysis_sha256"],
                "d5_assessment": assessment,
            }
            if d5
            else {}
        ),
        "execution_commit": commit,
        "analysis_commit": commit,
        "resource_release": resource,
        "resource_ledger": history,
        "device_journal_sha256": digest(samples_path) if samples_path.exists() else None,
        "verified_thermal_target": target,
        "runs": runs,
        "pairs": pairs,
        "decision": {
            "thermal_control": "verified" if heat_verified else "unverified",
            "observed_endpoint_stability": endpoint,
            "endpoint_stability_under_control": endpoint if heat_verified else "unverified",
            "qualification_started": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(not sys.flags.optimize, "run unoptimized D2 auditor")
        report = review(args.attempt.resolve(), args.proposal.resolve())
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, default=str, allow_nan=False)
            stream.write("\n")
    except (KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"M5 D2 evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
