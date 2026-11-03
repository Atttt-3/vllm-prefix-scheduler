"""Independently review the thermal and endpoint evidence of the authorized D2."""

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from check_m5_d1 import compare_pair, matched_requests
from m5_seal import ROOT, git_identity, sealed_leaf, verify_analysis

from vllm_prefix_scheduler.m5_campaign import digest, read_json, require
from vllm_prefix_scheduler.m5_contract import audit_run
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA
from vllm_prefix_scheduler.m5_resources import audit_history, device

ROLE = "D2-E-E"
PROPOSAL_SHA = "1d23948556bc102b1151e3bb4215ca2e161ce17fa786b5752ff8fe42f7eed96c"
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


def bind_thermal_artifact(leaf, row):
    path = leaf / "raw/d2-thermal.json"
    want = digest(path)
    require(want == row["d2_thermal_sha256"], "D2 thermal artifact differs from index")
    entries = read_json(leaf / "collector/manifest.json")["files"]
    require(
        sum(
            item["path"].startswith("configs/")
            and item["source"].endswith("/proposed-d2.json")
            and item["sha256"] == PROPOSAL_SHA
            for item in entries
        )
        == 1,
        "D2 reviewed proposal is not in the collector",
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
    cap = limits["pilot_cap_s" if role == "pilot" else "per_measured_engine_cap_s"] * SECOND
    issues = []
    selected = None
    selected_after = None
    previous_observed = 0
    complete_waves = 0
    start_ok = type(start) is int and start > 0
    if not start_ok or type(end) is not int or not 0 < end - start <= cap:
        issues.append("conditioning_clock_or_cap_invalid")
    if start_ok:
        previous = start
        for index, wave in enumerate(raw["waves"], start=1):
            if not completed_wave(wave, index, workload, raw["preheat_boundaries"], previous):
                issues.append("preheat_wave_incomplete")
                break
            previous = wave["reset_after_ns"]
            if previous > start + cap or type(end) is not int or previous > end:
                issues.append("preheat_wave_after_conditioning_deadline")
                break
            complete_waves += 1
            if selected is None:
                observed = wave["sample_observed_through_ns"]
                if observed is None:
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
                    selected_after = previous
            else:
                issues.append("continued_after_first_qualifying_wave")
    if selected is None:
        issues.append("no_qualifying_prelude_window")
    if raw["plateau"] != selected:
        issues.append("reported_plateau_differs_from_recomputation")
    if selected is not None and not selected_after <= end:
        issues.append("conditioning_did_not_end_after_first_qualifying_wave")
    return {
        "verified": not issues,
        "issues": issues,
        "producer_status": raw["status"],
        "producer_reason": raw["reason"],
        "complete_preheat_waves": complete_waves,
        "conditioning_seconds": (end - start) / SECOND if start_ok and type(end) is int else None,
        "selected_plateau": selected,
        "conditioned_after_ns": selected_after,
    }


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
    require(digest(proposal_path) == PROPOSAL_SHA, "D2 plan differs from the authorized proposal")
    plan = read_json(proposal_path)
    limits = plan["conditioning"]
    expected = plan["target"]
    require(
        expected["input_sha256"] == D1_INPUT_SHA
        and (tuple(expected["fresh_engines"]), tuple(tuple(p) for p in expected["primary_pairs"]))
        == (RUNS, PAIRS[:2])
        and tuple(expected["order_drift_sentinel"]) == PAIRS[-1],
        "D2 frozen engine or pair inventory differs",
    )
    c3 = attempt.parent / "c3-01/node/calibration-result.json"
    require(
        digest(c3) == C3_BLOCKED_SHA and read_json(c3)["status"] == "blocked",
        "D2 original C3 blocked outcome differs",
    )
    history = audit_history(attempt.parent)[:7]
    job = read_json(attempt / "submission.json")["stdout"].strip().split(";")[0]
    auth = read_json(attempt / "authorization.json")
    require(
        len(history) == 7
        and sum(row["requested_minutes"] for row in history) == 140
        and sum(row["role"] == ROLE for row in history) == 1
        and history[-1]["role"] == ROLE
        and history[-1]["job_id"] == job
        and history[-1]["phase"] == "calibration"
        and history[-1]["requested_minutes"] == 20
        and history[-1]["max_attempts"] == 7
        and auth["roles"] == [ROLE],
        "D2 resource receipt or one-allocation authorization differs",
    )
    resource = history[-1]
    planned = read_json(attempt / "planned.json")
    config_path = ROOT / "configs/m5-performance.json"
    config = read_json(config_path)
    gates = config["gates"]
    require(
        planned["config_sha256"] == digest(config_path)
        and planned["auxiliary_sha256"] == PROPOSAL_SHA
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
        require(index["stage"] == ROLE and index["schema_version"] == 1, "D2 index scope differs")
        rows = index["runs"]
    else:
        rows = []
    require(
        tuple(row["id"] for row in rows) == RUNS[: len(rows)], "D2 executed engine order differs"
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
        require(
            row["mode"] == expected["mode"]
            and row["tier"] == expected["tier"]
            and row["seed"] == expected["seed"]
            and row["scenario"] == expected["scenario"]
            and row["count"] == expected["count_per_engine"]
            and row["rate_rps"] == expected["arrival_rate_rps"],
            f"D2 run parameters differ for {identity}",
        )
        if row.get("returncode") != 0 or not row.get("audit_sha256"):
            runs[identity] = {"status": "incomplete", "returncode": row.get("returncode")}
            continue
        leaf = node / identity
        table = leaf / "input.json"
        require(digest(table) == D1_INPUT_SHA, f"D2 input differs for {identity}")
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
        )
        thermal = bind_thermal_artifact(leaf, row)
        require(
            thermal["schema_version"] == 1
            and thermal["plan_sha256"] == PROPOSAL_SHA
            and thermal["role"] == ("pilot" if pilot else "measure")
            and thermal["status"] == row["d2_thermal_status"]
            and thermal["status"]
            in ({"target-frozen", "invalid"} if pilot else {"admitted", "invalid"}),
            f"D2 thermal artifact scope differs for {identity}",
        )
        client, engine = (read_json(leaf / "raw" / f"{name}.json") for name in ("client", "engine"))
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
        prelude = prelude_summary(thermal, read_json(table), leaf_samples, limits, target)
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
        if complete:
            valid_metrics[identity] = metrics
            requests[identity] = {row["request_id"]: row for row in client["requests"]}
    require(len(environments) <= 1, "D2 node clock domain or engine profile varied")

    if (node / "d2-result.json").exists():
        node_result = read_json(node / "d2-result.json")
        require(
            node_result["status"] in {"diagnostic-complete", "diagnostic-incomplete"}
            and node_result["proposal_sha256"] == PROPOSAL_SHA
            and node_result["input_sha256"] == D1_INPUT_SHA
            and node_result["retained_c3_sha256"] == C3_BLOCKED_SHA
            and node_result["audited_runs"] == sum("metrics" in run for run in runs.values()),
            "D2 node result differs from independent evidence",
        )
    elif resource["complete"]:
        raise ValueError("Completed D2 lacks the node result")

    pairs = []
    for first, second in PAIRS:
        result = {"reference": first, "observed": second, "thermal": None, "performance": None}
        if all("in_window" in runs.get(key, {}) for key in (first, second)):
            result["thermal"] = compare_thermal(
                runs[first]["in_window"], runs[second]["in_window"], limits
            )
        if first in valid_metrics and second in valid_metrics:
            result["performance"] = {
                **compare_pair(valid_metrics[first], valid_metrics[second], gates),
                **matched_requests(requests[first], requests[second]),
            }
        pairs.append(result)

    heat_verified = (
        resource["complete"]
        and target is not None
        and len(runs) == len(RUNS)
        and all(
            runs[identity].get("pre_t0_verified") is True and runs[identity]["prelude"]["verified"]
            for identity in RUNS[1:]
        )
        and all(pair["thermal"] is not None and pair["thermal"]["comparable"] for pair in pairs)
    )
    outcomes = [
        pair["performance"]["within_tolerance"] for pair in pairs if pair["performance"] is not None
    ]
    endpoint = (
        "unstable"
        if False in outcomes
        else "stable"
        if len(outcomes) == len(PAIRS)
        else "unverified"
    )
    return {
        "status": "diagnostic-complete"
        if resource["complete"] and len(outcomes) == len(PAIRS)
        else "diagnostic-incomplete",
        "retained_original_C3": {"status": "blocked", "stage_result_sha256": C3_BLOCKED_SHA},
        "proposal_sha256": PROPOSAL_SHA,
        "input_sha256": D1_INPUT_SHA,
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
