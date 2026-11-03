"""Independently audit the one-allocation M5 D1 diagnostic against sealed originals."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

from m5_seal import ROOT, git_identity, sealed_leaf, verify_analysis

from vllm_prefix_scheduler.m5_campaign import D1_ROLE, digest, read_json, require
from vllm_prefix_scheduler.m5_contract import audit_run
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA, D1_PROPOSAL_SHA
from vllm_prefix_scheduler.m5_resources import audit_history, device

TELEMETRY_FIELDS = ("sm_clock_mhz", "memory_clock_mhz", "temperature_c", "power_w")


def compare_pair(reference, observed, gates):
    """Use the original C3 denominator and equivalence bounds for one pair."""
    reference_ttft, observed_ttft = (
        metrics["strata"]["all"]["ttft_ns"]["p95"] for metrics in (reference, observed)
    )
    require(
        all(type(value) is int and value > 0 for value in (reference_ttft, observed_ttft)),
        "D1 TTFT p95 absent or invalid",
    )
    delta = observed_ttft - reference_ttft
    tolerance = max(gates["overhead_ttft_ns"], reference_ttft * gates["overhead_ttft_rel"])
    throughputs = {}
    for kind in ("requests", "tokens"):
        field = f"last_terminal_{kind}_per_s"
        before, after = (metrics["throughput"][field] for metrics in (reference, observed))
        require(
            all(
                type(value) in (int, float) and math.isfinite(value) and value > 0
                for value in (before, after)
            ),
            f"D1 {kind} throughput absent or invalid",
        )
        throughputs[kind] = (after - before) / before
    ttft_within = abs(delta) <= tolerance
    throughput_within = all(
        abs(value) <= gates["overhead_throughput_rel"] for value in throughputs.values()
    )
    return {
        "reference_ttft_p95_ns": reference_ttft,
        "observed_ttft_p95_ns": observed_ttft,
        "ttft_delta_ns": delta,
        "ttft_tolerance_ns": tolerance,
        "relative_throughput_delta": throughputs,
        "ttft_within_tolerance": ttft_within,
        "throughput_within_tolerance": throughput_within,
        "within_tolerance": ttft_within and throughput_within,
    }


def qualification_decision(pairs):
    """Classify D1 only, without granting qualification or changing original C3."""
    endpoint, light, cross = (pairs[key] for key in ("E/E", "L/L", "E/L"))
    require((len(endpoint), len(light), len(cross)) == (2, 2, 4), "incomplete D1 pair inventory")
    if all(row["within_tolerance"] for rows in pairs.values() for row in rows):
        return {"enter_qualification": True, "findings": ["all_eight_pairs_within_tolerance"]}
    findings = []
    if any(not row["within_tolerance"] for row in endpoint):
        findings.append("endpoint_repeat_instability")
    if any(not row["within_tolerance"] for row in light):
        findings.append("light_repeat_instability")
    cross_outside_same_direction = any(
        all(sign * row["ttft_delta_ns"] > row["ttft_tolerance_ns"] for row in cross)
        for sign in (-1, 1)
    )
    if not findings and cross_outside_same_direction:
        findings.append("repeatable_tier_ttft_difference")
    elif not findings:
        findings.append("cross_tier_difference_not_resolved")
    return {"enter_qualification": False, "findings": findings}


def telemetry(path, expected_gpu):
    snapshot = read_json(path)
    gpu = snapshot["gpu"]
    require(
        type(gpu["returncode"]) is int and gpu["returncode"] == 0 and not gpu["stderr"].strip(),
        "D1 device query did not succeed",
    )
    parts = [part.strip() for part in gpu["stdout"].strip().split(",")]
    require(len(parts) == 5 and parts[0] == expected_gpu, "D1 device identity differs")
    affinity = snapshot["cpu_affinity"]
    require(
        type(affinity) is list
        and affinity
        and all(type(core) is int and core >= 0 for core in affinity)
        and affinity == sorted(set(affinity)),
        "D1 CPU affinity evidence absent or invalid",
    )
    metrics = {}
    for name, raw in zip(TELEMETRY_FIELDS, parts[1:], strict=True):
        value = None if raw in {"N/A", "[N/A]", "[Not Supported]"} else float(raw)
        require(value is None or math.isfinite(value) and value >= 0, "invalid D1 device sample")
        metrics[name] = value
    stamp = snapshot["monotonic_ns"]
    require(type(stamp) is int and stamp > 0, "invalid D1 device sample clock")
    return {"monotonic_ns": stamp, "cpu_affinity": affinity, **metrics}


def matched_requests(reference, observed):
    require(
        reference.keys() == observed.keys() and len(reference) == 60, "D1 request match differs"
    )
    deltas = []
    cache_matches, token_matches = 0, 0
    for identity, before in reference.items():
        after = observed[identity]
        deltas.append(
            (after["first_token_ns"] - after["submit_ns"])
            - (before["first_token_ns"] - before["submit_ns"])
        )
        cache_matches += before["cached_tokens"] == after["cached_tokens"]
        token_matches += before["token_ids"] == after["token_ids"]
    return {
        "request_count": len(deltas),
        "same_cache_count": cache_matches,
        "same_output_count": token_matches,
        "submit_to_first_token_median_delta_ns": median(deltas),
        "observed_faster_requests": sum(delta < 0 for delta in deltas),
    }


def review(attempt, proposal_path):
    require(digest(proposal_path) == D1_PROPOSAL_SHA, "D1 proposal differs from approved matrix")
    proposal = read_json(proposal_path)
    historical_path = attempt.parent / "c3-01" / "node" / "calibration-result.json"
    require(
        digest(historical_path) == C3_BLOCKED_SHA
        and read_json(historical_path)["status"] == "blocked",
        "retained C3 failure evidence differs",
    )
    job = read_json(attempt / "submission.json")["stdout"].strip().split(";")[0]
    history = audit_history(attempt.parent)[:6]
    require(len(history) == 6, "D1 must consume the sixth and final approved allocation")
    resource = history[-1]
    require(
        resource["complete"] is True
        and resource["role"] == D1_ROLE
        and resource["phase"] == "calibration"
        and resource["max_attempts"] == 6
        and resource["requested_minutes"] == 20
        and sum(row["requested_minutes"] for row in history) == 120
        and sum(row["role"] == D1_ROLE for row in history) == 1,
        "D1 completion or authorized allocation scope differs",
    )
    require(
        job == resource["job_id"],
        "D1 resource receipt is not the reviewed attempt",
    )
    auth = read_json(attempt / "authorization.json")
    require(auth["roles"] == [D1_ROLE], "D1 approval must be restricted to its diagnostic role")

    planned = read_json(attempt / "planned.json")
    config_path = ROOT / "configs/m5-performance.json"
    config = read_json(config_path)
    gates = config["gates"]
    require(
        planned["auxiliary_sha256"] == D1_PROPOSAL_SHA
        and planned["config_sha256"] == digest(config_path)
        and (
            gates["overhead_ttft_ns"],
            gates["overhead_ttft_rel"],
            gates["overhead_throughput_rel"],
        )
        == (2_000_000, 0.05, 0.05),
        "D1 proposal, config or frozen equivalence bounds differ",
    )
    commit = planned["execution_commit"]
    verify_analysis(commit, commit)
    git_files = git_identity(commit)
    node = attempt / "node"
    allocation = read_json(node / "allocation.json")
    gpu = device(allocation["gpu_before"])[0]
    index = read_json(node / "index.json")
    require(index["stage"] == D1_ROLE and index["schema_version"] == 1, "D1 index scope differs")

    blocks = proposal["diagnostic"]["blocks"]
    specs = [(block, run) for block in blocks for run in block["ordered_runs"]]
    results, raw_requests, telemetry_records = {}, {}, {}
    environments, client_pids, engine_pids = set(), set(), set()
    original_hash = proposal["diagnostic"]["input_sha256"]
    require(original_hash == D1_INPUT_SHA, "D1 expected input differs")
    previous_boundary = allocation["gpu_before"]["monotonic_ns"]
    for row, (block, spec) in zip(index["runs"], specs, strict=True):
        identity = spec["id"]
        require(
            row["id"] == identity
            and row["tier"] == spec["tier"]
            and all(
                row[key] == block[key] for key in ("scenario", "seed", "mode", "rate_rps", "count")
            )
            and row["returncode"] == 0
            and row["remaining_processes"] == [],
            f"D1 ordered execution differs for {identity}",
        )
        leaf = node / identity
        table = leaf / "input.json"
        require(digest(table) == original_hash, f"D1 input differs for {identity}")
        evidence = sealed_leaf(
            attempt,
            planned["remote_attempt"],
            row,
            table,
            config_path,
            config,
            resource["job_id"],
            git_files,
            commit,
        )
        report = audit_run(
            config_path,
            table,
            leaf / "raw",
            pair_id=identity,
            mode=block["mode"],
            phase="calibration",
        )
        require(report == read_json(leaf / "audit.json"), f"D1 audit mismatch for {identity}")
        counts = report["metrics"]["strata"]["all"]["counts"]
        require(
            counts["planned"]
            == counts["completed"]
            == counts["submitted"]
            == counts["timely_first_token"]
            == block["count"],
            f"D1 request terminal inventory differs for {identity}",
        )
        client = read_json(leaf / "raw" / "client.json")
        engine = read_json(leaf / "raw" / "engine.json")
        boundary = {}
        for point in ("before", "after"):
            path = leaf / f"diagnostic-device-{point}.json"
            key = f"diagnostic_device_{point}_sha256"
            require(digest(path) == row[key], f"D1 {point} telemetry differs for {identity}")
            boundary[point] = telemetry(path, gpu)
        phases = client["phases"]
        require(
            previous_boundary
            <= boundary["before"]["monotonic_ns"]
            < phases["load_start_ns"]
            <= phases["drain_end_ns"]
            < boundary["after"]["monotonic_ns"]
            <= allocation["gpu_after"]["monotonic_ns"],
            f"D1 device telemetry did not bracket the fresh engine for {identity}",
        )
        require(
            boundary["before"]["cpu_affinity"] == boundary["after"]["cpu_affinity"],
            f"D1 CPU affinity changed within {identity}",
        )
        previous_boundary = boundary["after"]["monotonic_ns"]
        clock = client["clock"]
        environments.add(
            (
                clock["host"],
                clock["boot_id"],
                clock["time_namespace"],
                engine["profile"]["num_gpu_blocks"],
                tuple(boundary["before"]["cpu_affinity"]),
            )
        )
        client_pids.add(clock["pid"])
        engine_pids.add(engine["clock"]["pid"])
        results[identity] = {
            "tier": spec["tier"],
            "evidence": evidence,
            "metrics": report["metrics"],
        }
        raw_requests[identity] = {request["request_id"]: request for request in client["requests"]}
        telemetry_records[identity] = boundary
    require(
        len(environments) == 1 and len(client_pids) == len(engine_pids) == len(results) == 8,
        "D1 environment varied or fresh engines were not independently identified",
    )
    node_result = read_json(node / "d1-result.json")
    require(
        node_result["status"] == "diagnostic-complete"
        and node_result["proposal_sha256"] == D1_PROPOSAL_SHA
        and node_result["input_sha256"] == original_hash
        and node_result["audited_runs"] == 8
        and node_result["retained_c3_sha256"] == C3_BLOCKED_SHA,
        "D1 runner result scope differs",
    )

    pairs = {"E/E": [], "L/L": [], "E/L": []}
    for block in blocks:
        for kind, (reference, observed) in [
            *zip(("E/E", "L/L"), block["same_tier_controls"], strict=True),
            *(("E/L", pair) for pair in block["endpoint_light_pairs"]),
        ]:
            pairs[kind].append(
                {
                    "block": block["id"],
                    "reference": reference,
                    "observed": observed,
                    **compare_pair(
                        results[reference]["metrics"], results[observed]["metrics"], gates
                    ),
                    **matched_requests(raw_requests[reference], raw_requests[observed]),
                }
            )
    return {
        "status": "diagnostic-complete",
        "proposal_sha256": D1_PROPOSAL_SHA,
        "input_sha256": original_hash,
        "retained_original_C3": {"status": "blocked", "stage_result_sha256": C3_BLOCKED_SHA},
        "D1_is_C3_qualification": False,
        "execution_commit": commit,
        "analysis_commit": commit,
        "resource_release": resource,
        "resource_ledger": history,
        "runs": results,
        "device_boundaries": telemetry_records,
        "pairs": pairs,
        "decision": qualification_decision(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(not sys.flags.optimize, "run unoptimized D1 auditor")
        report = review(args.attempt.resolve(), args.proposal.resolve())
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, default=str, allow_nan=False)
            stream.write("\n")
    except (KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"M5 D1 evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
