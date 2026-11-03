"""Recompute the D6 client token-gap signature on sealed D5-R evidence."""

import argparse
import json
from pathlib import Path
from statistics import median

from m5_d3_audit import p95_row, timing_leaf, timing_pair, verify_native_sources
from m5_seal import ROOT, git_identity, sealed_leaf

from vllm_prefix_scheduler.m5_campaign import digest, read_json, require
from vllm_prefix_scheduler.m5_contract import audit_run
from vllm_prefix_scheduler.m5_d5 import d5_profile
from vllm_prefix_scheduler.m5_resources import audit_attempt

BIN_NS = 500_000_000
HORIZON_NS = 3_000_000_000
MIN_INTERVALS = 10
MIN_IDS = 2
MEDIAN_GATE_NS = 30_000_000
EARLY_IDS = tuple(f"request-{number:06d}" for number in range(8, 17))
MEASURED = ("E1", "E2", "E3", "E4")
QUEUE_STEP = "native_queued_to_first_scheduled"
D5R_ANALYSIS_SHA256 = "bade7e99c3fb214f8545a1285df28a148f2062b1f6fa64d2d410aeeb55130882"


def client_gap_signature(client, *, cutoff_ns=None):
    """Assign every same-ID inter-token interval by its later client timestamp."""
    start = client["phases"]["measure_start_ns"]
    end = start + HORIZON_NS
    if cutoff_ns is None:
        cutoff_ns = end
    require(
        type(start) is int and type(cutoff_ns) is int and cutoff_ns >= start,
        "D6 observation cutoff differs",
    )
    gaps = [[] for _ in range(HORIZON_NS // BIN_NS)]
    request_ids = [set() for _ in gaps]
    seen = set()
    coalesced = 0
    for row in client["requests"]:
        identity = row["request_id"]
        require(
            type(identity) is str and identity not in seen, "D6 duplicate or invalid request ID"
        )
        seen.add(identity)
        previous = None
        for chunk in row["chunk_tokens"]:
            stamp, tokens = chunk["at_ns"], chunk["token_ids"]
            require(
                type(stamp) is int
                and stamp >= start
                and (previous is None or stamp >= previous)
                and type(tokens) is list
                and tokens
                and all(type(token) is int and token >= 0 for token in tokens),
                "D6 client chunk is not an ordered real token arrival",
            )
            offset = stamp - start
            if stamp < cutoff_ns and offset < HORIZON_NS:
                index = offset // BIN_NS
                if previous is not None:
                    gaps[index].append(stamp - previous)
                    request_ids[index].add(identity)
                if len(tokens) > 1:
                    gaps[index].extend([0] * (len(tokens) - 1))
                    request_ids[index].add(identity)
                    coalesced += len(tokens) - 1
            previous = stamp

    bins = []
    for index, (intervals, ids) in enumerate(zip(gaps, request_ids, strict=True)):
        bin_end = (index + 1) * BIN_NS
        complete = start + bin_end <= cutoff_ns
        midpoint = median(intervals) if intervals else None
        eligible = complete and len(intervals) >= MIN_INTERVALS and len(ids) >= MIN_IDS
        bins.append(
            {
                "index": index,
                "start_offset_ns": index * BIN_NS,
                "end_offset_ns": bin_end,
                "complete": complete,
                "interval_count": len(intervals),
                "distinct_request_count": len(ids),
                "median_ns": midpoint,
                "eligible": eligible,
                "above_threshold": eligible and midpoint >= MEDIAN_GATE_NS,
            }
        )
    supported = False
    trigger = None
    for left, right in zip(bins, bins[1:], strict=False):
        supported |= left["eligible"] and right["eligible"]
        if left["above_threshold"] and right["above_threshold"]:
            trigger = [left["index"], right["index"]]
            break
    return {
        "candidate": "detected"
        if trigger
        else "not_detected"
        if cutoff_ns >= end and supported
        else "unverified",
        "first_trigger_bin_indices": trigger,
        "first_trigger_offset_ns": bins[trigger[-1]]["end_offset_ns"] if trigger else None,
        "observed_through_offset_ns": min(cutoff_ns, end) - start,
        "coalesced_zero_interval_count": coalesced,
        "bins": bins,
    }


def signature(client, engine, workload, gates, *, leaf_end_ns, cutoff_ns=None):
    return {
        **client_gap_signature(client, cutoff_ns=cutoff_ns),
        "measure_start_ns": client["phases"]["measure_start_ns"],
        "timing": timing_leaf(client, engine, workload, gates, leaf_end_ns=leaf_end_ns),
    }


def request_chronology(row, start):
    first, end = row["first_components_ns"], row["end_components_ns"]
    require(first is not None and end is not None, "D6 archived early request timing is incomplete")
    queued = row["submit_ns"] + first["submit_to_native_queued"]
    scheduled = queued + first[QUEUE_STEP]
    native_first = scheduled + first["native_first_scheduled_to_first_token"]
    return {
        "planned_offset_ns": row["planned_ns"] - start,
        "submit_offset_ns": row["submit_ns"] - start,
        "native_queued_offset_ns": queued - start,
        "native_first_scheduled_offset_ns": scheduled - start,
        "native_first_token_offset_ns": native_first - start,
        "client_first_token_offset_ns": row["first_token_ns"] - start,
        "native_last_token_offset_ns": native_first + end["native_first_to_last_token"] - start,
        "terminal_offset_ns": row["end_ns"] - start,
        "ttft_ns": row["ttft_ns"],
        "first_components_ns": first,
    }


def summarize_signature(runs):
    """Contrast archived E1/E4 p95 anchors and the prespecified early request IDs."""
    require(tuple(runs) == MEASURED, "D6 archival measured leaf inventory differs")
    e1, e4 = (runs[key]["timing"] for key in ("E1", "E4"))
    pairing = timing_pair(e1, e4)
    return {
        "p95_by_leaf": {
            name: dict(
                zip(("ttft_ns", "request_id"), p95_row(run["timing"]["per_id"]), strict=True)
            )
            for name, run in runs.items()
        },
        "candidate_by_leaf": {name: run["candidate"] for name, run in runs.items()},
        "E1_E4": {
            "orientation": "E4 minus E1",
            "p95_anchors": pairing["p95_anchors"],
            "early_same_id": {
                identity: {
                    "E1": request_chronology(
                        e1["per_id"][identity], runs["E1"]["measure_start_ns"]
                    ),
                    "E4": request_chronology(
                        e4["per_id"][identity], runs["E4"]["measure_start_ns"]
                    ),
                    "delta": pairing["per_id"][identity],
                }
                for identity in EARLY_IDS
            },
        },
    }


def stage_times(index, thermal, client):
    """Split the engine envelope using the clock-bound raw timestamps."""
    start, end = index["started_monotonic_ns"], index["ended_monotonic_ns"]
    phases = client["phases"]
    load_start, load_end = phases["load_start_ns"], phases["load_end_ns"]
    heat_start, heat_end = thermal["conditioning_start_ns"], thermal["conditioning_end_ns"]
    require(
        start <= load_start <= load_end <= heat_start <= heat_end <= end, "D6 stage order differs"
    )
    t0, drain = phases.get("measure_start_ns"), phases.get("drain_end_ns")
    post_heat = measure_drain = None
    if t0 is not None:
        require(heat_end <= t0 <= drain <= end, "D6 measured stage order differs")
        post_heat, measure_drain = t0 - heat_end, drain - t0
    return {
        "engine_envelope_ns": end - start,
        "load_ns": load_end - load_start,
        "conditioning_ns": heat_end - heat_start,
        "non_conditioning_ns": end - start - (heat_end - heat_start),
        "post_conditioning_to_t0_ns": post_heat,
        "measure_and_drain_ns": measure_drain,
        "after_drain_ns": end - drain if drain is not None else None,
    }


def historical_signature(attempt, analysis_path):
    require(digest(analysis_path) == D5R_ANALYSIS_SHA256, "D6 retained D5-R analysis differs")
    analysis = read_json(analysis_path)
    planned = read_json(attempt / "planned.json")
    resource = audit_attempt(attempt, require_completed=True)
    require(
        resource["role"] == planned["role"] == analysis["role"] == "D5-R"
        and json.loads(json.dumps(resource, default=str)) == analysis["resource_release"]
        and planned["execution_commit"] == analysis["execution_commit"]
        and analysis["status"] == "diagnostic-complete"
        and analysis["d5_assessment"]["request_evidence_complete"] is True,
        "D6 D5-R allocation is detached from the retained complete review",
    )
    execution = planned["execution_commit"]
    git_files = git_identity(execution)
    config_path = ROOT / "configs/m5-performance.json"
    d3_path, d5_path, d5r_path = (
        ROOT / "configs" / f"m5-{role}.json" for role in ("d3", "d5", "d5r")
    )
    require(
        digest(config_path) == planned["config_sha256"] == git_files["configs/m5-performance.json"]
        and digest(d3_path) == analysis["inherited_d3_proposal_sha256"]
        and digest(d5_path) == analysis["inherited_d5_proposal_sha256"]
        and digest(d5r_path) == planned["auxiliary_sha256"] == analysis["proposal_sha256"],
        "D6 D5-R execution configuration differs",
    )
    config, d3, d5 = map(read_json, (config_path, d3_path, d5_path))
    node = attempt / "node"
    index = read_json(node / "index.json")
    rows = index["runs"]
    require(
        index["stage"] == "D5-R" and [row["id"] for row in rows] == ["thermal-pilot", *MEASURED],
        "D6 retained D5-R leaf order differs",
    )
    runs, sources, stages = {}, {}, {}
    for row in rows:
        name = row["id"]
        leaf = node / name
        table, raw = leaf / "input.json", leaf / "raw"
        binding = sealed_leaf(
            attempt,
            planned["remote_attempt"],
            row,
            table,
            config_path,
            config,
            resource["job_id"],
            git_files,
            execution,
            d5_expected_profile=d5_profile(d5, name),
        )
        old = analysis["runs"][name]
        require(
            all(
                binding[key] == old["evidence"][key]
                for key in ("raw_hashes", "input_sha256", "collector_sha256")
            )
            and binding["input_sha256"] == analysis["input_sha256"],
            "D6 raw or collector bytes differ from the reviewed D5-R originals",
        )
        thermal_path = raw / "d2-thermal.json"
        require(
            digest(thermal_path) == row["d2_thermal_sha256"] == old["thermal_artifact_sha256"],
            "D6 D5-R thermal source differs",
        )
        client, engine, workload, source, thermal = map(
            read_json,
            (raw / "client.json", raw / "engine.json", table, raw / "sources.json", thermal_path),
        )
        verify_native_sources(d3, source)
        sources[name] = {
            **binding["raw_hashes"],
            "collector_sha256": binding["collector_sha256"],
            "thermal_sha256": digest(thermal_path),
        }
        stages[name] = stage_times(row, thermal, client)
        if name == "thermal-pilot":
            continue
        metrics = audit_run(
            config_path, table, raw, pair_id=name, mode=row["mode"], phase="calibration"
        )
        require(
            metrics == read_json(leaf / "audit.json") and metrics["metrics"] == old["metrics"],
            "D6 endpoint metric binding differs",
        )
        cutoff = client["phases"]["measure_start_ns"] + HORIZON_NS
        require(
            cutoff <= client["phases"]["drain_end_ns"],
            "D6 archival observation ends before its full signature horizon",
        )
        runs[name] = signature(
            client,
            engine,
            workload,
            config["gates"],
            leaf_end_ns=row["ended_monotonic_ns"],
            cutoff_ns=cutoff,
        )
        require(
            runs[name]["timing"] == old["native_timing"]
            and runs[name]["timing"]["status"] == "complete"
            and runs[name]["timing"]["clock"]["verified"] is True
            and all(
                req["status"] == "completed" for req in runs[name]["timing"]["per_id"].values()
            ),
            "D6 archived native-clock proof differs or is incomplete",
        )
    return {
        "schema_version": 1,
        "scope": "sealed D5-R historical signature replay",
        "historical_analysis_sha256": D5R_ANALYSIS_SHA256,
        "execution_commit": execution,
        "resource_release": resource,
        "source_hashes_by_leaf": sources,
        "native_timing_source_sha256": d3["native_timing"]["source_sha256"],
        "source_binding": {
            "config_sha256": digest(config_path),
            "input_sha256": analysis["input_sha256"],
            "index_sha256": digest(node / "index.json"),
        },
        "bin_contract": {
            "horizon_ns": HORIZON_NS,
            "bin_ns": BIN_NS,
            "min_intervals": MIN_INTERVALS,
            "min_distinct_requests": MIN_IDS,
            "median_gate_ns": MEDIAN_GATE_NS,
            "assignment": "Later real client token timestamp, left closed and right open. Coalesced real tokens have zero client-observed spacing.",
        },
        "stage_times_by_leaf": stages,
        "runs": runs,
        "summary": summarize_signature(runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis or args.attempt.parent.parent / "d5-r-review/analysis.json"
    try:
        report = historical_signature(args.attempt, analysis)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False, default=str)
            stream.write("\n")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
