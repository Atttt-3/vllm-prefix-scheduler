"""Verify the frozen D4 design and stage both tables without loading a model."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from vllm_prefix_scheduler.m5_campaign import digest, read_json, require
from vllm_prefix_scheduler.m5_contract import expected_arrival_offsets, workload_gate
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA

ROOT = Path(__file__).resolve().parents[1]
ORDER = (8, 6, 6, 8)
NANOSECOND = 1_000_000_000
D3_ANALYSIS_SHA256 = "d7f18b84ac663ce7d6463cdf1ee8547bac0381b414216ba87faaa0db19b401a5"


def admission_summary(timing, capacity, tail_count):
    """Integrate exact native waiting intervals against an occupancy proxy."""
    rows = sorted(timing["per_id"].values(), key=lambda row: row["planned_ns"])
    require(
        timing["status"] == "complete"
        and timing["clock"]["verified"] is True
        and timing["planned_count"] == len(rows) >= tail_count > 0,
        "incomplete native admission timing",
    )
    events = defaultdict(lambda: [0, 0])
    waits = []
    output_envelopes = []
    for row in rows:
        first, end = row["first_components_ns"], row["end_components_ns"]
        queued = row["submit_ns"] + first["submit_to_native_queued"]
        wait = first["native_queued_to_first_scheduled"]
        scheduled = queued + wait
        native_last = (
            scheduled
            + first["native_first_scheduled_to_first_token"]
            + end["native_first_to_last_token"]
        )
        require(queued <= scheduled <= native_last, "reversed native admission interval")
        events[queued][0] += 1
        events[scheduled][0] -= 1
        events[scheduled][1] += 1
        events[native_last][1] -= 1
        waits.append(wait)
        output_envelopes.append(end["native_first_to_last_token"])
    waiting = active = peak_waiting = peak_active = total = overlap = 0
    previous = min(events)
    for at, (waiting_change, active_change) in sorted(events.items()):
        area = (at - previous) * waiting
        total += area
        if active >= capacity:
            overlap += area
        waiting += waiting_change
        active += active_change
        peak_waiting = max(peak_waiting, waiting)
        peak_active = max(peak_active, active)
        previous = at
    require(waiting == active == 0 and total == sum(waits), "native interval ledger differs")
    return {
        "tail_wait_median_ns": median(waits[-tail_count:]),
        "tail_native_output_envelope_median_ns": median(output_envelopes[-tail_count:]),
        "peak_waiting": peak_waiting,
        "peak_native_output_envelopes": peak_active,
        "wait_total_ns": total,
        "wait_overlap_with_capacity_ns": overlap,
        "wait_overlap_with_capacity_fraction": overlap / total if total else None,
    }


def prepare(plan_path, config_path, history, d3_analysis_path, output):
    plan = read_json(plan_path)
    config = read_json(config_path)
    target, budget = plan["target"], plan["budget"]
    archived = history / "attempts/c3-01/node"
    original = archived / "mixed-2-endpoint/input.json"
    c3 = archived / "calibration-result.json"
    d3_plan = ROOT / "configs/m5-d3.json"
    require(
        plan["schema_version"] == 1
        and plan["role"] == "D4"
        and plan["scope"] == "rate-sensitivity-diagnostic"
        and digest(c3) == plan["retained_C3_sha256"] == C3_BLOCKED_SHA
        and read_json(c3)["status"] == "blocked"
        and digest(d3_plan) == plan["D3_plan_sha256"]
        and digest(d3_analysis_path) == plan["D3_analysis_sha256"] == D3_ANALYSIS_SHA256,
        "D4 historical source differs",
    )
    inherited = read_json(d3_plan)
    gates = config["gates"]
    require(
        (
            gates["overhead_ttft_ns"],
            gates["overhead_ttft_rel"],
            gates["overhead_throughput_rel"],
            gates["submit_p99_ns"],
            gates["submit_max_ns"],
        )
        == (2_000_000, 0.05, 0.05, 5_000_000, 25_000_000)
        and plan["assessment"]["can_qualify_C3"] is False,
        "D4 changed original gates or qualification scope",
    )
    keys = ("scenario", "seed", "mode", "tier", "count_per_engine")
    baseline = read_json(ROOT / config["baseline_config"])
    require(
        all(target[key] == inherited["target"][key] for key in keys)
        and target["preheat_rate_rps"] == inherited["target"]["arrival_rate_rps"]
        and target["max_num_seqs"] == baseline["max_num_seqs"] == 4
        and tuple(target["measure_order_rps"]) == ORDER
        and target["same_rate_pairs"] == [["E1", "E4"], ["E2", "E3"]]
        and plan["assessment"]["tail_request_count"] == 15,
        "D4 rate contrast or runtime differs",
    )
    require(
        digest(original) == D1_INPUT_SHA,
        "D4 archived input differs",
    )
    archived_table = read_json(original)
    count = target["count_per_engine"]
    require(
        archived_table["scenario"] == target["scenario"]
        and archived_table["seed"] == target["seed"]
        and len(archived_table["requests"]) == count,
        "D4 archived table scope differs",
    )
    tables = {}
    for rate in sorted(set(ORDER)):
        offsets = expected_arrival_offsets(
            config, target["scenario"], target["seed"], count, rate, None
        )
        table = {
            **archived_table,
            "rate_rps": float(rate),
            "requests": [
                {**row, "offset_ns": offset}
                for row, offset in zip(archived_table["requests"], offsets, strict=True)
            ],
        }
        workload_gate(table, config, target, "calibration")
        raw = (json.dumps(table, separators=(",", ":")) + "\n").encode()
        require(
            target["arrival_tables"][str(rate)]
            == {"sha256": hashlib.sha256(raw).hexdigest(), "last_offset_ns": offsets[-1]},
            "D4 derived input table differs",
        )
        tables[rate] = raw
    require(tables[8] == original.read_bytes(), "D4 8 rps table must be byte-identical")
    extra_ns = ORDER.count(6) * (
        target["arrival_tables"]["6"]["last_offset_ns"]
        - target["arrival_tables"]["8"]["last_offset_ns"]
    )
    extra_s = (extra_ns + NANOSECOND - 1) // NANOSECOND
    require(
        budget["additional_allocations"] == budget["devices_per_allocation"] == 1
        and budget["gpu"] == "RTX 3090"
        and budget["serial"] is True
        and budget["request_minutes"] == 20
        and budget["retries"] == 0
        and budget["D3_estimate_s"] == inherited["budget"]["estimated_work_envelope_seconds"]
        and budget["additional_arrival_envelope_s_rounded_up"] == extra_s
        and budget["work_estimate_s"] == budget["D3_estimate_s"] + extra_s
        and budget["work_estimate_s"]
        <= budget["work_deadline_s"]
        == gates["max_allocation_work_s"],
        "D4 budget differs from frozen arrival and D3 envelopes",
    )
    analysis = read_json(d3_analysis_path)
    require(
        analysis["role"] == "D3"
        and analysis["status"] == "diagnostic-complete"
        and analysis["decision"]["thermal_control"] == "verified"
        and analysis["decision"]["endpoint_stability_under_control"] == "unstable"
        and analysis["proposal_sha256"] == plan["D3_plan_sha256"],
        "D4 D3 evidence differs",
    )
    summaries = {
        label: admission_summary(
            analysis["runs"][label]["native_timing"],
            target["max_num_seqs"],
            plan["assessment"]["tail_request_count"],
        )
        for label in ("E1", "E2", "E3", "E4")
    }
    manifest = {
        "status": "offline-validated",
        "role": plan["role"],
        "plan_sha256": digest(plan_path),
        "config_sha256": digest(config_path),
        "D3_analysis_sha256": digest(d3_analysis_path),
        "retained_C3_sha256": digest(c3),
        "tables": target["arrival_tables"],
        "D3_admission_summary": summaries,
        "budget": budget,
    }
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for rate, raw in tables.items():
        (output / f"mixed-{rate}rps.json").write_bytes(raw)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--d3-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.plan, args.config, args.history, args.d3_analysis, args.output)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"M5 D4 plan rejected: {error}\n")


if __name__ == "__main__":
    main()
