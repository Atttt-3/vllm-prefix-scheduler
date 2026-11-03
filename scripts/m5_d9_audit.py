"""Reconstruct bounded D9 host evidence on the original four-runner baseline."""

from collections import defaultdict
from itertools import combinations
from statistics import median

from check_m5_d1 import compare_pair
from m5_d6_audit import client_tokens, distinct_batch_pairs, ordered, whole
from m5_d6_signature import BIN_NS, MEDIAN_GATE_NS, MIN_IDS, MIN_INTERVALS
from m5_d7_audit import (
    ENDPOINTS,
    MEASURED,
    STEPS,
    append_packet,
    audit_d7_trace,
    comparable_gaps,
    later_wait_worsening,
    observe_slots,
    repeated_boundaries,
    slower_gap,
    waiting_summary,
)

from vllm_prefix_scheduler.m5_campaign import digest, read_json, require

EARLY = (0, 3_000_000_000)
LATE = (6_000_000_000, 8_500_000_000)
WINDOWS = {"early": EARLY, "late": LATE}
PRIMARY_PAIRS = (("E1", "E4"), ("E2", "E3"), ("E1", "E2"), ("E4", "E3"))
D8_PAIRS = (("E1", "E2"), ("E3", "E4"), ("E1", "E4"))
ALL_PAIRS = (*PRIMARY_PAIRS, ("E3", "E4"), ("E1", "E3"), ("E4", "E2"))


def packet_rows(window):
    rows = list(window["preroll_tokens"])
    for batch in [*window["carry_in"], *window["batches"]]:
        rows.extend(
            {
                "request_id": row["request_id"],
                "generation": row["generation"],
                "ordinal": row["ordinal"],
                "token_id": row["token_ids"][0],
                "packet_ns": row["packet_ns"],
            }
            for row in batch["output_counts"]
        )
    return rows


def last_native_step(window):
    return max([window["boundary"]["at_ns"], *(row["end_ns"] for row in window["steps"])])


def audit_d9_trace(trace, client, limits, *, leaf_end_ns):
    """Verify both windows and a single continuous request and packet lineage."""
    status = trace["status"]
    require(
        type(trace["schema_version"]) is int
        and trace["schema_version"] == 1
        and status in {"complete", "incomplete", "not-started"}
        and client.get("d9_capture") == "step",
        "D9 trace schema, status or capture differs",
    )
    if status != "complete":
        partial = (
            set(trace) == {"schema_version", "status"}
            and "d9_arm_return_ns" not in client
            and "measure_start_ns" not in client["phases"]
        )
        require(
            partial
            if status == "not-started"
            else trace.get("overflow") is True or bool(trace.get("errors")),
            "D9 partial trace lacks matching arm/failure evidence",
        )
        return {"verified": False, "status": status, "issues": trace.get("errors", [])}

    t0, armed, exported = (trace[key] for key in ("t0_ns", "armed_ns", "export_ns"))
    phases = client["phases"]
    drain = phases["drain_end_ns"]
    drained = trace["drained"]
    require(
        t0 == phases["measure_start_ns"]
        and ordered(max(t0 + LATE[1], drain), exported, leaf_end_ns)
        and trace["overflow"] is False
        and trace["errors"] == []
        and set(drained) == {"pending", "native_fifo", "sampling", "has_requests"}
        and all(
            whole(drained[key]) and drained[key] == 0 for key in set(drained) - {"has_requests"}
        )
        and drained["has_requests"] is False,
        "D9 native drain or export observation horizon differs",
    )
    early, late = trace["windows"]
    require(
        all(
            window["status"] == status
            and window["t0_ns"] == t0
            and window["armed_ns"] == armed
            and window["export_ns"] == exported
            for window in (early, late)
        )
        and not early["preroll_tokens"]
        and not early["carry_in"]
        and all(type(trace[key]) is list for key in ("gap_tokens", "tail_tokens")),
        "D9 windows do not share a single idle arm and export",
    )
    audited = {
        name: audit_d7_trace(
            window,
            client,
            {**limits, "start_offset_ns": start, "window_ns": end - start},
            leaf_end_ns=leaf_end_ns,
            window=(start, end),
            role="d9",
        )
        for (name, (start, end)), window in zip(WINDOWS.items(), (early, late), strict=True)
    }
    gap, tail = trace["gap_tokens"], trace["tail_tokens"]
    early_rows = packet_rows(early)
    early_end, late_end = (last_native_step(window) for window in (early, late))
    late_start = late["boundary"]["at_ns"]
    require(
        early_end <= late_start
        and late["enqueues"][: len(early["enqueues"])] == early["enqueues"]
        and all(max(early_end, early["end_ns"]) <= row["packet_ns"] <= late_start for row in gap)
        and all(max(late_end, late["end_ns"]) <= row["packet_ns"] <= drain for row in tail)
        and late["preroll_tokens"] == [*early_rows, *gap],
        "D9 gap, tail or cross-window ordinal inventory differs",
    )
    events = (
        sum(
            len(window[key])
            for window in (early, late)
            for key in ("batches", "carry_in", "input_polls", "loops", "steps")
        )
        + len(late["enqueues"])
        + len(gap)
        + len(tail)
    )
    batches = sum(len(window["batches"]) + len(window["carry_in"]) for window in (early, late))
    require(
        events <= limits["event_limit"] and batches <= limits["batch_limit"],
        "D9 aggregate bounded trace inventories differ",
    )
    rows = {row["request_id"]: row for row in client["requests"]}
    tokens = {identity: client_tokens(row) for identity, row in rows.items()}
    generations = {row["request_id"]: row["generation"] for row in late["enqueues"]}
    all_packets = [*packet_rows(late), *tail]
    require(
        len(client["requests"]) == len(rows) == len(generations) == 60
        and all(row["status"] == "completed" for row in rows.values())
        and all(len(stamps) == 16 for stamps in tokens.values())
        and len(all_packets) == 960,
        "D9 complete observation lacks the frozen request and real-token denominator",
    )
    links = defaultdict(list)
    previous = t0
    for row in all_packets:
        stamp = row["packet_ns"]
        require(ordered(previous, stamp, drain), "D9 packet chronology differs across windows")
        append_packet(
            links,
            rows,
            tokens,
            generations,
            identity=row["request_id"],
            generation=row["generation"],
            ordinal=row["ordinal"],
            packet_ns=stamp,
            token=row["token_id"],
            batch_id=None,
        )
        previous = stamp
    require(
        links.keys() == rows.keys() and all(len(chain) == 16 for chain in links.values()),
        "D9 cross-window packet inventory does not close",
    )
    return {
        "verified": True,
        "status": status,
        "observed_through_offset_ns": exported - t0,
        "aggregate_events": events,
        "aggregate_batches": batches,
        "gap_packets": len(gap),
        "tail_packets": len(tail),
        "native_real_tokens": len(all_packets),
        "windows": audited,
    }


def signature(client, bounds, *, observation_end_ns=None):
    """Assign gaps to the later token receipt in fixed half-open bins."""
    t0 = client["phases"]["measure_start_ns"]
    drain = client["phases"]["drain_end_ns"]
    if observation_end_ns is not None:
        require(ordered(drain, observation_end_ns), "D9 observation end precedes actual drain")
    observed = (observation_end_ns if observation_end_ns is not None else drain) - t0
    lo, hi = bounds
    samples = [[] for _ in range((hi - lo) // BIN_NS)]
    for row in client["requests"]:
        stamps = [stamp for stamp, _ in client_tokens(row)]
        for before, after in zip(stamps, stamps[1:], strict=False):
            require(ordered(t0, before, after, drain), "D9 client packet chronology differs")
            if lo <= after - t0 < hi:
                samples[(after - t0 - lo) // BIN_NS].append((row["request_id"], after - before))
    bins = []
    for index, values in enumerate(samples):
        left = lo + index * BIN_NS
        right = left + BIN_NS
        count = len({identity for identity, _ in values})
        gap = median(value for _, value in values) if values else None
        eligible = right <= observed and len(values) >= MIN_INTERVALS and count >= MIN_IDS
        bins.append(
            {
                "start_offset_ns": left,
                "end_offset_ns": right,
                "complete": right <= observed,
                "interval_count": len(values),
                "request_count": count,
                "eligible": eligible,
                "median_gap_ns": gap,
                "above_threshold": eligible and gap >= MEDIAN_GATE_NS,
            }
        )
    starts = [
        left["start_offset_ns"]
        for left, right in zip(bins, bins[1:], strict=False)
        if left["above_threshold"] and right["above_threshold"]
    ]
    comparable = any(
        left["eligible"] and right["eligible"] for left, right in zip(bins, bins[1:], strict=False)
    )
    return {
        "candidate": "detected" if starts else "not_detected" if comparable else "unverified",
        "trigger_starts_offset_ns": starts,
        "bins": bins,
    }


def phase_candidates(rows):
    """Separate changed shapes from comparable host-boundary changes."""
    contrasted = slower_gap(rows)
    host = repeated_boundaries(contrasted)
    shape = distinct_batch_pairs([row for row in contrasted if not row["same_shape"]])
    controls = distinct_batch_pairs(
        [
            row
            for row in rows
            if row["matched_state"]
            and row["same_shape"]
            and row["same_composition"]
            and max(row["fast_client_gap_ns"], row["slow_client_gap_ns"]) < MEDIAN_GATE_NS
        ]
    )
    return {
        "host_boundaries": host,
        "batch_composition": {
            "status": "associated" if len(shape) >= 2 and len(controls) >= 2 else "unresolved",
            "changed_shape_slow_gaps": shape,
            "same_shape_fast_controls": controls,
        },
    }


def classify_window(name, signatures, runs, traces, *, comparable):
    """Retain a descriptive backtrace even when the equivalence gate fails."""
    report = {"status": "unresolved", "reason": "required_comparison_unavailable"}
    if any(signatures[leaf] is None for leaf in MEASURED):
        return report
    endpoint_starts = {
        stamp for leaf in ENDPOINTS for stamp in signatures[leaf]["trigger_starts_offset_ns"]
    }
    if not endpoint_starts:
        report["reason"] = "endpoint_signature_not_reproduced"
        return report
    slow = [leaf for leaf in STEPS if signatures[leaf]["candidate"] == "detected"]
    fast = [leaf for leaf in STEPS if signatures[leaf]["candidate"] == "not_detected"]
    if (
        len(slow) != 1
        or len(fast) != 1
        or not all(traces.get(leaf, {}).get("verified") for leaf in STEPS)
    ):
        report["reason"] = "verified_stepped_fast_slow_contrast_absent"
        return report
    slow, fast = slow[0], fast[0]
    matching = endpoint_starts.intersection(signatures[slow]["trigger_starts_offset_ns"])
    if not matching:
        report["reason"] = "stepped_and_endpoint_slow_bins_do_not_overlap"
        return report
    start = min(matching)
    fast_bins = {row["start_offset_ns"]: row for row in signatures[fast]["bins"]}
    if not all(
        fast_bins[stamp]["eligible"] and not fast_bins[stamp]["above_threshold"]
        for stamp in (start, start + BIN_NS)
    ):
        report["reason"] = "fast_reference_bins_not_eligible_or_not_fast"
        return report
    lo, hi = max(WINDOWS[name][0], start - BIN_NS), start + 2 * BIN_NS
    rows, excluded = comparable_gaps(
        traces[fast]["windows"][name],
        traces[slow]["windows"][name],
        runs[fast]["native_timing"]["per_id"],
        runs[slow]["native_timing"]["per_id"],
        lo,
        hi,
    )
    # The fast token must also have arrived within this detailed window.
    first, last = WINDOWS[name]
    fast_trace = traces[fast]["windows"][name]
    rows = [
        row
        for row in rows
        if first
        <= fast_trace["links"][row["request_id"]][row["generated_token_ordinal"] - 1]["client_ns"]
        - fast_trace["window_start_ns"]
        < last
    ]
    candidates = phase_candidates(rows)
    report.update(
        fast=fast,
        slow=slow,
        slow_start_offset_ns=start,
        left_edge_incomplete_comparisons=excluded,
        comparisons=rows,
        **candidates,
    )
    if not comparable:
        report["reason"] = "capture_equivalence_or_thermal_comparability_unverified"
    elif len(candidates["host_boundaries"]) != 1:
        report["reason"] = "no_unique_repeated_comparable_host_boundary"
    else:
        category, boundary = next(iter(candidates["host_boundaries"].items()))
        report.update(status="host_association", reason=category, host_boundary=boundary)
    return report


def slot_holds(trace, timing):
    """Bind first admission to the same request's observed native release."""
    t0 = trace["window_start_ns"]
    observations = [*trace["batches"].values(), *([trace["carry"]] if trace["carry"] else [])]
    result = {}
    for batch in observations:
        before = {identity for identity, _ in batch["consume_before"]["running"]}
        after = {identity for identity, _ in batch["consume_after"]["running"]}
        generated = {row["request_id"] for row in batch["output_counts"] if row["ordinal"] == 16}
        for identity in (before - after) & generated:
            req = timing["per_id"][identity]
            if req["native_preemptions"] != 0 or identity in result:
                continue
            first = req["first_components_ns"]
            admitted = (
                req["submit_ns"]
                + first["submit_to_native_queued"]
                + first["native_queued_to_first_scheduled"]
            )
            lower, upper = batch["consume_start_ns"], batch["consume_end_ns"]
            if ordered(admitted, lower, upper):
                result[identity] = {
                    "admit_offset_ns": admitted - t0,
                    "release_offset_ns": upper - t0,
                    "release_batch_id": batch["batch_id"],
                    "duration_lower_ns": lower - admitted,
                    "duration_upper_ns": upper - admitted,
                }
    return result


def queue_chain(window, traces, runs, waits, *, late):
    """Require directly observed release-to-admit and, in the late case, longer hold."""
    if window["status"] != "host_association":
        return {"status": "unresolved", "reason": window["reason"]}
    slow, fast = window["slow"], window["fast"]
    label = "late" if late else "early"
    slow_trace, fast_trace = (traces[leaf]["windows"][label] for leaf in (slow, fast))
    slow_slots, fast_slots = observe_slots(slow_trace), observe_slots(fast_trace)
    host = window["host_boundary"]
    boundary = host["boundary"]
    linked = {row["slow_batch_id"] for row in host["linked_gaps"]}
    # Test the end of the measured stage, not merely its start.
    end_key = {
        "executor_call": "execute_end_ns",
        "sampling_call": "sample_end_ns",
        "native_materialization_host_envelope": "materialize_end_ns",
        "fifo_result_after_materialization": "future_result_end_ns",
        "fifo_result_to_consume": "consume_start_ns",
        "scheduler_consumption_to_packet": "consume_end_ns",
    }[boundary]
    onset = waits[slow]["first_continuous_wait_ge_500ms"]
    if onset is None:
        return {"status": "unresolved", "reason": "sustained_queue_unobserved"}
    onset = onset["onset_offset_ns"]
    end_offsets = {
        batch_id: slow_trace["batches"][str(batch_id)][end_key] - slow_trace["window_start_ns"]
        for batch_id in linked
    }
    if not late:
        first = waits[slow]["first_wait_ge_50ms"]
        direct = next(
            (
                row
                for row in slow_slots["release_to_admit"]
                if first and row["request_id"] == first["request_id"]
            ),
            None,
        )
        if (
            first
            and direct
            and onset < EARLY[1]
            and max(end_offsets.values()) < min(onset, first["queued_offset_ns"])
            and any(
                row["running"] == 4
                and first["request_id"] in row["waiting_ids"]
                and first["queued_offset_ns"] <= row["offset_ns"] < first["scheduled_offset_ns"]
                for row in slow_slots["samples"]
            )
        ):
            return {"status": "associated", "host_boundary": host, "release_to_admit": direct}
        return {"status": "unresolved", "reason": "no_direct_early_chain_before_queue_onset"}
    eligible = {batch for batch, end in end_offsets.items() if onset < end}
    if not eligible:
        return {"status": "unresolved", "reason": "no_verified_existing_queue_before_late_stage"}
    anchor = min(end_offsets[batch] for batch in eligible)
    worsening = later_wait_worsening(anchor, slow_slots, fast_slots)
    slow_holds, fast_holds = (
        slot_holds(trace, runs[leaf]["native_timing"])
        for trace, leaf in ((slow_trace, slow), (fast_trace, fast))
    )
    fast_handoffs = {row["request_id"]: row for row in fast_slots["release_to_admit"]}
    matches = []
    if worsening:
        worsened_at = worsening["first_deeper_full_queue"]["offset_ns"]
        affected_ids = {
            row["request_id"] for row in host["linked_gaps"] if row["slow_batch_id"] in eligible
        }
        for row in slow_slots["release_to_admit"]:
            ref = fast_handoffs.get(row["request_id"])
            if not ref or not (
                worsened_at <= row["release_offset_ns"]
                and row["enqueue_to_admit_ns"] > ref["enqueue_to_admit_ns"]
            ):
                continue
            for released in affected_ids.intersection(row["released_ids"]):
                a, b = fast_holds.get(released), slow_holds.get(released)
                if (
                    a
                    and b
                    and a["release_batch_id"] == ref["release_batch_id"]
                    and b["release_batch_id"] == row["release_batch_id"]
                    and b["duration_lower_ns"] > a["duration_upper_ns"]
                ):
                    matches.append(
                        {
                            "held_request_id": released,
                            "fast_hold": a,
                            "slow_hold": b,
                            "fast_handoff": ref,
                            "slow_handoff": row,
                        }
                    )
    return {
        "status": "associated" if matches else "unresolved",
        "reason": "matched_longer_hold_release_and_wait"
        if matches
        else "no_matched_later_slot_chain",
        "later_queue_worsening": worsening,
        "matched_handoffs": matches,
    }


def assess_d9(runs, clients, traces, gates, *, heat_verified):
    """Apply frozen ABBA margins independently of the diagnostic verdicts."""
    pairs = {}
    for first, second in ALL_PAIRS:
        a, b = runs.get(first, {}), runs.get(second, {})
        pairs[f"{first}/{second}"] = (
            compare_pair(a["metrics"], b["metrics"], gates)
            if a.get("endpoint_evidence_complete") and b.get("endpoint_evidence_complete")
            else None
        )
    traces_verified = all(traces.get(leaf, {}).get("verified") for leaf in STEPS)
    primary = [pairs[f"{a}/{b}"] for a, b in PRIMARY_PAIRS]
    equivalent = (
        "unverified"
        if not (heat_verified and traces_verified and all(primary))
        else "supported"
        if all(row["within_tolerance"] for row in primary)
        else "outside_tolerance"
    )
    waits = {
        leaf: waiting_summary(clients[leaf], runs.get(leaf, {}).get("native_timing", {}))
        for leaf in clients
    }
    signatures = {
        name: {
            leaf: signature(
                clients[leaf],
                bounds,
                observation_end_ns=(
                    clients[leaf]["phases"]["measure_start_ns"]
                    + traces[leaf]["observed_through_offset_ns"]
                )
                if traces.get(leaf, {}).get("verified")
                else None,
            )
            if leaf in clients
            else None
            for leaf in MEASURED
        }
        for name, bounds in WINDOWS.items()
    }
    windows = {
        name: classify_window(name, rows, runs, traces, comparable=equivalent == "supported")
        for name, rows in signatures.items()
    }
    formation = queue_chain(windows["early"], traces, runs, waits, late=False)
    amplification = queue_chain(windows["late"], traces, runs, waits, late=True)
    return {
        "capture_equivalence": equivalent,
        "primary_pairs": PRIMARY_PAIRS,
        "d8_pair_directions": D8_PAIRS,
        "pairs": pairs,
        "signatures": signatures,
        "waiting": waits,
        "windows": windows,
        "backlog_formation": formation,
        "late_amplification": amplification,
        "scope": "Host timing envelopes. Device execution and exact IPC publication are unobserved.",
    }


def d9_report(
    node,
    plan,
    thermal_plan,
    proposal_sha,
    commit,
    resource,
    history,
    samples_path,
    target,
    runs,
    clients,
    traces,
    gates,
    compare_thermal,
):
    """Bind result claims to independently audited leaves and all six thermal pairs."""
    lineage = {
        "role": "D9",
        "proposal_sha256": proposal_sha,
        "inherited_d8_proposal_sha256": plan["retained_D8"]["proposal_sha256"],
        "inherited_d2r110_proposal_sha256": plan["D2R110_plan_sha256"],
        "retained_d8_analysis_sha256": plan["retained_D8"]["analysis_sha256"],
        "retained_d7_analysis_sha256": thermal_plan["retained_D7"]["analysis_sha256"],
        "input_sha256": plan["target"]["input_sha256"],
        "retained_c3_sha256": plan["retained_C3"]["sha256"],
    }
    leaf_ok = {
        identity: runs.get(identity, {}).get("endpoint_evidence_complete") is True
        and (identity not in STEPS or traces.get(identity, {}).get("verified") is True)
        for identity in MEASURED
    }
    for first, second in zip(MEASURED, MEASURED[1:], strict=False):
        require(
            second not in runs or leaf_ok[first],
            "D9 continued after independently incomplete endpoint or dual-window evidence",
        )
    result_path = node / "d9-result.json"
    if result_path.exists():
        require(
            read_json(result_path)
            == {
                **lineage,
                "status": "diagnostic-complete"
                if all(leaf_ok.values())
                else "diagnostic-incomplete",
                "audited_runs": sum("metrics" in run for run in runs.values()),
            },
            "D9 node result differs from independent evidence",
        )
    elif resource["complete"]:
        raise ValueError("Completed D9 lacks the node result")
    limits = thermal_plan["conditioning"]
    thermal_pairs = [
        {
            "reference": first,
            "observed": second,
            "thermal": compare_thermal(runs[first]["in_window"], runs[second]["in_window"], limits)
            if all("in_window" in runs.get(key, {}) for key in (first, second))
            else None,
        }
        for first, second in combinations(MEASURED, 2)
    ]
    heat_verified = (
        resource["complete"]
        and target is not None
        and all(
            runs.get(identity, {}).get("pre_t0_verified") is True
            and runs[identity]["prelude"]["verified"]
            for identity in MEASURED
        )
        and all(pair["thermal"] and pair["thermal"]["comparable"] for pair in thermal_pairs)
    )
    assessment = assess_d9(runs, clients, traces, gates, heat_verified=heat_verified)
    for identity, trace in traces.items():
        runs[identity]["step_trace"] = trace
    return {
        **lineage,
        "status": "diagnostic-complete"
        if heat_verified and all(leaf_ok.values())
        else "diagnostic-incomplete",
        "retained_original_C3": {
            "status": "blocked",
            "stage_result_sha256": lineage["retained_c3_sha256"],
        },
        "retained_history_sha256": plan["history_sha256"],
        "execution_commit": commit,
        "analysis_commit": commit,
        "resource_release": resource,
        "resource_ledger": history,
        "device_journal_sha256": digest(samples_path) if samples_path.exists() else None,
        "verified_thermal_target": target,
        "runs": runs,
        "pairs": thermal_pairs,
        "d9_assessment": assessment,
        "decision": {
            "thermal_control": "verified" if heat_verified else "unverified",
            "capture_equivalence": assessment["capture_equivalence"],
            "backlog_formation": assessment["backlog_formation"]["status"],
            "late_amplification": assessment["late_amplification"]["status"],
            "qualification_started": False,
        },
    }
