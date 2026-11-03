"""Independently reconstruct the D7 late-output and queue evidence."""

from collections import defaultdict, deque
from statistics import median

from check_m5_d1 import compare_pair
from m5_d3_audit import timing_pair
from m5_d6_audit import (
    AFTER_READY,
    BOUNDARY_START,
    EXECUTION,
    STATE_QUEUES,
    batch_packets,
    check_snapshots,
    checked_requests,
    client_tokens,
    describe_pair,
    distinct_batch_pairs,
    engine_loops,
    enqueued_requests,
    ordered,
    packet_path,
    verify_step_sources,
    whole,
)
from m5_d6_signature import BIN_NS, MEDIAN_GATE_NS, MIN_IDS, MIN_INTERVALS

from vllm_prefix_scheduler.m5_campaign import require

SECOND = 1_000_000_000
TRACE_START_NS = 3_500_000_000
TRACE_END_NS = 7_500_000_000
WAIT_THRESHOLD_NS = 50_000_000
SUSTAINED_WAIT_NS = 500_000_000
ENDPOINTS = ("E1", "E4")
STEPS = ("E2", "E3")
MEASURED = ("E1", "E2", "E3", "E4")
QUEUE_COMPONENT = "native_queued_to_first_scheduled"
CLOCK_ARRIVAL_COMPONENT = "submit_to_native_queued"
MECHANISMS = (
    "batch_composition_association",
    "execution_or_materialization_slowdown",
    "engine_progress_pause",
    "full_slot_amplification",
)


def verify_d7_sources(plan, source):
    verify_step_sources(plan, source)


def late_signature(client):
    """Count same-request real-token gaps by their later arrival."""
    t0 = client["phases"]["measure_start_ns"]
    observed = client["phases"]["drain_end_ns"] - t0
    n_bins = (TRACE_END_NS - TRACE_START_NS) // BIN_NS
    samples = [[] for _ in range(n_bins)]
    seen = set()
    for row in client["requests"]:
        identity = row["request_id"]
        require(type(identity) is str and identity not in seen, "D7 request IDs differ")
        seen.add(identity)
        previous = t0
        have_previous = False
        for chunk in row["chunk_tokens"]:
            at = chunk["at_ns"]
            tokens = chunk["token_ids"]
            require(
                whole(at)
                and previous <= at <= t0 + observed
                and type(tokens) is list
                and tokens
                and all(whole(token) for token in tokens),
                "D7 real client token chunks are malformed",
            )
            offset = at - t0
            if TRACE_START_NS <= offset < TRACE_END_NS:
                index = (offset - TRACE_START_NS) // BIN_NS
                if have_previous:
                    samples[index].append((identity, at - previous))
                samples[index].extend((identity, 0) for _ in tokens[1:])
            previous, have_previous = at, True
    bins = []
    for index, gaps in enumerate(samples):
        lo, hi = TRACE_START_NS + index * BIN_NS, TRACE_START_NS + (index + 1) * BIN_NS
        n_ids = len({identity for identity, _ in gaps})
        midpoint = median(value for _, value in gaps) if gaps else None
        complete = hi <= observed
        eligible = complete and len(gaps) >= MIN_INTERVALS and n_ids >= MIN_IDS
        bins.append(
            {
                "start_offset_ns": lo,
                "end_offset_ns": hi,
                "complete": complete,
                "interval_count": len(gaps),
                "request_count": n_ids,
                "eligible": eligible,
                "median_gap_ns": midpoint,
                "above_threshold": eligible and midpoint >= MEDIAN_GATE_NS,
            }
        )
    adjacent = list(zip(bins, bins[1:], strict=False))
    trigger = next(
        ((lo, hi) for lo, hi in adjacent if lo["above_threshold"] and hi["above_threshold"]), None
    )
    supported = observed >= TRACE_END_NS and any(
        lo["eligible"] and hi["eligible"] for lo, hi in adjacent
    )
    return {
        "candidate": "detected" if trigger else "not_detected" if supported else "unverified",
        "first_pair_start_offset_ns": trigger[0]["start_offset_ns"] if trigger else None,
        "first_pair_decision_offset_ns": trigger[1]["end_offset_ns"] if trigger else None,
        "bins": bins,
    }


def waiting_summary(client, timing):
    """Reconstruct first-admission waiting from independently bound native endpoints."""
    if timing.get("status") != "complete" or timing.get("clock", {}).get("verified") is not True:
        return {"verified": False, "reason": "incomplete_native_request_timing"}
    t0 = client["phases"]["measure_start_ns"]
    changes = defaultdict(int)
    admissions = []
    for identity, request in timing["per_id"].items():
        first = request["first_components_ns"]
        queued = request["submit_ns"] + first[CLOCK_ARRIVAL_COMPONENT] - t0
        wait = first[QUEUE_COMPONENT]
        scheduled = queued + wait
        require(ordered(0, queued, scheduled), "D7 native queue interval is reversed")
        changes[queued] += 1
        changes[scheduled] -= 1
        admissions.append(
            {
                "request_id": identity,
                "planned_offset_ns": request["planned_ns"] - t0,
                "queued_offset_ns": queued,
                "scheduled_offset_ns": scheduled,
                "wait_ns": wait,
            }
        )
    admissions.sort(key=lambda row: (row["queued_offset_ns"], row["request_id"]))
    spans = []
    occupancy = peak = area = 0
    previous = 0
    for stamp, change in sorted(changes.items()):
        area += (stamp - previous) * occupancy
        occupancy += change
        require(occupancy >= 0, "D7 native waiting balance is negative")
        peak = max(peak, occupancy)
        if occupancy and (not spans or spans[-1][1] is not None):
            spans.append([stamp, None])
        elif not occupancy and spans and spans[-1][1] is None:
            spans[-1][1] = stamp
        previous = stamp
    require(
        occupancy == 0 and area == sum(row["wait_ns"] for row in admissions),
        "D7 native waiting area does not close to per-request ledger",
    )
    first = next((row for row in admissions if row["wait_ns"] >= WAIT_THRESHOLD_NS), None)
    span = next((row for row in spans if row[1] - row[0] >= SUSTAINED_WAIT_NS), None)
    return {
        "verified": True,
        "peak_waiting_before_first_schedule": peak,
        "first_wait_ge_50ms": first,
        "first_continuous_wait_ge_500ms": {
            "onset_offset_ns": span[0],
            "verified_at_offset_ns": span[0] + SUSTAINED_WAIT_NS,
            "end_offset_ns": span[1],
        }
        if span
        else None,
        "late_15_wait_median_ns": median(
            row["wait_ns"]
            for row in sorted(
                admissions, key=lambda row: (row["planned_offset_ns"], row["request_id"])
            )[-15:]
        )
        if admissions
        else None,
        "wait_area_request_ns": area,
    }


def paired_endpoints(runs, gates):
    """Retain original TTFT/throughput comparisons in their prespecified orientation."""
    pairs = {}
    for first, second in (
        ("E1", "E4"),
        ("E2", "E3"),
        *((reference, observed) for reference in ENDPOINTS for observed in STEPS),
    ):
        a, b = runs.get(first, {}), runs.get(second, {})
        complete = a.get("endpoint_evidence_complete") and b.get("endpoint_evidence_complete")
        pair = compare_pair(a["metrics"], b["metrics"], gates) if complete else None
        if pair is not None and all(
            run.get("native_timing", {}).get("status") == "complete" for run in (a, b)
        ):
            native = timing_pair(a["native_timing"], b["native_timing"])
            require(
                native["p95_anchors"]["ttft_p95_difference_ns"] == pair["ttft_delta_ns"],
                "D7 paired native p95 anchors differ from endpoint metrics",
            )
            pair["native_p95_anchors"] = native["p95_anchors"]
        pairs[f"{first}/{second}"] = pair
    return pairs


def fifo_inventory(trace, batches, carry, end):
    """Resolve the left carry and each actual native enqueue/dequeue edge."""
    scheduled, consumed, consume_pending = {}, {}, {}
    queue = deque([0] if carry else [])
    all_ids = batches.keys() | ({0} if carry else set())
    previous_end, previous_depth = trace["boundary"]["at_ns"], len(queue)
    for number, step in enumerate(trace["steps"], 1):
        before, after = step["queue_before"], step["queue_after"]
        new, old = step["scheduled_batch_id"], step["consumed_batch_id"]
        require(
            whole(step["step_id"], 1)
            and step["step_id"] == number
            and ordered(previous_end, step["start_ns"], step["end_ns"], trace["export_ns"])
            and (step["start_ns"] < end or queue)
            and whole(before)
            and whole(after)
            and before == previous_depth
            and before <= 1
            and after <= 1
            and type(step["model_executed"]) is bool,
            "D7 native tick order or FIFO depth differs",
        )
        previous_end, previous_depth = step["end_ns"], after
        snapshot_pending = list(reversed(queue))
        if new is not None:
            require(
                whole(new, 1)
                and new == len(scheduled) + 1
                and new in batches
                and step["model_executed"] == bool(batches[new]["scheduled"]),
                "D7 native schedule tick identity differs",
            )
            require(
                all(
                    batches[new][edge]["pending_batch_ids"] == snapshot_pending
                    for edge in ("schedule_before", "schedule_after")
                ),
                "D7 schedule snapshots differ from native FIFO",
            )
            queue.append(new)
            scheduled[new] = step
        elif step["end_ns"] < end:
            require(not step["model_executed"], "D7 native tick omitted a positive batch")
        if old is not None:
            require(
                whole(old) and queue and old == queue.popleft(),
                "D7 carry or native batch was consumed outside FIFO order",
            )
            consumed[old] = step
            consume_pending[old] = list(reversed(queue))
            if new is not None:
                launched = batches[new]
                previous = carry if old == 0 else batches[old]
                last_launch = (
                    launched["sample_end_ns"]
                    if launched["scheduled"]
                    else launched["execute_end_ns"]
                )
                require(
                    ordered(last_launch, previous["future_result_start_ns"]),
                    "D7 consumed old work before the native step's new launch",
                )
        require(
            after == len(queue) if step["end_ns"] < end else after >= len(queue),
            "D7 native FIFO depth does not close",
        )
    require(
        not queue and scheduled.keys() == batches.keys() and consumed.keys() == all_ids,
        "D7 scheduler, FIFO, and carry inventory differ",
    )
    return scheduled, consumed, consume_pending


def boundary_state(snapshot, generations, pending, *, right_suffix=False):
    seen = set()
    for name in STATE_QUEUES:
        for identity, generation in snapshot[name]:
            require(
                identity in generations
                and identity not in seen
                and whole(generation)
                and generation == generations[identity],
                "D7 boundary request identity or generation differs",
            )
            seen.add(identity)
    require(
        len(snapshot["running"]) <= 4
        and (
            snapshot["pending_batch_ids"] == pending
            or right_suffix
            and len(snapshot["pending_batch_ids"]) <= 1
            and snapshot["pending_batch_ids"] == [None, *pending]
        ),
        "D7 boundary active slots or native FIFO differs",
    )


def carry_packets(batch, step, generations, expected_pending, right_end):
    require(
        type(batch["batch_id"]) is int
        and batch["batch_id"] == 0
        and not any(
            field in batch
            for field in ("runner", "schedule_start_ns", "execute_start_ns", "sample_start_ns")
        ),
        "D7 carry claims unavailable upstream execution",
    )
    positive = bool(batch["scheduled"])
    times = ["future_result_start_ns", "future_result_end_ns", "consume_start_ns", "consume_end_ns"]
    if positive:
        times[1:1] = ["materialize_start_ns", "materialize_end_ns"]
    else:
        require(
            batch["materialize_start_ns"] is batch["materialize_end_ns"] is None
            and batch["materialized_counts"] == batch["output_counts"] == [],
            "D7 empty left carry claims materialization",
        )
    require(
        ordered(step["start_ns"], *(batch[name] for name in times), step["end_ns"]),
        "D7 carry result/materialization/consumption chronology differs",
    )
    for edge in ("consume_before", "consume_after"):
        boundary_state(
            batch[edge],
            generations,
            expected_pending,
            right_suffix=batch["consume_end_ns"] >= right_end,
        )
    scheduled = checked_requests(batch["scheduled"], generations, "D7 carry schedule")
    materialized = checked_requests(
        batch["materialized_counts"], generations, "D7 carry materialization", known=scheduled
    )
    packets = checked_requests(
        batch["output_counts"], generations, "D7 carry packet", known=scheduled
    )
    require(
        all(whole(row["tokens"], 1) for row in scheduled.values())
        and materialized.keys() == scheduled.keys()
        and all(
            whole(row["generated_tokens"]) and row["generated_tokens"] <= 1
            for row in materialized.values()
        )
        and all(
            type(row["generated_tokens"]) is int
            and row["generated_tokens"] == 1
            and row["generated_tokens"] <= materialized[identity]["generated_tokens"]
            and type(row["token_ids"]) is list
            and len(row["token_ids"]) == 1
            and whole(row["token_ids"][0])
            and ordered(batch["consume_start_ns"], row["packet_ns"], batch["consume_end_ns"])
            for identity, row in packets.items()
        )
        and len({row["packet_ns"] for row in packets.values()}) <= 1,
        "D7 carry packet and materialized tokens differ",
    )
    return packets


def append_packet(
    links,
    client_rows,
    tokens,
    generations,
    *,
    identity,
    generation,
    ordinal,
    packet_ns,
    token,
    batch_id,
):
    chain = links[identity]
    request = client_rows[identity]
    arrivals = tokens[identity]
    require(
        whole(generation)
        and generations.get(identity) == generation
        and whole(ordinal, 1)
        and ordinal == len(chain) + 1 <= len(arrivals)
        and whole(packet_ns)
        and whole(token)
        and (not chain or chain[-1]["packet_ns"] <= packet_ns)
        and packet_ns <= arrivals[ordinal - 1][0]
        and token == arrivals[ordinal - 1][1],
        "D7 native packet ordinal, generation, or real client token differs",
    )
    key = "first" if ordinal == 1 else "last" if ordinal == len(arrivals) else None
    native = (request.get("native_metrics") or {}).get(key)
    if native is not None and (ordinal == 1 or request["status"] == "completed"):
        require(
            round(native[key + "_token_ts"] * SECOND) == packet_ns,
            "D7 native first/last timestamp does not anchor its packet",
        )
    chain.append(
        {
            "ordinal": ordinal,
            "generation": generation,
            "batch_id": batch_id,
            "packet_ns": packet_ns,
            "client_ns": arrivals[ordinal - 1][0],
        }
    )


def audit_d7_trace(trace, client, limits, *, leaf_end_ns):
    """Bind late host events, preserving the prefix/carry evidence boundary."""
    status = trace["status"]
    require(
        type(trace["schema_version"]) is int
        and trace["schema_version"] == 1
        and status in {"complete", "incomplete", "not-started"}
        and client.get("d7_capture") == "step",
        "D7 trace schema, status or capture differs",
    )
    if status != "complete":
        require(
            (trace.get("overflow") is True or bool(trace.get("errors")))
            if status == "incomplete"
            else set(trace) == {"schema_version", "status"}
            and "d7_arm_return_ns" not in client
            and "measure_start_ns" not in client["phases"],
            "D7 partial trace lacks matching arm/failure evidence",
        )
        return {"verified": False, "status": status, "issues": trace.get("errors", [])}

    t0, start, end, armed, exported = (
        trace[key] for key in ("t0_ns", "start_ns", "end_ns", "armed_ns", "export_ns")
    )
    phases = client["phases"]
    boundary = trace["boundary"]
    arm_return = client["d7_arm_return_ns"]
    require(
        ordered(
            phases["warmup_end_ns"],
            armed,
            arm_return,
            t0,
            start,
            boundary["at_ns"],
            end,
            exported,
            leaf_end_ns,
        )
        and arm_return < t0
        and boundary["at_ns"] < end
        and start - t0 == limits["start_offset_ns"] == TRACE_START_NS
        and end - start == limits["window_ns"] == TRACE_END_NS - TRACE_START_NS
        and t0 == phases["measure_start_ns"]
        and t0 - armed <= limits["arm_lead_ns"]
        and phases["drain_end_ns"] <= exported
        and trace["overflow"] is False
        and trace["errors"] == [],
        "D7 trace arm, observed window, or export boundary differs",
    )
    rows = trace["batches"]
    carry_rows = trace["carry_in"]
    require(
        all(
            type(trace[key]) is list
            for key in (
                "batches",
                "enqueues",
                "steps",
                "loops",
                "input_polls",
                "preroll_tokens",
                "carry_in",
            )
        )
        and rows
        and trace["enqueues"]
        and trace["steps"]
        and any(row["scheduled"] for row in rows)
        and len(carry_rows) <= 1
        and len(rows) + len(carry_rows) <= limits["batch_limit"]
        and sum(
            len(trace[key])
            for key in (
                "batches",
                "enqueues",
                "steps",
                "loops",
                "input_polls",
                "preroll_tokens",
                "carry_in",
            )
        )
        <= limits["event_limit"],
        "D7 bounded trace inventories differ",
    )
    batches = {row["batch_id"]: row for row in rows}
    require(
        list(batches) == list(range(1, len(rows) + 1))
        and all(whole(row["batch_id"], 1) for row in rows),
        "D7 detailed batch identity inventory differs",
    )
    carry = carry_rows[0] if carry_rows else None
    client_rows = {row["request_id"]: row for row in client["requests"]}
    require(len(client_rows) == len(client["requests"]), "D7 client denominator has duplicate IDs")
    generations = enqueued_requests(trace, client_rows, t0, end)
    enqueue_times = {row["request_id"]: row["end_ns"] for row in trace["enqueues"]}
    boundary_generations = {
        identity: generation
        for identity, generation in generations.items()
        if enqueue_times[identity] <= boundary["at_ns"]
    }
    boundary_state(boundary["state"], boundary_generations, [0] if carry else [])
    engine_loops(trace, start, end, exported)
    require(
        boundary["at_ns"] <= trace["loops"][0]["start_ns"],
        "D7 detailed outer loop precedes actual native boundary",
    )
    scheduled, consumed, pending = fifo_inventory(trace, batches, carry, end)
    tokens = {identity: client_tokens(row) for identity, row in client_rows.items()}
    links = defaultdict(list)

    previous = t0
    for row in trace["preroll_tokens"]:
        identity, packet_ns = row["request_id"], row["packet_ns"]
        require(
            identity in enqueue_times
            and ordered(previous, packet_ns, boundary["at_ns"])
            and enqueue_times[identity] <= packet_ns,
            "D7 pre-roll packet chronology differs from native boundary",
        )
        append_packet(
            links,
            client_rows,
            tokens,
            generations,
            identity=identity,
            generation=row["generation"],
            ordinal=row["ordinal"],
            packet_ns=packet_ns,
            token=row["token_id"],
            batch_id=None,
        )
        previous = packet_ns

    def append_batch_packets(packets, batch_id):
        for identity, row in packets.items():
            append_packet(
                links,
                client_rows,
                tokens,
                generations,
                identity=identity,
                generation=row["generation"],
                ordinal=row["ordinal"],
                packet_ns=row["packet_ns"],
                token=row["token_ids"][0],
                batch_id=batch_id,
            )

    if carry:
        append_batch_packets(carry_packets(carry, consumed[0], generations, pending[0], end), 0)
    first_dispatch = {}
    for batch_id, batch in batches.items():
        packets = batch_packets(
            batch, scheduled[batch_id], consumed[batch_id], generations, enqueue_times, end
        )
        check_snapshots(batch, client_rows, generations, pending[batch_id], end)
        append_batch_packets(packets, batch_id)
        for row in batch["scheduled"]:
            first_dispatch.setdefault(row["request_id"], batch)
    for identity, request in client_rows.items():
        first = (request.get("native_metrics") or {}).get("first")
        if first:
            scheduled_ns = round(first["scheduled_ts"] * SECOND)
            if boundary["at_ns"] <= scheduled_ns < end:
                batch = first_dispatch.get(identity)
                require(
                    batch is not None
                    and ordered(batch["schedule_start_ns"], scheduled_ns, batch["schedule_end_ns"]),
                    "D7 late first native dispatch lacks its detailed scheduler batch",
                )
    require(
        all(
            sum(stamp < end for stamp, _ in stamps) <= len(links.get(identity, ()))
            for identity, stamps in tokens.items()
        ),
        "D7 client real token before the right edge lacks native ordinal proof",
    )
    return {
        "verified": True,
        "status": status,
        "window_start_ns": t0,
        "window_end_ns": end,
        "configured_capture_start_offset_ns": start - t0,
        "actual_capture_start_offset_ns": boundary["at_ns"] - t0,
        "carry_batch_count": len(carry_rows),
        "preroll_tokens": len(trace["preroll_tokens"]),
        "positive_batches": sum(bool(row["scheduled"]) for row in rows),
        "empty_batches": sum(not row["scheduled"] for row in rows),
        "linked_late_real_tokens": sum(
            row["batch_id"] is not None and row["batch_id"] > 0
            for chain in links.values()
            for row in chain
        ),
        "enqueue_end_ns": enqueue_times,
        "links": dict(links),
        "batches": {str(key): row for key, row in batches.items()},
        "carry": carry,
        "boundary": boundary,
        "fifo": {
            str(key): {
                "schedule_queue_before": scheduled[key]["queue_before"],
                "consume_queue_before": consumed[key]["queue_before"],
                "schedule_step_id": scheduled[key]["step_id"],
                "consume_step_id": consumed[key]["step_id"],
            }
            for key in batches
        },
        "carry_consume_step_id": consumed[0]["step_id"] if carry else None,
    }


def observe_slots(trace):
    """Observe actual scheduler states and identify a release followed by admission."""
    t0 = trace["window_start_ns"]
    events = []
    for identity, batch in trace["batches"].items():
        events.extend(
            (
                batch[kind + "_start_ns"],
                kind,
                int(identity),
                batch,
                trace["fifo"][identity][kind + "_step_id"],
            )
            for kind in ("schedule", "consume")
        )
    carry = trace["carry"]
    if carry:
        events.append(
            (carry["consume_start_ns"], "consume", 0, carry, trace["carry_consume_step_id"])
        )
    initial = trace["boundary"]["state"]
    initial_waiting = [
        identity for queue in ("waiting", "kv_holding_waiting") for identity, _ in initial[queue]
    ]
    samples = [
        {
            "offset_ns": trace["boundary"]["at_ns"] - t0,
            "step_id": None,
            "batch_id": None,
            "boundary": "left_boundary",
            "running": len(initial["running"]),
            "waiting": len(initial_waiting),
            "waiting_ids": sorted(initial_waiting),
        }
    ]
    handoffs, release = [], None
    for start, kind, identity, batch, tick in sorted(events):
        before, after = (batch[f"{kind}_{edge}"] for edge in ("before", "after"))
        end = batch[kind + "_end_ns"]
        states = [
            (
                {req for req, _ in state["running"]},
                {req for name in ("waiting", "kv_holding_waiting") for req, _ in state[name]},
            )
            for state in (before, after)
        ]
        for stamp, (running, waiting) in zip((start, end), states, strict=True):
            if stamp < trace["window_end_ns"]:
                samples.append(
                    {
                        "offset_ns": stamp - t0,
                        "step_id": tick,
                        "batch_id": identity,
                        "boundary": kind,
                        "running": len(running),
                        "waiting": len(waiting),
                        "waiting_ids": sorted(waiting),
                    }
                )
        (running, waiting), (next_running, _) = states
        released = running - next_running
        admitted = next_running - running
        if kind == "consume" and released:
            release = (
                (end, identity, tick, sorted(released), waiting) if len(running) == 4 else None
            )
        if kind == "schedule" and admitted:
            if release and end < trace["window_end_ns"]:
                released_at, release_id, release_tick, leaving, prior_waiting = release
                dispatch = {row["request_id"] for row in batch["scheduled"]}
                for request in sorted(admitted & waiting & prior_waiting & dispatch):
                    enqueued = trace["enqueue_end_ns"].get(request)
                    if enqueued is not None and released_at <= start:
                        handoffs.append(
                            {
                                "request_id": request,
                                "released_ids": leaving,
                                "release_batch_id": release_id,
                                "release_step_id": release_tick,
                                "release_offset_ns": released_at - t0,
                                "admission_batch_id": identity,
                                "admission_step_id": tick,
                                "admission_offset_ns": start - t0,
                                "release_to_admit_ns": start - released_at,
                                "enqueue_to_admit_ns": start - enqueued,
                            }
                        )
            release = None
    full_wait = [row for row in samples if row["running"] == 4 and row["waiting"] > 0]
    return {
        "peak_observed_wait": max((row["waiting"] for row in samples), default=0),
        "first_observed_full_with_wait": full_wait[0] if full_wait else None,
        "samples": samples,
        "release_to_admit": handoffs,
    }


def comparable_gaps(fast, slow, fast_timing, slow_timing, lo, hi):
    """Join exact same-ID adjacent ordinals with complete host-stage boundaries."""
    rows = []
    excluded_left = 0
    for identity, slow_chain in slow["links"].items():
        fast_chain = fast["links"].get(identity, [])
        for prior, current in zip(slow_chain, slow_chain[1:], strict=False):
            offset = current["client_ns"] - slow["window_start_ns"]
            ordinal = current["ordinal"]
            if not lo <= offset < hi or ordinal > len(fast_chain):
                continue
            fast_links = fast_chain[ordinal - 2 : ordinal]
            slow_links = [prior, current]
            if any(row["batch_id"] in (None, 0) for row in (*fast_links, *slow_links)):
                excluded_left += 1
                continue
            if fast_links[-1]["client_ns"] >= fast["window_end_ns"]:
                continue
            baseline, observed = packet_path(fast, *fast_links), packet_path(slow, *slow_links)
            deltas = {key: observed[key] - value for key, value in baseline.items()}
            positive = sorted(
                ((value, key) for key, value in deltas.items() if value > 0), reverse=True
            )
            winner = (
                positive[0][1]
                if positive and positive[0][0] > sum(value for value, _ in positive[1:])
                else None
            )
            match = describe_pair(fast, slow, fast_links, slow_links, [fast_timing, slow_timing])
            rows.append(
                {
                    "request_id": identity,
                    "generated_token_ordinal": ordinal,
                    "slow_client_offset_ns": offset,
                    "slow_batch_id": current["batch_id"],
                    "fast_client_gap_ns": sum(baseline.values()),
                    "slow_client_gap_ns": sum(observed.values()),
                    "fast_components_ns": baseline,
                    "slow_components_ns": observed,
                    "slow_minus_fast_ns": deltas,
                    "batches": match,
                    "same_shape": all(row["fast_shape"] == row["slow_shape"] for row in match),
                    "same_composition": all(
                        row["same_native_schedule"] and row["same_observed_state"] for row in match
                    ),
                    "matched_state": all(
                        row["same_cache_inventory"] and row["same_queue_depths"] for row in match
                    ),
                    "dominant_boundary": winner,
                }
            )
    rows.sort(
        key=lambda row: (
            row["slow_client_offset_ns"],
            row["request_id"],
            row["generated_token_ordinal"],
        )
    )
    return rows, excluded_left


def later_wait_worsening(anchor, slow, fast):
    """Find a deeper observed full-slot queue after a matched host boundary."""
    before = [row for row in slow["samples"] if anchor - BIN_NS <= row["offset_ns"] <= anchor]
    if not before:
        return None
    initial = max(row["waiting"] for row in before)
    for row in slow["samples"]:
        offset = row["offset_ns"]
        if offset <= anchor or row["running"] != 4 or row["waiting"] <= initial:
            continue
        matched_fast = [
            sample["waiting"]
            for sample in fast["samples"]
            if offset - BIN_NS <= sample["offset_ns"] < offset + BIN_NS
        ]
        if matched_fast and row["waiting"] > max(matched_fast):
            return {
                "prior_peak_wait": initial,
                "fast_neighbor_peak_wait": max(matched_fast),
                "first_deeper_full_queue": row,
            }
    return None


def formation_evidence(waits, slots):
    """Describe repeat waiting and native release/admit even without a late trigger."""
    endpoints = {
        name: {
            key: waits[name].get(key)
            for key in ("first_wait_ge_50ms", "first_continuous_wait_ge_500ms")
        }
        for name in ENDPOINTS
    }
    direct = {}
    for name, evidence in slots.items():
        first = waits[name].get("first_wait_ge_50ms")
        identity = first["request_id"] if first else None
        full_wait = next(
            (
                sample
                for sample in evidence["samples"]
                if first is not None
                and identity in sample["waiting_ids"]
                and sample["running"] == 4
                and first["queued_offset_ns"] <= sample["offset_ns"] < first["scheduled_offset_ns"]
            ),
            None,
        )
        direct[name] = {
            "first_observed_full_with_wait": evidence["first_observed_full_with_wait"],
            "first_noticeable_wait_full_slot_sample": full_wait,
            "first_noticeable_wait_release_to_admit": next(
                (row for row in evidence["release_to_admit"] if row["request_id"] == identity), None
            ),
            "linked_release_to_admit_count": len(evidence["release_to_admit"]),
        }
    return {
        "status": "unresolved",
        "reason": "no_verified_repeated_pre_onset_boundary",
        "endpoint_native_wait_anchors": endpoints,
        "stepped_direct_slot_evidence": direct,
    }


def slower_gap(rows):
    return [
        row
        for row in rows
        if row["matched_state"]
        and row["slow_client_gap_ns"] >= MEDIAN_GATE_NS > row["fast_client_gap_ns"]
    ]


def repeated_boundaries(contrasted):
    exact = [row for row in contrasted if row["same_shape"] and row["same_composition"]]
    result = {}
    for category, boundaries in (
        ("execution_or_materialization_slowdown", EXECUTION),
        ("engine_progress_pause", AFTER_READY),
    ):
        groups = []
        for boundary in sorted(boundaries):
            matching = [row for row in exact if row["dominant_boundary"] == boundary]
            links = distinct_batch_pairs(matching)
            if len(links) >= 2:
                anchor = min(
                    row["batches"][-1]["slow_event_offsets_ns"][BOUNDARY_START[boundary]]
                    for row in matching
                )
                groups.append(
                    {"boundary": boundary, "anchor_offset_ns": anchor, "linked_gaps": links}
                )
        if len(groups) == 1:
            result[category] = groups[0]
    return result


def pre_onset_backtrace(waits, runs, traces, formation):
    """Look for a pre-onset repeated host event independently of the late bin screen."""
    if not all(
        all(anchor for anchor in row.values())
        for row in formation["endpoint_native_wait_anchors"].values()
    ):
        return
    observed = []
    scans = []
    for slow, fast in (("E2", "E3"), ("E3", "E2")):
        slot = formation["stepped_direct_slot_evidence"].get(slow, {})
        first, sustained = (
            waits[slow][key] for key in ("first_wait_ge_50ms", "first_continuous_wait_ge_500ms")
        )
        other = waits[fast]["first_wait_ge_50ms"]
        if not (
            first and sustained and other and slot.get("first_noticeable_wait_release_to_admit")
        ):
            continue
        hi = min(first["queued_offset_ns"], sustained["onset_offset_ns"], other["queued_offset_ns"])
        if hi <= TRACE_START_NS or first["queued_offset_ns"] >= other["queued_offset_ns"]:
            continue
        rows, excluded = comparable_gaps(
            traces[fast],
            traces[slow],
            runs[fast]["native_timing"]["per_id"],
            runs[slow]["native_timing"]["per_id"],
            TRACE_START_NS,
            hi,
        )
        groups = repeated_boundaries(slower_gap(rows))
        scans.append(
            {
                "slow": slow,
                "fast": fast,
                "end_offset_ns": hi,
                "left_edge_incomplete_comparisons": excluded,
                "comparisons": rows,
            }
        )
        if len(groups) == 1:
            category, evidence = next(iter(groups.items()))
            observed.append({"mechanism": category, "slow": slow, "fast": fast, **evidence})
    formation["pre_onset_scans"] = scans
    if len(observed) == 1:
        formation.update(
            status="supported_host_boundary",
            reason="repeated_host_boundary_precedes_both_wait_anchors",
            host_boundary=observed[0],
        )


def assess_d7(runs, clients, traces, gates, *, heat_verified):
    """Keep ABBA capture equivalence separate from a supported host association."""
    signatures = {
        name: late_signature(clients[name]) if name in clients else None for name in MEASURED
    }
    waits = {
        name: waiting_summary(clients[name], runs.get(name, {}).get("native_timing", {}))
        if name in clients
        else {"verified": False, "reason": "unattempted"}
        for name in MEASURED
    }
    pairs = paired_endpoints(runs, gates)
    primary = ("E1/E2", "E4/E3")
    same_tier = ("E1/E4", "E2/E3")
    relevant = [pairs[name] for name in (*primary, *same_tier)]
    endpoints_complete = all(
        runs.get(name, {}).get("endpoint_evidence_complete") and waits[name]["verified"]
        for name in MEASURED
    )
    traces_complete = all(traces.get(name, {}).get("verified") for name in STEPS)
    slots = {
        name: observe_slots(traces[name]) for name in STEPS if traces.get(name, {}).get("verified")
    }
    capture_equivalence = "unverified"
    if heat_verified and endpoints_complete and traces_complete and all(relevant):
        capture_equivalence = (
            "supported"
            if all(row["within_tolerance"] for row in relevant)
            else "same_tier_unstable"
            if any(not pairs[name]["within_tolerance"] for name in same_tier)
            else "primary_capture_contrast_outside_tolerance"
        )
    verdicts = {
        name: {"status": "unresolved", "reason": "required_comparison_unavailable"}
        for name in MECHANISMS
    }
    result = {
        "localization": "unresolved",
        "reason": None,
        "capture_equivalence": capture_equivalence,
        "same_tier_pairs": same_tier,
        "primary_capture_pairs": primary,
        "supplementary_capture_pairs": ("E1/E3", "E4/E2"),
        "pairs": pairs,
        "late_signatures": signatures,
        "waiting": waits,
        "direct_slot_evidence": slots,
        "backlog_formation": formation_evidence(waits, slots),
        "late_amplification": {"status": "unresolved", "reason": "comparison_not_available"},
        "mechanisms": verdicts,
        "scope": "Native request and host trace intervals. No device execution time or exact IPC publication time.",
    }
    if not (heat_verified and endpoints_complete and traces_complete):
        result["reason"] = "thermal_request_or_trace_evidence_incomplete"
        result["late_amplification"]["reason"] = result["reason"]
        return result
    formation = result["backlog_formation"]
    pre_onset_backtrace(waits, runs, traces, formation)
    if formation["status"] == "supported_host_boundary":
        result["localization"] = formation["host_boundary"]["boundary"]
    if not any(signatures[name]["candidate"] == "detected" for name in ENDPOINTS):
        result["reason"] = "endpoint_late_signature_not_reproduced"
        result["late_amplification"]["reason"] = result["reason"]
        return result
    slow = [name for name in STEPS if signatures[name]["candidate"] == "detected"]
    fast = [name for name in STEPS if signatures[name]["candidate"] == "not_detected"]
    if len(slow) != 1 or len(fast) != 1:
        result["reason"] = "stepped_late_fast_slow_contrast_absent"
        result["late_amplification"]["reason"] = result["reason"]
        return result
    slow, fast = slow[0], fast[0]
    first = signatures[slow]["first_pair_start_offset_ns"]
    lo = max(TRACE_START_NS, first - BIN_NS)
    hi = signatures[slow]["first_pair_decision_offset_ns"]
    rows, excluded = comparable_gaps(
        traces[fast],
        traces[slow],
        runs[fast]["native_timing"]["per_id"],
        runs[slow]["native_timing"]["per_id"],
        lo,
        hi,
    )
    result["matched_backtrace"] = {
        "slow": slow,
        "fast": fast,
        "start_offset_ns": lo,
        "end_offset_ns": hi,
        "left_edge_incomplete_comparisons": excluded,
        "comparisons": rows,
    }
    contrasted = slower_gap(rows)
    candidate_groups = repeated_boundaries(contrasted)
    for category in ("execution_or_materialization_slowdown", "engine_progress_pause"):
        if category in candidate_groups:
            verdicts[category] = {"status": "host_association", **candidate_groups[category]}
        else:
            verdicts[category]["reason"] = "no_unique_repeated_comparable_host_boundary"
    shapes = distinct_batch_pairs([row for row in contrasted if not row["same_shape"]])
    controls = distinct_batch_pairs(
        [
            row
            for row in rows
            if row["matched_state"]
            and row["same_shape"]
            and max(row["fast_client_gap_ns"], row["slow_client_gap_ns"]) < MEDIAN_GATE_NS
        ]
    )
    verdicts["batch_composition_association"] = {
        "status": "associated" if len(shapes) >= 2 and len(controls) >= 2 else "unresolved",
        "changed_shape_slow_gaps": shapes,
        "same_shape_fast_controls": controls,
    }
    if len(candidate_groups) != 1:
        result["reason"] = "no_unique_repeated_comparable_host_boundary"
        result["late_amplification"]["reason"] = result["reason"]
        return result
    group = next(iter(candidate_groups.values()))
    anchor = group["anchor_offset_ns"]
    sustained = waits[slow]["first_continuous_wait_ge_500ms"]
    first_wait = waits[slow]["first_wait_ge_50ms"]
    precedes_backlog = sustained is not None and anchor < sustained["onset_offset_ns"]
    precedes_first_long_wait = first_wait is not None and anchor < first_wait["queued_offset_ns"]
    worsening = later_wait_worsening(anchor, slots[slow], slots[fast])
    fast_handoffs = {row["request_id"]: row for row in slots[fast]["release_to_admit"]}
    handoffs = [
        {"slow": row, "fast": fast_handoffs[row["request_id"]]}
        for row in slots[slow]["release_to_admit"]
        if row["request_id"] in fast_handoffs
        and worsening is not None
        and row["release_offset_ns"] >= worsening["first_deeper_full_queue"]["offset_ns"]
        and row["enqueue_to_admit_ns"] > fast_handoffs[row["request_id"]]["enqueue_to_admit_ns"]
    ]
    verdicts["full_slot_amplification"] = {
        "status": "associated" if handoffs else "unresolved",
        "later_queue_worsening": worsening,
        "matched_release_to_admit": handoffs,
    }
    result["ordered_evidence"] = {
        "first_repeated_host_boundary": group,
        "precedes_first_50ms_wait_enqueue": precedes_first_long_wait,
        "precedes_first_sustained_wait": precedes_backlog,
        "precedes_direct_queue_worsening": worsening is not None,
    }
    if precedes_backlog and precedes_first_long_wait:
        result["backlog_formation"].update(
            status="supported_host_boundary",
            reason="repeated_host_boundary_precedes_both_wait_anchors",
            host_boundary=group,
        )
        result.update(
            localization=group["boundary"], reason="repeated_host_boundary_precedes_queue_onset"
        )
    elif handoffs:
        result.update(
            localization=group["boundary"],
            reason="repeated_host_boundary_before_later_slot_amplification",
        )
    else:
        result["reason"] = "host_association_without_verified_queue_onset_or_later_amplification"
    result["late_amplification"] = {
        "status": "associated" if handoffs else "unresolved",
        "reason": "repeated_host_boundary_precedes_deeper_full_queue_and_slower_admission"
        if handoffs
        else "no_verified_later_queue_worsening_and_matched_handoff",
        "host_boundary": group,
        "later_queue_worsening": worsening,
        "matched_release_to_admit": handoffs,
    }
    return result
