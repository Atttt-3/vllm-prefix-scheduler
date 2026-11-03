"""Independently bind D6 host-observed steps to native packets and real client tokens."""

from collections import defaultdict, deque

from check_m5_d1 import compare_pair
from m5_d6_signature import BIN_NS, MEDIAN_GATE_NS, client_gap_signature

from vllm_prefix_scheduler.m5_campaign import require

NS = 1_000_000_000
MEASURED = ("E1", "E2", "E3")
STEPPED = ("E2", "E3")
STATE_QUEUES = ("running", "waiting", "kv_holding_waiting")
BATCH_TIMES = (
    "schedule_start_ns",
    "schedule_end_ns",
    "execute_start_ns",
    "execute_end_ns",
    "sample_start_ns",
    "sample_end_ns",
    "future_result_start_ns",
    "materialize_start_ns",
    "materialize_end_ns",
    "future_result_end_ns",
    "consume_start_ns",
    "consume_end_ns",
)


def verify_step_sources(plan, source):
    expected = plan["runtime_sources"]
    require(
        expected
        and all(
            source["upstream_sha256"].get(name) == sha
            and source["upstream_modules"].get(name, {}).get("sha256") == sha
            for name, sha in expected.items()
        ),
        "D6 installed host trace source differs",
    )


def whole(value, lower=0):
    return type(value) is int and value >= lower


def ordered(*times):
    return all(whole(stamp) for stamp in times) and list(times) == sorted(times)


def client_tokens(request):
    return [
        (chunk["at_ns"], token) for chunk in request["chunk_tokens"] for token in chunk["token_ids"]
    ]


def checked_requests(rows, generations, label, *, known=None):
    require(type(rows) is list, f"D6 {label} rows malformed")
    result = {}
    for row in rows:
        identity = row["request_id"]
        generation = row["generation"]
        require(
            identity in generations
            and identity not in result
            and whole(generation)
            and generation == generations[identity]
            and (known is None or identity in known),
            f"D6 {label} request generation differs",
        )
        result[identity] = row
    return result


def enqueued_requests(trace, requests, start, end):
    generations, previous = {}, start
    for expected_generation, row in enumerate(trace["enqueues"]):
        identity = row["request_id"]
        require(
            identity in requests
            and identity not in generations
            and whole(row["generation"])
            and row["generation"] == expected_generation
            and ordered(requests[identity]["submit_ns"], row["start_ns"], row["end_ns"])
            and previous <= row["start_ns"] < end,
            "D6 enqueue identity, order or generation differs",
        )
        generations[identity] = expected_generation
        previous = row["end_ns"]
    for identity, row in requests.items():
        first = (row.get("native_metrics") or {}).get("first")
        if first:
            queued = round(first["queued_ts"] * NS)
            require(
                not (start <= queued < end) or identity in generations,
                "D6 in-window native enqueue absent",
            )
    for row in trace["enqueues"]:
        first = (requests[row["request_id"]].get("native_metrics") or {}).get("first")
        if first:
            queued = round(first["queued_ts"] * NS)
            require(
                row["start_ns"] <= queued <= row["end_ns"],
                "D6 enqueue envelope disagrees with native queue stamp",
            )
    return generations


def step_inventory(steps, batches, start, end, exported):
    """Bind each scheduler output and each consumption to one native FIFO tick."""
    scheduled, consumed, queue_states, pending = {}, {}, {}, deque()
    previous_end, queue_after = start, 0
    for number, step in enumerate(steps, 1):
        before, after = step["queue_before"], step["queue_after"]
        a, b = step["scheduled_batch_id"], step["consumed_batch_id"]
        require(
            whole(step["step_id"], 1)
            and step["step_id"] == number
            and ordered(previous_end, step["start_ns"], step["end_ns"], exported)
            and (step["start_ns"] < end or pending)
            and whole(before)
            and whole(after)
            and before == queue_after
            and before <= 1
            and after <= 1
            and type(step["model_executed"]) is bool,
            "D6 engine tick chronology or FIFO capacity differs",
        )
        previous_end, queue_after = step["end_ns"], after
        scheduler_pending = list(reversed(pending))
        if a is not None:
            require(
                whole(a, 1)
                and a == len(scheduled) + 1
                and a in batches
                and step["model_executed"] == bool(batches[a]["scheduled"]),
                "D6 scheduler dispatch differs from its native tick",
            )
            pending.append(a)
            scheduled[a] = step
        elif step["end_ns"] < end:
            require(not step["model_executed"], "D6 native tick lacks its scheduled batch")
        if b is not None:
            require(
                whole(b, 1) and pending and b == pending.popleft(),
                "D6 consumed output violates native FIFO order",
            )
            consumed[b] = step
            queue_states[b] = list(reversed(pending))
        if a is not None:
            for key in ("schedule_before", "schedule_after"):
                require(
                    batches[a][key]["pending_batch_ids"] == scheduler_pending,
                    "D6 scheduler snapshot differs from native FIFO",
                )
        # Native work scheduled at or beyond the observation cutoff can coexist
        # with the final tracked result. Only the tracked prefix is complete.
        if step["end_ns"] < end:
            require(after == len(pending), "D6 native queue depth differs from captured FIFO")
        else:
            require(after >= len(pending), "D6 tracked batch absent from native FIFO")
    require(
        not pending and scheduled.keys() == consumed.keys() == batches.keys(),
        "D6 scheduler, FIFO and consumption inventories differ",
    )
    return scheduled, consumed, queue_states


def engine_loops(trace, start, end, exported):
    loops, polls, steps = (trace[key] for key in ("loops", "input_polls", "steps"))
    previous = start
    require(len(loops) == len(steps), "D6 outer-loop and native step inventories differ")
    for loop, step in zip(loops, steps, strict=True):
        require(
            ordered(
                previous,
                loop["start_ns"],
                step["start_ns"],
                step["end_ns"],
                loop["end_ns"],
                exported,
            )
            and whole(loop["step_id"], 1)
            and loop["step_id"] == step["step_id"]
            and loop["model_executed"] is step["model_executed"],
            "D6 outer-loop chronology or native step identity differs",
        )
        previous = loop["end_ns"]
    previous = trace["armed_ns"]
    for poll in polls:
        require(
            ordered(previous, poll["start_ns"], poll["end_ns"], exported)
            and poll["start_ns"] < end
            and poll["end_ns"] >= start,
            "D6 input poll window or chronology differs",
        )
        previous = poll["end_ns"]
    timeline = sorted(loops + polls, key=lambda row: row["start_ns"])
    require(
        all(
            left["end_ns"] <= right["start_ns"]
            for left, right in zip(timeline, timeline[1:], strict=False)
        ),
        "D6 input polling overlaps native engine loop",
    )
    for row in trace["enqueues"]:
        require(
            not polls
            or row["end_ns"] <= polls[0]["start_ns"]
            or any(
                poll["start_ns"] <= row["start_ns"] <= row["end_ns"] <= poll["end_ns"]
                for poll in polls
            ),
            "D6 enqueue is outside observed native input polling",
        )


def check_snapshots(batch, requests, generations, pending, cutoff):
    for boundary in ("schedule_before", "schedule_after", "consume_before", "consume_after"):
        snapshot = batch[boundary]
        seen = set()
        stamp = batch[boundary.replace("before", "start_ns").replace("after", "end_ns")]
        for name in STATE_QUEUES:
            for identity, generation in snapshot[name]:
                require(
                    identity in requests
                    and identity not in seen
                    and whole(generation)
                    and (
                        generations.get(identity) == generation
                        or identity not in generations
                        and stamp >= cutoff
                        and generation >= len(generations)
                    )
                    and ordered(requests[identity]["submit_ns"], stamp),
                    "D6 scheduler state identity, generation or time differs",
                )
                seen.add(identity)
        require(len(snapshot["running"]) <= 4, "D6 scheduler state exceeds active cap")
        if boundary.startswith("consume"):
            expected = pending
            observed = snapshot["pending_batch_ids"]
            require(
                observed == expected
                or stamp >= cutoff
                and observed == [None, *expected]
                and len(observed) <= 1,
                "D6 consumption snapshot differs from native FIFO",
            )


def batch_packets(batch, scheduled_step, consumed_step, generations, enqueue_times, end):
    require(
        type(batch["effective_active_cap"]) is int
        and batch["effective_active_cap"] == 4
        and batch["admission_stop_reason"] == "unknown",
        "D6 native admission cap or stop reason differs",
    )
    positive = bool(batch["scheduled"])
    times = [
        batch[key]
        for key in BATCH_TIMES
        if positive
        or key
        not in {
            "sample_start_ns",
            "sample_end_ns",
            "materialize_start_ns",
            "materialize_end_ns",
        }
    ]
    require(
        ordered(*times)
        and batch["schedule_start_ns"] < end
        and scheduled_step["start_ns"] <= batch["schedule_start_ns"]
        and (batch["sample_end_ns"] if positive else batch["execute_end_ns"])
        <= scheduled_step["end_ns"]
        and consumed_step["start_ns"] <= batch["future_result_start_ns"]
        and batch["consume_end_ns"] <= consumed_step["end_ns"],
        "D6 scheduler, executor, materialization or consumption chronology differs",
    )
    scheduled = checked_requests(batch["scheduled"], generations, "scheduler")
    require(
        all(
            whole(row["tokens"], 1)
            and whole(row["start"])
            and whole(row["end"], 1)
            and row["end"] - row["start"] == row["tokens"]
            and whole(row["prompt_length"], 1)
            and row["prompt_length"] == 641
            and whole(row["preemptions"])
            and enqueue_times[identity] <= batch["schedule_start_ns"]
            for identity, row in scheduled.items()
        ),
        "D6 scheduled absolute span, count or enqueue ordering differs",
    )
    if not positive:
        require(
            all(
                batch[key] is None
                for key in (
                    "sample_start_ns",
                    "sample_end_ns",
                    "materialize_start_ns",
                    "materialize_end_ns",
                    "runner",
                )
            )
            and batch["materialized_counts"] == batch["output_counts"] == [],
            "D6 empty batch unexpectedly contains runner or token evidence",
        )
        return {}
    runner = batch["runner"]
    effective, padded = runner["effective_tokens"], runner["padded_tokens"]
    runner_rows = checked_requests(runner["rows"], generations, "runner", known=scheduled)
    require(
        whole(effective, 1) and whole(padded, effective) and runner_rows.keys() == scheduled.keys(),
        "D6 runner effective/padded tokens or request inventory differs",
    )
    offset = 0
    for identity, row in runner_rows.items():
        require(
            whole(row["query_start"])
            and row["query_start"] == offset
            and whole(row["query_end"], offset + 1)
            and row["query_end"] - offset == scheduled[identity]["tokens"]
            and whole(row["prefill_length"], 1),
            "D6 CPU runner query span differs from scheduled tokens",
        )
        offset = row["query_end"]
    require(offset == effective, "D6 runner CPU spans do not exhaust effective tokens")
    materialized = checked_requests(
        batch["materialized_counts"], generations, "materialized", known=scheduled
    )
    packets = checked_requests(
        batch["output_counts"], generations, "native packet", known=scheduled
    )
    require(
        materialized.keys() == scheduled.keys()
        and all(
            whole(row["generated_tokens"]) and row["generated_tokens"] <= 1
            for row in materialized.values()
        )
        and all(
            whole(row["generated_tokens"], 1)
            and row["generated_tokens"] <= materialized[identity]["generated_tokens"]
            and type(row["token_ids"]) is list
            and len(row["token_ids"]) == row["generated_tokens"]
            and all(whole(token) for token in row["token_ids"])
            and ordered(batch["consume_start_ns"], row["packet_ns"], batch["consume_end_ns"])
            for identity, row in packets.items()
        )
        and len({row["packet_ns"] for row in packets.values()}) <= 1,
        "D6 native packet generation or materialized token count differs",
    )
    return packets


def audit_step_trace(trace, client, limits, *, leaf_end_ns):
    """Verify host links. A native packet stamp is not its IPC send timestamp."""
    status = trace["status"]
    require(
        whole(trace["schema_version"], 1)
        and trace["schema_version"] == 1
        and status in {"complete", "incomplete", "not-started"}
        and client.get("d6_capture") == "step",
        "D6 raw trace schema or client capture differs",
    )
    if status != "complete":
        if status == "not-started":
            require(
                set(trace) == {"schema_version", "status"}
                and "d6_arm_return_ns" not in client
                and "measure_start_ns" not in client["phases"],
                "D6 not-started trace has a measured or armed boundary",
            )
        else:
            require(
                trace.get("overflow") is True or bool(trace.get("errors")),
                "D6 incomplete trace has no recorder failure evidence",
            )
        return {
            "verified": False,
            "status": status,
            "issues": trace.get("errors", []),
            "overflow": trace.get("overflow"),
        }

    start, end, armed, exported = (
        trace[key] for key in ("start_ns", "end_ns", "armed_ns", "export_ns")
    )
    phases = client["phases"]
    arm_return = client["d6_arm_return_ns"]
    require(
        ordered(phases["warmup_end_ns"], armed, arm_return, start, end, exported, leaf_end_ns)
        and arm_return < start
        and start - armed <= limits["arm_lead_ns"]
        and start == phases["measure_start_ns"]
        and end - start == limits["window_ns"]
        and ordered(phases["drain_end_ns"], exported)
        and trace["overflow"] is False
        and trace["errors"] == [],
        "D6 complete trace arm, window, export or recorder health differs",
    )
    enqueues, steps, batch_rows, loops, polls = (
        trace[key] for key in ("enqueues", "steps", "batches", "loops", "input_polls")
    )
    require(
        all(type(value) is list and value for value in (enqueues, steps, batch_rows))
        and len(batch_rows) <= limits["batch_limit"]
        and type(polls) is list
        and type(loops) is list
        and sum(map(len, (enqueues, steps, batch_rows, loops, polls))) <= limits["event_limit"]
        and any(row["scheduled"] for row in batch_rows),
        "D6 complete trace lacks bounded positive inventory",
    )
    batches = {row["batch_id"]: row for row in batch_rows}
    require(
        list(batches) == list(range(1, len(batch_rows) + 1))
        and all(whole(row["batch_id"], 1) for row in batch_rows),
        "D6 batch ID inventory differs",
    )
    requests = {row["request_id"]: row for row in client["requests"]}
    generations = enqueued_requests(trace, requests, start, end)
    enqueue_times = {row["request_id"]: row["end_ns"] for row in enqueues}
    engine_loops(trace, start, end, exported)
    scheduled, consumed, queue_states = step_inventory(steps, batches, start, end, exported)
    client_stamps = {identity: client_tokens(row) for identity, row in requests.items()}
    links = defaultdict(list)
    first_dispatch = {}
    for batch_id, batch in batches.items():
        packets = batch_packets(
            batch, scheduled[batch_id], consumed[batch_id], generations, enqueue_times, end
        )
        check_snapshots(batch, requests, generations, queue_states[batch_id], end)
        for row in batch["scheduled"]:
            first_dispatch.setdefault(row["request_id"], batch)
        for identity, packet in packets.items():
            ordinal = len(links[identity]) + 1
            stamps = client_stamps[identity]
            packet_ns = packet["packet_ns"]
            require(
                ordinal <= len(stamps)
                and packet_ns <= stamps[ordinal - 1][0]
                and packet["token_ids"] == [stamps[ordinal - 1][1]]
                and (not links[identity] or links[identity][-1]["packet_ns"] <= packet_ns),
                "D6 native packet ordinal or client receipt differs",
            )
            snapshot = (requests[identity].get("native_metrics") or {}).get(
                "first" if ordinal == 1 else "last" if ordinal == len(stamps) else ""
            )
            if snapshot and (ordinal == 1 or requests[identity]["status"] == "completed"):
                native_stamp = snapshot["first_token_ts" if ordinal == 1 else "last_token_ts"]
                require(round(native_stamp * NS) == packet_ns, "D6 native packet timestamp differs")
            links[identity].append(
                {
                    "ordinal": ordinal,
                    "generation": generations[identity],
                    "batch_id": batch_id,
                    "packet_ns": packet_ns,
                    "client_ns": stamps[ordinal - 1][0],
                }
            )
    for identity, request in requests.items():
        first = (request.get("native_metrics") or {}).get("first")
        if first and start <= (stamp := round(first["scheduled_ts"] * NS)) < end:
            batch = first_dispatch.get(identity)
            require(
                batch is not None
                and batch["schedule_start_ns"] <= stamp <= batch["schedule_end_ns"],
                "D6 in-window first native dispatch lacks matching scheduler batch",
            )
    require(
        all(
            sum(stamp < end for stamp, _ in stamps) <= len(links.get(identity, ()))
            for identity, stamps in client_stamps.items()
        ),
        "D6 in-window client real token lacks a native FIFO/packet link",
    )
    return {
        "verified": True,
        "status": status,
        "window_start_ns": start,
        "window_end_ns": end,
        "enqueued_requests": len(enqueues),
        "enqueue_end_ns": enqueue_times,
        "ticks": len(steps),
        "input_polls": len(polls),
        "positive_batches": sum(bool(row["scheduled"]) for row in batch_rows),
        "empty_batches": sum(not row["scheduled"] for row in batch_rows),
        "linked_real_tokens": sum(map(len, links.values())),
        "links": dict(links),
        "batches": {str(key): row for key, row in batches.items()},
        "fifo": {
            str(key): {
                "schedule_queue_before": scheduled[key]["queue_before"],
                "consume_queue_before": consumed[key]["queue_before"],
                "schedule_step_id": scheduled[key]["step_id"],
                "consume_step_id": consumed[key]["step_id"],
            }
            for key in batches
        },
    }


def packet_path(trace, previous, current):
    """Partition packet-to-packet host elapsed time, clipping overlapped submission."""
    batch = trace["batches"][str(current["batch_id"])]
    lo, hi = previous["packet_ns"], current["packet_ns"]
    stops = {
        "previous_packet_to_schedule": batch["schedule_start_ns"],
        "scheduler_call": batch["schedule_end_ns"],
        "schedule_to_execute": batch["execute_start_ns"],
        "executor_call": batch["execute_end_ns"],
        "execute_to_sample": batch["sample_start_ns"],
        "sampling_call": batch["sample_end_ns"],
        "sample_to_fifo_result": batch["future_result_start_ns"],
        "fifo_result_before_materialization": batch["materialize_start_ns"],
        "native_materialization_host_envelope": batch["materialize_end_ns"],
        "fifo_result_after_materialization": batch["future_result_end_ns"],
        "fifo_result_to_consume": batch["consume_start_ns"],
        "scheduler_consumption_to_packet": hi,
    }
    components = {}
    last = lo
    for label, stop in stops.items():
        stamp = min(hi, max(lo, stop))
        components[label] = stamp - last
        last = stamp
    components["packet_to_client_envelope_change"] = (
        current["client_ns"] - hi - (previous["client_ns"] - lo)
    )
    require(
        sum(components.values()) == current["client_ns"] - previous["client_ns"],
        "D6 host path does not close to real-token gap",
    )
    return components


VERDICTS = (
    "batch_composition_association",
    "execution_or_materialization_slowdown",
    "engine_progress_pause",
    "full_slot_amplification",
)
EXECUTION = {"executor_call", "sampling_call", "native_materialization_host_envelope"}
AFTER_READY = {
    "fifo_result_after_materialization",
    "fifo_result_to_consume",
    "scheduler_consumption_to_packet",
}
BOUNDARY_START = {
    "executor_call": "execute_start_ns",
    "sampling_call": "sample_start_ns",
    "native_materialization_host_envelope": "materialize_start_ns",
    "fifo_result_after_materialization": "materialize_end_ns",
    "fifo_result_to_consume": "future_result_end_ns",
    "scheduler_consumption_to_packet": "consume_start_ns",
}
QUEUE_BOUNDARIES = ("schedule_before", "schedule_after", "consume_before", "consume_after")


def batch_shape(batch):
    """CPU inputs and native scheduler spans, with no GPU-execution inference."""
    rows = []
    spans = {row["request_id"]: row for row in batch["scheduled"]}
    runner = batch["runner"]
    for row in runner["rows"]:
        dispatch = spans[row["request_id"]]
        effective = row["query_end"] - row["query_start"]
        prefill = max(0, min(dispatch["end"], dispatch["prompt_length"]) - dispatch["start"])
        rows.append([effective, prefill, effective - prefill])
    return {
        "effective_tokens": runner["effective_tokens"],
        "padded_tokens": runner["padded_tokens"],
        "query_prefill_decode_counts": rows,
    }


def state_signature(batch):
    return [
        [[identity for identity, _ in batch[boundary][queue]] for queue in STATE_QUEUES]
        for boundary in QUEUE_BOUNDARIES
    ]


def describe_pair(fast, slow, fast_links, slow_links, requests):
    comparisons = []
    for before, after in zip(fast_links, slow_links, strict=True):
        a, b = str(before["batch_id"]), str(after["batch_id"])
        first, second = fast["batches"][a], slow["batches"][b]
        states = [state_signature(batch) for batch in (first, second)]
        ids = [
            {row["request_id"] for row in batch["scheduled"]}
            | {identity for boundary in state for queue in boundary for identity in queue}
            for batch, state in zip((first, second), states, strict=True)
        ]
        counts = [
            sorted((rows[identity]["cached_tokens"] for identity in observed), key=str)
            for rows, observed in zip(requests, ids, strict=True)
        ]
        shared_cache_comparable = all(
            requests[0][identity]["cached_tokens"] is not None
            and requests[0][identity]["cached_tokens"] == requests[1][identity]["cached_tokens"]
            for identity in ids[0] & ids[1]
        )
        comparisons.append(
            {
                "fast_batch_id": before["batch_id"],
                "slow_batch_id": after["batch_id"],
                "fast_steps": fast["fifo"][a],
                "slow_steps": slow["fifo"][b],
                "fast_shape": batch_shape(first),
                "slow_shape": batch_shape(second),
                "same_native_schedule": [
                    {key: val for key, val in row.items() if key != "generation"}
                    for row in first["scheduled"]
                ]
                == [
                    {key: val for key, val in row.items() if key != "generation"}
                    for row in second["scheduled"]
                ],
                "same_observed_state": states[0] == states[1],
                "same_queue_depths": [[len(queue) for queue in boundary] for boundary in states[0]]
                == [[len(queue) for queue in boundary] for boundary in states[1]]
                and all(
                    fast["fifo"][a][key] == slow["fifo"][b][key]
                    for key in ("schedule_queue_before", "consume_queue_before")
                ),
                "same_cache_inventory": shared_cache_comparable
                and counts[0] == counts[1]
                and None not in counts[0]
                and all(
                    rows[identity]["native_preemptions"] == 0
                    for rows, observed in zip(requests, ids, strict=True)
                    for identity in observed
                ),
                "fast_event_offsets_ns": {
                    key: first[key] - fast["window_start_ns"] for key in BATCH_TIMES
                },
                "slow_event_offsets_ns": {
                    key: second[key] - slow["window_start_ns"] for key in BATCH_TIMES
                },
            }
        )
    return comparisons


def gap_comparisons(fast, slow, lo, hi, requests):
    """Start one bin before the detected onset and retain chronological controls."""
    rows = []
    for identity, slow_chain in slow["links"].items():
        fast_chain = fast["links"].get(identity, [])
        for left, right in zip(slow_chain, slow_chain[1:], strict=False):
            offset = right["client_ns"] - slow["window_start_ns"]
            if not lo <= offset < hi:
                continue
            ordinal = right["ordinal"]
            slow_gap = right["client_ns"] - left["client_ns"]
            row = {
                "request_id": identity,
                "generated_token_ordinal": ordinal,
                "slow_client_offset_ns": offset,
                "slow_client_gap_ns": slow_gap,
                "slow_batch_id": right["batch_id"],
                "dominant_boundary": None,
                "fast_client_gap_ns": None,
            }
            rows.append(row)
            if (
                ordinal > len(fast_chain)
                or fast_chain[ordinal - 1]["client_ns"] >= fast["window_end_ns"]
            ):
                continue
            fast_pair = fast_chain[ordinal - 2 : ordinal]
            slow_pair = (left, right)
            before, after = packet_path(fast, *fast_pair), packet_path(slow, *slow_pair)
            delta = {label: after[label] - value for label, value in before.items()}
            positive = sorted(
                ((value, label) for label, value in delta.items() if value > 0), reverse=True
            )
            # A boundary must account for more of the growth than all other
            # positive host changes combined. Contractions never add credit.
            dominant = (
                positive[0][1]
                if positive and positive[0][0] > sum(value for value, _ in positive[1:])
                else None
            )
            paired = describe_pair(fast, slow, fast_pair, slow_pair, requests)
            row.update(
                fast_client_gap_ns=sum(before.values()),
                fast_components_ns=before,
                slow_components_ns=after,
                slow_minus_fast_ns=delta,
                fast_client_offset_ns=fast_pair[-1]["client_ns"] - fast["window_start_ns"],
                fast_packet_offset_ns=fast_pair[-1]["packet_ns"] - fast["window_start_ns"],
                slow_packet_offset_ns=right["packet_ns"] - slow["window_start_ns"],
                dominant_boundary=dominant,
                batches=paired,
                same_shape=all(pair["fast_shape"] == pair["slow_shape"] for pair in paired),
                matched_state=all(
                    pair["same_cache_inventory"] and pair["same_queue_depths"] for pair in paired
                ),
                exact_composition=all(
                    pair["same_native_schedule"] and pair["same_observed_state"] for pair in paired
                ),
            )
    return sorted(
        rows,
        key=lambda row: (
            row["slow_client_offset_ns"],
            row["request_id"],
            row["generated_token_ordinal"],
        ),
    )


def queue_transitions(trace, lo, hi):
    """Observe slot release and admission using the native state either side."""
    start = trace["window_start_ns"]
    events = sorted(
        (batch[kind + "_end_ns"] - start, kind, int(identity), batch)
        for identity, batch in trace["batches"].items()
        for kind in ("schedule", "consume")
        if batch[kind + "_end_ns"] - start < hi
    )
    handoffs, series = [], []
    first_admissions, slot_holds = {}, {}
    last_release = None
    for offset, kind, identity, batch in events:
        before, after = (batch[kind + "_" + bound] for bound in ("before", "after"))
        running_before, running_after = (
            [item[0] for item in state["running"]] for state in (before, after)
        )
        waiting_before, waiting_after = (
            [item[0] for key in ("waiting", "kv_holding_waiting") for item in state[key]]
            for state in (before, after)
        )
        begin = batch[kind + "_start_ns"] - start
        step_id = trace["fifo"][str(identity)][f"{kind}_step_id"]
        if kind == "schedule":
            for request in running_after:
                if request not in running_before:
                    first_admissions.setdefault(request, (begin, offset, identity, step_id))
        else:
            for request in running_before:
                admission = first_admissions.get(request)
                if admission and request not in running_after:
                    entered, entered_by, admitted_batch, admitted_step = admission
                    slot_holds.setdefault(
                        request,
                        {
                            "admission_batch_id": admitted_batch,
                            "admission_step_id": admitted_step,
                            "release_batch_id": identity,
                            "release_step_id": step_id,
                            "duration_lower_ns": begin - entered_by,
                            "duration_upper_ns": offset - entered,
                        },
                    )
        for stamp, waiting, running in (
            (begin, waiting_before, running_before),
            (offset, waiting_after, running_after),
        ):
            series.append(
                {
                    "offset_ns": stamp,
                    "kind": kind,
                    "batch_id": identity,
                    "step_id": step_id,
                    "running": len(running),
                    "waiting": len(waiting),
                }
            )
        if (
            kind == "consume"
            and len(running_before) == 4
            and len(running_after) < 4
            and waiting_before
        ):
            last_release = {
                "release_batch_id": identity,
                "release_step_id": trace["fifo"][str(identity)]["consume_step_id"],
                "release_start_offset_ns": begin,
                "release_end_offset_ns": offset,
                "released_requests": [
                    request for request in running_before if request not in running_after
                ],
                "waiting_at_release": waiting_before,
            }
        elif kind == "consume" and len(running_before) > len(running_after):
            last_release = None
        if kind == "schedule" and last_release is not None:
            scheduled = {row["request_id"] for row in batch["scheduled"]}
            admissions = [
                request
                for request in running_after
                if request in scheduled
                and request in waiting_before
                and request in last_release["waiting_at_release"]
                and request not in running_before
            ]
            if admissions and begin >= last_release["release_end_offset_ns"]:
                for request in admissions:
                    enqueued = trace["enqueue_end_ns"].get(request)
                    if offset >= lo and enqueued is not None:
                        handoffs.append(
                            {
                                **last_release,
                                "request_id": request,
                                "admission_batch_id": identity,
                                "admission_step_id": trace["fifo"][str(identity)][
                                    "schedule_step_id"
                                ],
                                "admission_start_offset_ns": begin,
                                "admission_end_offset_ns": offset,
                                "observed_wait_until_admission_ns": start + begin - enqueued,
                                "release_end_to_admission_start_ns": begin
                                - last_release["release_end_offset_ns"],
                            }
                        )
            if set(running_after) - set(running_before):
                last_release = None
    full = [row for row in series if row["running"] == 4 and row["waiting"] > 0]
    peak = max(full, key=lambda row: row["waiting"]) if full else None
    first = full[0] if full else None
    sustained = None
    pending = None
    for row in series:
        if row["running"] != 4 or row["waiting"] == 0:
            pending = None
        elif pending is None:
            pending = row
        elif row["step_id"] != pending["step_id"]:
            sustained = pending
            break
    return {
        "first_full_with_waiting": first,
        "first_sustained_full_wait": sustained,
        "peak_full_waiting": peak,
        "release_to_admission": handoffs,
        "first_observed_slot_holds": slot_holds,
    }


def distinct_batch_pairs(rows):
    """Two distinct batch/step pairs are independent of repeated rows per batch."""
    result, seen_fast, seen_slow = [], set(), set()
    for row in rows:
        a, b = (row["batches"][-1][key] for key in ("fast_batch_id", "slow_batch_id"))
        if a in seen_fast or b in seen_slow:
            continue
        seen_fast.add(a)
        seen_slow.add(b)
        result.append(
            {
                "request_id": row["request_id"],
                "generated_token_ordinal": row["generated_token_ordinal"],
                "fast_batch_id": a,
                "slow_batch_id": b,
                "dominant_boundary": row["dominant_boundary"],
            }
        )
    return result


def assess_d6(runs, clients, traces, gates, *, heat_verified):
    signatures = {
        name: client_gap_signature(clients[name]) if name in clients else None for name in MEASURED
    }
    endpoint_pairs = {
        name: compare_pair(runs["E1"]["metrics"], runs[name]["metrics"], gates)
        for name in STEPPED
        if all(runs.get(key, {}).get("endpoint_evidence_complete") for key in ("E1", name))
    }
    result = {
        "signatures": signatures,
        "original_endpoint_tolerance_comparisons": endpoint_pairs,
        "localization": "unresolved",
        "reason": None,
        "candidate": None,
        "verdicts": {
            name: {"status": "unresolved", "reason": "comparison_not_available"}
            for name in VERDICTS
        },
        "scope": "Host envelopes only. Native packet construction is earlier than IPC publication.",
    }
    if (
        not heat_verified
        or not all(
            runs.get(name, {}).get("endpoint_evidence_complete")
            and runs[name]["native_timing"]["status"] == "complete"
            for name in MEASURED
        )
        or not all(traces.get(name, {}).get("verified") for name in STEPPED)
    ):
        result["reason"] = "thermal_request_or_trace_evidence_incomplete"
        return result
    if signatures["E1"]["candidate"] != "detected":
        result["reason"] = "endpoint_signature_not_reproduced"
        return result
    slow = [name for name in STEPPED if signatures[name]["candidate"] == "detected"]
    fast = [name for name in STEPPED if signatures[name]["candidate"] == "not_detected"]
    if len(slow) != 1 or len(fast) != 1:
        result["reason"] = "stepped_fast_and_slow_contrast_missing"
        return result
    slow_name, fast_name = slow[0], fast[0]
    slow_trace, fast_trace = traces[slow_name], traces[fast_name]
    first_bin = signatures[slow_name]["first_trigger_bin_indices"][0]
    lo, hi = max(0, first_bin - 1) * BIN_NS, (first_bin + 1) * BIN_NS
    comparisons = gap_comparisons(
        fast_trace,
        slow_trace,
        lo,
        hi,
        [runs[name]["native_timing"]["per_id"] for name in (fast_name, slow_name)],
    )
    candidates = [row for row in comparisons if row["slow_client_gap_ns"] >= MEDIAN_GATE_NS]
    if not candidates:
        result["reason"] = "triggering_client_gap_lacks_native_packet_links"
        return result
    candidate = {"slow": slow_name, "fast": fast_name, **candidates[0]}
    result["candidate"] = candidate
    result["backtrace"] = {"start_offset_ns": lo, "end_offset_ns": hi, "comparisons": comparisons}
    contrasted = [
        row
        for row in candidates
        if row["fast_client_gap_ns"] is not None
        and row["fast_client_gap_ns"] < MEDIAN_GATE_NS
        and row["matched_state"]
    ]
    exact = [row for row in contrasted if row["same_shape"] and row["exact_composition"]]
    queue_hi = (signatures[slow_name]["first_trigger_bin_indices"][-1] + 1) * BIN_NS
    fast_queue, slow_queue = (
        queue_transitions(trace, lo, queue_hi) for trace in (fast_trace, slow_trace)
    )
    result["queue_transitions"] = {fast_name: fast_queue, slow_name: slow_queue}
    growth = slow_queue["peak_full_waiting"]
    sustained = slow_queue["first_sustained_full_wait"]
    verdicts = result["verdicts"]
    for category, boundaries in (
        ("execution_or_materialization_slowdown", EXECUTION),
        ("engine_progress_pause", AFTER_READY),
    ):
        supported = []
        for boundary in sorted(boundaries):
            matches = [row for row in exact if row["dominant_boundary"] == boundary]
            rows = distinct_batch_pairs(matches)
            if len(rows) < 2:
                continue
            anchor = min(
                (
                    row["batches"][-1]["slow_event_offsets_ns"][BOUNDARY_START[boundary]],
                    row["slow_batch_id"],
                )
                for row in matches
            )
            if sustained is None or anchor[0] < sustained["offset_ns"]:
                supported.append((boundary, rows, anchor))
        if len(supported) == 1:
            boundary, rows, (anchor, batch_id) = supported[0]
            verdicts[category] = {
                "status": "supported",
                "boundary": boundary,
                "linked_gaps": rows,
                "first_boundary_offset_ns": anchor,
                "first_boundary_batch_id": batch_id,
            }
        else:
            verdicts[category]["reason"] = (
                "no_unique_repeated_comparable_boundary_before_sustained_wait"
            )

    changed_shape = distinct_batch_pairs(
        [
            row
            for row in contrasted
            if not row["same_shape"]
            and (sustained is None or row["slow_packet_offset_ns"] < sustained["offset_ns"])
        ]
    )
    shape_controls = distinct_batch_pairs(
        [
            row
            for row in comparisons
            if row.get("same_shape")
            and row.get("matched_state")
            and row["slow_client_gap_ns"] < MEDIAN_GATE_NS
            and row["fast_client_gap_ns"] < MEDIAN_GATE_NS
        ]
    )
    verdicts["batch_composition_association"].update(
        changed_shape_slow_gaps=changed_shape, same_shape_fast_controls=shape_controls
    )
    if len(changed_shape) >= 2 and len(shape_controls) >= 2:
        verdicts["batch_composition_association"]["status"] = "associated"
    else:
        verdicts["batch_composition_association"]["reason"] = (
            "repeated_shape_contrast_or_control_missing"
        )

    first_full = slow_queue["first_full_with_waiting"]
    fast_peak = fast_queue["peak_full_waiting"]
    native_by_id = {name: runs[name]["native_timing"]["per_id"] for name in (fast_name, slow_name)}
    hold = {
        fast_name: fast_queue["first_observed_slot_holds"],
        slow_name: slow_queue["first_observed_slot_holds"],
    }
    anchor = candidate.get("slow_packet_offset_ns")
    handoffs = []
    for row in slow_queue["release_to_admission"]:
        request = row["request_id"]
        fast_wait, slow_wait = (
            native_by_id[name][request]["first_components_ns"]["native_queued_to_first_scheduled"]
            for name in (fast_name, slow_name)
        )
        longer_holders = [
            {"request_id": holder, "fast": hold[fast_name][holder], "slow": hold[slow_name][holder]}
            for holder in row["released_requests"]
            if holder in hold[fast_name]
            and holder in hold[slow_name]
            and hold[slow_name][holder]["release_batch_id"] == row["release_batch_id"]
            and hold[slow_name][holder]["duration_lower_ns"]
            > hold[fast_name][holder]["duration_upper_ns"]
            and all(native_by_id[name][holder]["native_preemptions"] == 0 for name in hold)
        ]
        if anchor is not None and row["release_end_offset_ns"] >= anchor and slow_wait > fast_wait:
            handoffs.append(
                {
                    "slow": row,
                    "fast_native_wait_ns": fast_wait,
                    "slow_native_wait_ns": slow_wait,
                    "longer_holders": longer_holders,
                }
            )
    verdicts["full_slot_amplification"]["matched_handoffs"] = handoffs
    if (
        first_full
        and growth
        and anchor is not None
        and anchor <= growth["offset_ns"]
        and first_full["waiting"] < growth["waiting"]
        and (fast_peak is None or fast_peak["waiting"] < growth["waiting"])
        and any(row["longer_holders"] for row in handoffs)
    ):
        verdicts["full_slot_amplification"] = {"status": "associated", "matched_handoffs": handoffs}
    else:
        verdicts["full_slot_amplification"]["reason"] = (
            "queue_growth_or_linked_slot_handoff_not_established"
        )

    # A shape association or post-onset backlog is not proof of where a
    # slowdown began. Preserve both if the host boundaries are ambiguous.
    supported = [
        verdict["boundary"] for verdict in verdicts.values() if verdict["status"] == "supported"
    ]
    if len(supported) == 1:
        result.update(
            localization=supported[0], reason="repeated_matched_host_boundary_before_sustained_wait"
        )
    else:
        result["reason"] = "host_envelopes_do_not_isolate_one_repeated_boundary"
    return result
