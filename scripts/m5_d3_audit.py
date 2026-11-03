"""Reconstruct D3 native-clock observations without changing endpoint gates."""

import math

from vllm_prefix_scheduler.m5_campaign import is_hash, require
from vllm_prefix_scheduler.m5_contract import distribution
from vllm_prefix_scheduler.m5_d2 import D3_NATIVE_SOURCES

SECOND = 1_000_000_000
PROBE_ID = "request-000048"
NATIVE_STAMPS = ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
SNAPSHOT_FIELDS = {
    *NATIVE_STAMPS,
    "client_received_ns",
    "native_generation_tokens",
    "native_preemptions",
}
FIRST_STEPS = (
    "planned_to_submit",
    "submit_to_native_queued",
    "native_queued_to_first_scheduled",
    "native_first_scheduled_to_first_token",
    "native_first_token_to_client_first_token",
)
END_STEPS = (
    *FIRST_STEPS[:-1],
    "native_first_to_last_token",
    "native_last_token_to_client_last_token",
    "client_last_token_to_terminal",
)


def verify_native_sources(plan, source):
    native = plan["native_timing"]
    expected = native["source_sha256"]
    observed, modules = source["upstream_sha256"], source["upstream_modules"]
    require(
        native["schema_version"] == 1
        and set(expected) == D3_NATIVE_SOURCES
        and all(
            is_hash(sha)
            and observed.get(name) == sha
            and modules.get(name, {}).get("sha256") == sha
            for name, sha in expected.items()
        ),
        "D3 native timing source differs from frozen installed source",
    )


def clock_samples_valid(samples, chosen, start, end, maximum_rtt):
    if type(samples) is not list or len(samples) != 5:
        return False
    previous = start
    for row in samples:
        if (
            type(row) is not dict
            or set(row) != {"send_ns", "engine_ns", "recv_ns"}
            or not all(type(value) is int for value in row.values())
            or not previous <= row["send_ns"] <= row["engine_ns"] <= row["recv_ns"]
            or end is not None
            and row["recv_ns"] > end
        ):
            return False
        previous = row["recv_ns"]
    shortest = min(samples, key=lambda row: row["recv_ns"] - row["send_ns"])
    return chosen == shortest and shortest["recv_ns"] - shortest["send_ns"] <= maximum_rtt


def native_clock(client, engine, gates, leaf_end_ns):
    clock = client.get("native_metrics_clock")
    if type(clock) is not dict:
        return {"verified": False, "issues": ["native_clock_metadata_missing"]}
    issues = []
    sender, receiver = client["clock"], engine["clock"]
    resolution = clock.get("resolution_ns")
    if not (
        client.get("native_metrics_schema_version") == 1
        and all(
            sender[key] == receiver[key] for key in ("host", "boot_id", "time_namespace", "clock")
        )
        and sender["clock"] == "CLOCK_MONOTONIC"
        and sender["pid"] != receiver["pid"]
        and clock.get("implementation") == "clock_gettime(CLOCK_MONOTONIC)"
        and clock.get("monotonic") is True
        and clock.get("adjustable") is False
        and type(resolution) in (int, float)
        and math.isfinite(resolution)
        and 0 < resolution <= 1_000
    ):
        issues.append("native_clock_domain_or_schema_differs")
    phases = client["phases"]
    maximum = gates["max_clock_rtt_ns"]
    if not clock_samples_valid(
        client.get("clock_samples"),
        client.get("clock_check"),
        phases["load_end_ns"],
        phases["warmup_start_ns"],
        maximum,
    ):
        issues.append("native_initial_clock_brackets_invalid")
    if not clock_samples_valid(
        clock.get("post_clock_samples"),
        clock.get("post_clock_check"),
        phases["drain_end_ns"],
        leaf_end_ns,
        maximum,
    ):
        issues.append("native_post_clock_brackets_invalid")
    return {"verified": not issues, "issues": issues, "resolution_ns": resolution}


def snapshot(raw, received, generated, submit):
    if raw is None:
        return None, "native_snapshot_missing"
    if (
        type(raw) is not dict
        or set(raw) != SNAPSHOT_FIELDS
        or type(raw["client_received_ns"]) is not int
        or raw["client_received_ns"] != received
        or type(raw["native_generation_tokens"]) is not int
        or raw["native_generation_tokens"] != generated
        or type(raw["native_preemptions"]) is not int
        or raw["native_preemptions"] < 0
    ):
        return None, "native_snapshot_not_bound_to_real_chunk"
    times = [raw[key] for key in NATIVE_STAMPS]
    if not all(
        type(value) in (int, float) and math.isfinite(value) and 0 < value <= received / SECOND + 1
        for value in times
    ):
        return None, "native_clock_stamp_missing_or_invalid"
    times_ns = [round(value * SECOND) for value in times]
    if times_ns != sorted(times_ns) or not submit <= times_ns[0] <= times_ns[-1] <= received:
        return None, "native_clock_stamp_outside_request"
    return times_ns, None


def native_request(row, clock):
    planned, submit, first, last, terminal = (
        row[key]
        for key in (
            "planned_ns",
            "submit_ns",
            "first_token_ns",
            "last_token_ns",
            "end_ns",
        )
    )
    result = {
        "request_id": row["request_id"],
        "status": row["status"],
        "planned_ns": planned,
        "submit_ns": submit,
        "first_token_ns": first,
        "end_ns": terminal,
        "cached_tokens": row["cached_tokens"],
        "ttft_ns": first - planned if first is not None else None,
        "submit_to_first_token_ns": first - submit if first is not None else None,
        "first_to_terminal_ns": terminal - first
        if first is not None and terminal is not None
        else None,
        "first_components_ns": None,
        "end_components_ns": None,
        "native_preemptions": None,
        "issues": [],
    }
    issues = result["issues"]
    if first is None or not clock["verified"]:
        issues.append("no_real_first_token" if first is None else "unverified_native_clock")
        return result
    raw = row.get("native_metrics")
    if type(raw) is not dict or set(raw) != {"first", "last"}:
        issues.append("native_request_snapshot_inventory_differs")
        return result
    first_stamps, error = snapshot(
        raw["first"], first, len(row["chunk_tokens"][0]["token_ids"]), submit
    )
    if error is not None:
        issues.append(f"first_{error}")
        return result
    queued, scheduled, native_first, first_extent = first_stamps
    components = dict(
        zip(
            FIRST_STEPS,
            (
                submit - planned,
                queued - submit,
                scheduled - queued,
                native_first - scheduled,
                first - native_first,
            ),
            strict=True,
        )
    )
    require(
        sum(components.values()) == result["ttft_ns"],
        "D3 native first-token decomposition does not add up",
    )
    result["first_components_ns"] = components
    if last is None or terminal is None:
        issues.append("no_real_last_token_or_terminal")
        return result
    last_stamps, error = snapshot(raw["last"], last, len(row["token_ids"]), submit)
    if error is not None:
        issues.append(f"last_{error}")
        return result
    if (
        last_stamps[:3] != first_stamps[:3]
        or last_stamps[-1] < first_extent
        or raw["last"]["native_preemptions"] < raw["first"]["native_preemptions"]
        or terminal < last
    ):
        issues.append("native_first_last_snapshot_lineage_differs")
        return result
    native_last = last_stamps[-1]
    end_components = dict(
        zip(
            END_STEPS,
            (
                *(components[key] for key in FIRST_STEPS[:-1]),
                native_last - native_first,
                last - native_last,
                terminal - last,
            ),
            strict=True,
        )
    )
    require(
        sum(end_components.values()) == terminal - planned,
        "D3 native terminal decomposition does not add up",
    )
    result.update(
        end_components_ns=end_components, native_preemptions=raw["last"]["native_preemptions"]
    )
    return result


def timing_leaf(client, engine, inputs, gates, *, leaf_end_ns=None):
    clock = native_clock(client, engine, gates, leaf_end_ns)
    rows = {row["request_id"]: row for row in client["requests"]}
    expected = [row["request_id"] for row in inputs["requests"]]
    require(
        len(expected) == len(set(expected)) == len(rows) == len(client["requests"])
        and set(rows) == set(expected),
        "D3 native timing request denominator differs",
    )
    per_id = {identity: native_request(rows[identity], clock) for identity in expected}
    planned = len(per_id)
    first = sum(row["first_components_ns"] is not None for row in per_id.values())
    terminal = sum(row["end_components_ns"] is not None for row in per_id.values())
    return {
        "status": "complete"
        if first == terminal == planned
        else "partial"
        if first
        else "unverified",
        "clock": clock,
        "planned_count": planned,
        "first_decomposed_count": first,
        "terminal_decomposed_count": terminal,
        "per_id": per_id,
    }


def delta(a, b):
    return None if a is None or b is None else b - a


def delta_components(a, b):
    return None if a is None or b is None else {key: b[key] - value for key, value in a.items()}


def matched_timing_row(a, b):
    first = delta_components(a["first_components_ns"], b["first_components_ns"])
    total = delta(a["ttft_ns"], b["ttft_ns"])
    if first is not None:
        require(sum(first.values()) == total, "D3 same-request TTFT decomposition does not add up")
    end = delta_components(a["end_components_ns"], b["end_components_ns"])
    terminal = delta(
        a["end_ns"] - a["planned_ns"] if a["end_ns"] is not None else None,
        b["end_ns"] - b["planned_ns"] if b["end_ns"] is not None else None,
    )
    if end is not None:
        require(
            sum(end.values()) == terminal, "D3 same-request terminal decomposition does not add up"
        )
    return {
        "ttft_delta_ns": total,
        "submit_to_first_token_delta_ns": delta(
            a["submit_to_first_token_ns"], b["submit_to_first_token_ns"]
        ),
        "first_to_terminal_delta_ns": delta(a["first_to_terminal_ns"], b["first_to_terminal_ns"]),
        "plan_to_terminal_delta_ns": terminal,
        "first_component_deltas_ns": first,
        "end_component_deltas_ns": end,
    }


def p95_row(rows):
    completed = sorted(
        (row["ttft_ns"], identity) for identity, row in rows.items() if row["status"] == "completed"
    )
    return completed[(95 * len(completed) + 99) // 100 - 1] if completed else (None, None)


def terminal_tail(rows):
    finished = [row for row in rows.values() if row["end_ns"] is not None]
    if not finished:
        return None
    last = max(finished, key=lambda row: (row["end_ns"], row["request_id"]))
    last_plan = max(row["planned_ns"] for row in rows.values())
    end_components = last["end_components_ns"]
    drain = last["end_ns"] - last_plan
    parts = (
        None
        if end_components is None
        else {
            "last_terminal_request_plan_minus_last_plan": last["planned_ns"] - last_plan,
            **end_components,
        }
    )
    if parts is not None:
        require(sum(parts.values()) == drain, "D3 drain decomposition does not add up")
    return {
        "request_id": last["request_id"],
        "last_plan_to_terminal_ns": drain,
        "components_ns": parts,
    }


def timing_pair(reference, observed):
    a, b = reference["per_id"], observed["per_id"]
    require(list(a) == list(b) and len(a) == 60, "D3 paired request table differs")
    per_id = {identity: matched_timing_row(a[identity], b[identity]) for identity in a}
    a_p95, a_id = p95_row(a)
    b_p95, b_id = p95_row(b)
    last_ids = list(a)[-15:]
    last_rows = [per_id[identity] for identity in last_ids]
    ref_tail, obs_tail = terminal_tail(a), terminal_tail(b)
    return {
        "matched_request_count": len(per_id),
        "first_decomposed_pair_count": sum(
            row["first_component_deltas_ns"] is not None for row in per_id.values()
        ),
        "terminal_decomposed_pair_count": sum(
            row["end_component_deltas_ns"] is not None for row in per_id.values()
        ),
        "p95_anchors": {
            "reference_request_id": a_id,
            "observed_request_id": b_id,
            "ttft_p95_difference_ns": delta(a_p95, b_p95),
            "reference_anchor_same_id_delta": per_id.get(a_id),
            "observed_anchor_same_id_delta": per_id.get(b_id),
            "same_id_additive_contrast": per_id[a_id]
            if a_id is not None and a_id == b_id
            else None,
        },
        "archived_request_000048": per_id[PROBE_ID],
        "last_15_planned": {
            "ids": last_ids,
            "ttft_delta_ns": distribution(
                [row["ttft_delta_ns"] for row in last_rows if row["ttft_delta_ns"] is not None]
            ),
            "component_deltas_ns": {
                key: distribution(
                    [
                        row["first_component_deltas_ns"][key]
                        for row in last_rows
                        if row["first_component_deltas_ns"] is not None
                    ]
                )
                for key in FIRST_STEPS
            },
        },
        "last_terminal": {
            "reference": ref_tail,
            "observed": obs_tail,
            "last_plan_to_terminal_delta_ns": delta(
                ref_tail["last_plan_to_terminal_ns"] if ref_tail else None,
                obs_tail["last_plan_to_terminal_ns"] if obs_tail else None,
            ),
            "component_deltas_ns": delta_components(
                ref_tail["components_ns"], obs_tail["components_ns"]
            )
            if ref_tail and obs_tail
            else None,
        },
        "per_id": per_id,
    }


def assess_d3(runs, pairs, gates):
    comparable = []
    for pair in pairs:
        if (
            pair["thermal"] is not None
            and pair["thermal"]["comparable"]
            and pair["performance"] is not None
            and all(
                runs[name].get("pre_t0_verified") and runs[name]["prelude"]["verified"]
                for name in (pair["reference"], pair["observed"])
            )
        ):
            comparable.append(pair)
    breached = [pair for pair in comparable if not pair["performance"]["within_tolerance"]]
    archived = next(
        (
            pair["performance"]
            for pair in comparable
            if (pair["reference"], pair["observed"]) == ("E3", "E4")
        ),
        None,
    )
    same_direction = archived is not None and (
        not archived["ttft_within_tolerance"]
        and not archived["throughput_within_tolerance"]
        and archived["ttft_delta_ns"] < 0
        and all(
            value > gates["overhead_throughput_rel"]
            for value in archived["relative_throughput_delta"].values()
        )
    )
    return {
        "comparable_pairs": [[row["reference"], row["observed"]] for row in comparable],
        "comparable_pair_breaches": [[row["reference"], row["observed"]] for row in breached],
        "same_tier_fluctuation": "observed"
        if breached
        else "not_observed"
        if len(comparable) == len(pairs)
        else "unverified",
        "archived_e3_e4_pattern": "observed"
        if same_direction
        else "not_observed"
        if archived is not None
        else "unverified",
        "native_timing_complete": all(
            runs.get(name, {}).get("native_timing", {}).get("status") == "complete"
            for name in ("E1", "E2", "E3", "E4")
        ),
        "native_timing_scope": "first native scheduling and EngineCore output stamps; stage intervals are elapsed time, not GPU compute attribution",
        "native_stamp_conversion": "round native monotonic seconds times 1000000000 to nanoseconds",
        "qualification_started": False,
    }
