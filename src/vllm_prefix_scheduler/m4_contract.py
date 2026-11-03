"""Independent M4 raw evidence audit; standard library, never production policy.

Schema-2 collectors remain immutable. These checks reconstruct normalized native
observations and, for GPU, bind them to the existing raw runner/output bridge.
Synthetic contract fixtures test rejection paths; they never certify execution.
"""

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from vllm_prefix_scheduler import m3_contract as bridge

TASK_ID = "M4-scheduler-stability"
BASELINE = "12705a8e392eaa90c019de8e347ee31134ab3ca4"
UPSTREAM = "5df63a7203cdb9328d824a27b682e77c7b9928be"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODES = ("native", "off", "prefix", "guarded")
SEEDS = (17, 29, 43)
GPU_SCENARIOS = (
    "chunk_complete",
    "capacity_preempt_resume",
    "waiting_cancel_reuse",
    "pending_cancel_reuse",
    "full_events_probe",
)
BARRIERS = (
    "blocked_head",
    "blocked_middle",
    "partial_prefill",
    "in_flight",
    "placeholder",
    "stale",
    "preempted_zero",
    "window_tail_age",
    "stable_ties",
    "prepend_failure",
)
FAULTS = ("native_schedule_fault", "native_free_fault", "shutdown_closed")
EXECUTION_GROUPS = {
    "all": ["native", "cpu", "inputs", "gpu", "regressions"],
    "metadata": ["native", "cpu", "regressions"],
    "gpu": ["inputs", "gpu"],
}


def require(condition, message):
    if not condition:
        raise ValueError(f"M4: {message}")


def require_assertions():
    if not __debug__ or sys.flags.optimize:
        raise RuntimeError(
            "M4 validation requires assertions enabled; remove -O/-OO/PYTHONOPTIMIZE"
        )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_int(value, minimum=0):
    return type(value) is int and value >= minimum


def rows(events, kind):
    return [event for event in events if event.get("event") == kind]


def one(events, kind):
    found = rows(events, kind)
    require(len(found) == 1, f"exactly one {kind} required")
    return found[0]


def location(event):
    return {key: event.get(key) for key in ("_source", "_line", "seq", "engine_id", "case_id")}


def read_events(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    require(paths, "no raw JSONL traces")
    result = []
    for source in paths:
        for line, raw in enumerate(source.read_text().splitlines(), 1):
            require(raw.strip(), "blank raw trace line")
            value = json.loads(raw)
            require(type(value) is dict, "raw event is not an object")
            result.append({**value, "_source": str(source), "_line": line})
    require(result, "empty raw event set")
    return result


def config_gate(config):
    require_assertions()
    require(
        type(config) is dict and config.get("schema_version") == 1, "configuration schema differs"
    )
    for key, value in {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE,
        "upstream_commit": UPSTREAM,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "window": 16,
        "max_wait_ms": 500,
        "seeds": list(SEEDS),
        "repeats": 3,
        "native_rounds_per_seed": 100,
        "execution_groups": EXECUTION_GROUPS,
    }.items():
        require(
            type(config.get(key)) is type(value) and config[key] == value,
            f"frozen configuration differs: {key}",
        )
    require(
        config.get("budget")
        == {
            "max_allocations": 3,
            "gpus_per_allocation": 1,
            "max_minutes_per_allocation": 20,
            "max_requested_minutes": 60,
            "serial": True,
            "all_attempts_recorded": True,
            "no_unknown_terminal_exception": True,
        },
        "independent M4 budget differs",
    )
    expected_gpu = [
        {"scenario": scenario, "mode": mode, "repeat": repeat, "seed": seed}
        for scenario in GPU_SCENARIOS
        for mode in MODES
        for repeat, seed in enumerate(SEEDS)
    ]
    require(config.get("gpu_matrix") == expected_gpu, "complete frozen GPU matrix differs")
    native = []
    for mode in MODES:
        for seed in SEEDS:
            native.append(
                {
                    "scenario": "long_cycle",
                    "variant": "complete_cancel_reuse",
                    "mode": mode,
                    "seed": seed,
                    "rounds": 100,
                }
            )
    for scenario, variants, modes in (
        ("async_interleave", ("chunk_barrier", "capacity_preempt_resume"), MODES),
        ("barriers", BARRIERS, ("prefix", "guarded")),
        ("terminal_contract", FAULTS, ("guarded",)),
    ):
        native.extend(
            {"scenario": scenario, "variant": variant, "mode": mode, "repeat": repeat, "seed": seed}
            for variant in variants
            for mode in modes
            for repeat, seed in enumerate(SEEDS)
        )
    require(
        config.get("native_matrix", {}).get("leaves") == native,
        "complete frozen native matrix differs",
    )
    require(
        config["native_matrix"].get("rounds_per_seed") == 100
        and config["native_matrix"].get("max_steps_per_phase") == 256,
        "long-loop bounds differ",
    )
    overhead = config.get("overhead", {})
    for key, value in {
        "modes": list(MODES),
        "workloads": ["cold", "tied_hot", "mixed"],
        "windows": [8, 16, 32],
        "outside_lengths": [0, 128, 4096],
        "repeats": 5,
        "samples_per_repeat": 100,
        "warmup_samples": 20,
        "clock": "perf_counter_ns",
        "observer_io": False,
        "snapshot_io": False,
    }.items():
        require(
            type(overhead.get(key)) is type(value) and overhead[key] == value,
            f"frozen overhead configuration differs: {key}",
        )
    profile = config.get("gpu_profile", {})
    for key, value in {
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "async_scheduling": True,
        "enable_chunked_prefill": True,
        "enforce_eager": True,
        "disable_log_stats": False,
        "kv_cache_metrics": True,
    }.items():
        require(profile.get(key) == value, f"fixed GPU profile differs: {key}")
    require(
        profile.get("kv_events_config", {}).get("enable_kv_cache_events") is True,
        "full KV events must be enabled",
    )
    return config


def expected_cases(config, layer):
    config_gate(config)
    require(layer in {"native", "gpu", "cpu"}, "unknown validation layer")
    if layer == "cpu":
        return {
            f"{mode}:{workload}:{window}:{outside}:{repeat}": {
                "mode": mode,
                "workload": workload,
                "window": window,
                "outside_length": outside,
                "repeat": repeat,
            }
            for mode in MODES
            for workload in config["overhead"]["workloads"]
            for window in config["overhead"]["windows"]
            for outside in config["overhead"]["outside_lengths"]
            for repeat in range(5)
        }
    result = {}
    for leaf in config["gpu_matrix"] if layer == "gpu" else config["native_matrix"]["leaves"]:
        meta = dict(leaf)
        meta.setdefault("repeat", SEEDS.index(meta["seed"]))
        case = ":".join(
            str(meta[key])
            for key in (
                ("scenario", "mode", "repeat")
                if layer == "gpu"
                else ("scenario", "variant", "mode", "repeat")
            )
        )
        result[case] = meta
    return result


def validate_envelopes(events, layer):
    require(type(events) is list and events, "empty raw event set")
    normalized = [event for event in events if event.get("event", "").startswith("m4_")]
    if layer == "cpu":
        require(
            normalized
            and all(e.get("layer") == "cpu" and strict_int(e.get("seq"), 1) for e in normalized),
            "CPU raw envelope invalid",
        )
        require(len({e["seq"] for e in normalized}) == len(normalized), "duplicate CPU sequence")
        return normalized
    require(normalized, "no M4 raw events")
    streams = defaultdict(list)
    for event in normalized:
        require(event.get("layer") == layer, "raw layer differs")
        require(event.get("mode") in MODES, "raw mode invalid")
        for key in ("engine_id", "clock_domain"):
            require(type(event.get(key)) is str and event[key], f"missing raw {key}")
        for key in ("seq", "monotonic_ns"):
            require(strict_int(event.get(key)), f"invalid raw {key}")
        require(
            event.get("event") not in {"m4_error", "m4_skipped", "m4_observer_error"},
            "failed/skipped observation cannot pass",
        )
        streams[(event["engine_id"], event.get("pid", 0))].append(event)
    for stream in streams.values():
        ordered = sorted(stream, key=lambda e: e["seq"])
        require(len({e["seq"] for e in ordered}) == len(ordered), "duplicate raw sequence")
        require(len({e["clock_domain"] for e in ordered}) == 1, "raw clock domain changed")
        require(
            all(
                a["monotonic_ns"] <= b["monotonic_ns"]
                for a, b in zip(ordered, ordered[1:], strict=False)
            ),
            "raw clock regressed",
        )
    return normalized


def matrix_gate(config, events, layer, partial=False):
    expected = expected_cases(config, layer)
    begins, ends = rows(events, "m4_case_begin"), rows(events, "m4_case_end")
    found = [event.get("case_id") for event in begins]
    require(found and len(found) == len(set(found)), "empty/duplicate case boundaries")
    require(set(found) <= set(expected), "unexpected or mislabeled case")
    require(
        Counter(found) == Counter(event.get("case_id") for event in ends),
        "case end missing or duplicated",
    )
    if not partial:
        require(set(found) == set(expected), "entire unique frozen matrix required")
    for event in begins + ends:
        meta = expected[event["case_id"]]
        require(
            all(event.get(key) == value for key, value in meta.items() if key != "rounds"),
            "case metadata mislabeled",
        )
        require(
            type(event.get("repeat")) is int and type(event.get("seed")) is int,
            "case seed/repeat must be integers",
        )
    for event in ends:
        require(event.get("status") == "passed", "case failed/skipped/incomplete")
    return {key: expected[key] for key in found}


def audit_drained(resources, *, allow_fault=False):
    require(type(resources) is dict, "native resource checkpoint missing")
    copy = dict(resources)
    if allow_fault:
        copy["guard_fault_code"] = None
    bridge.audit_drained(copy)
    if "observer_bindings" in resources:
        require(resources["observer_bindings"] == 0, "observer retained old bindings")


def request_key(request):
    require(type(request) is dict, "actual request state missing")
    for key in ("request_id", "object_id"):
        require(type(request.get(key)) is str and request[key], f"request identity lacks {key}")
    require(strict_int(request.get("generation")), "request generation missing")
    return request["request_id"], request["generation"], request["object_id"]


def guard_map(value):
    require(type(value) is list, "guard storage missing")
    result = {}
    for record in value:
        token = record.get("object_id", record.get("object"))
        require(type(token) is str and token not in result, "guard object missing/duplicated")
        require(
            strict_int(record.get("enqueued_ns")) and strict_int(record.get("sequence")),
            "guard time/sequence invalid",
        )
        result[token] = {key: record[key] for key in ("enqueued_ns", "sequence")}
    return result


def overlap(interval, previous):
    start, end = interval
    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in previous
        if max(start, left) < min(end, right)
    )
    total, cursor = 0, start
    for left, right in clipped:
        total += max(0, right - max(left, cursor))
        cursor = max(cursor, right)
    return total


def audit_lifecycle(events, mode, *, fault_case=False):
    bindings, active_ids, generations, guards = {}, {}, {}, {}
    active_previous, last_generation = {}, -1
    intervals, pending, outputs, terminations = defaultdict(list), {}, [], {}
    schedule_order, consume_order = [], []
    preemptions, cancellations, fences, partial_states = [], [], [], []
    notifications = []
    sequences, last_clock = [], None
    retired = set()
    for event in sorted(events, key=lambda e: (e.get("pid", 0), e["seq"])):
        kind = event["event"]
        if kind == "m4_request_bound":
            request = event["request"]
            rid, generation, token = key = request_key(request)
            require(
                key not in bindings and rid not in active_ids, "concurrent/duplicate generation"
            )
            require(generation > last_generation, "generation order differs")
            last_generation = generation
            if rid in generations:
                require(
                    fences and fences[-1]["seq"] > terminations[active_previous[rid]]["seq"],
                    "ID reused before old outputs/notifications/resources drained",
                )
            generations[rid] = generation
            bindings[key], active_ids[rid] = request, key
            record = request.get("guard")
            if mode == "guarded":
                require(type(record) is dict, "new guarded generation lacks record")
                require(
                    strict_int(record.get("enqueued_ns")) and strict_int(record.get("sequence")),
                    "guard admission time/sequence invalid",
                )
                require(
                    not sequences or record["sequence"] == sequences[-1] + 1,
                    "new generation sequence refreshed/reset",
                )
                require(
                    last_clock is None or record["enqueued_ns"] >= last_clock,
                    "new generation clock regressed",
                )
                require(
                    record["enqueued_ns"] <= event["monotonic_ns"], "guard time outside real clock"
                )
                last_clock = record["enqueued_ns"]
                sequences.append(record["sequence"])
                guards[token] = {k: record[k] for k in ("enqueued_ns", "sequence")}
            else:
                require(record is None, "off/native/prefix maintained guard metadata")
        elif kind == "m4_schedule":
            batch = event.get("batch_id")
            require(
                type(batch) is str and batch not in pending and batch not in schedule_order,
                "batch missing/duplicate",
            )
            if "guard_before" in event:
                require(guard_map(event["guard_before"]) == guards, "guard before dispatch drift")
            batch_intervals = event.get("intervals")
            require(type(batch_intervals) is list, "dispatch intervals missing")
            for interval in batch_intervals:
                key = request_key(interval)
                require(
                    key in bindings and active_ids.get(key[0]) == key, "dispatch generation unbound"
                )
                start, end, prompt = (interval.get(k) for k in ("start", "end", "prompt_length"))
                require(
                    strict_int(start)
                    and strict_int(end, 1)
                    and start < end
                    and strict_int(prompt, 1),
                    "dispatch interval invalid",
                )
                intervals[key].append({**interval, "seq": event["seq"], "batch_id": batch})
                guards.pop(key[2], None)
            require(
                len({r["request_id"] for r in batch_intervals}) == len(batch_intervals),
                "duplicate generation interval in one batch",
            )
            if "num_scheduled_tokens" in event:
                require(
                    event["num_scheduled_tokens"]
                    == {r["request_id"]: r["end"] - r["start"] for r in batch_intervals},
                    "normalized token map differs from dispatch intervals",
                )
            for name in ("finished_req_ids", "preempted_req_ids"):
                require(
                    type(event.get(name)) is list and len(set(event[name])) == len(event[name]),
                    "native terminal/preemption notification missing or duplicated",
                )
            notifications.append(event)
            pending[batch] = event
            schedule_order.append(batch)
            if "guard_after" in event:
                require(
                    guard_map(event["guard_after"]) == guards,
                    "first successful positive dispatch did not clear only its guard",
                )
            for state in event.get("states", []):
                if 0 < state.get("num_computed_tokens", 0) < state.get("prompt_length", 0):
                    partial_states.append(state)
        elif kind == "m4_consume_start":
            batch = event.get("batch_id")
            require(batch in pending, "orphan/duplicate output consume")
            require(
                event.get("intervals") == pending[batch].get("intervals"),
                "consume binding differs from original generation intervals",
            )
            consume_order.append(batch)
        elif kind == "m4_consume_success":
            batch = event.get("batch_id")
            require(
                batch in pending and consume_order and consume_order[-1] == batch,
                "consume success lacks ordered start",
            )
            require(
                event.get("intervals") == pending[batch].get("intervals"),
                "consume changed original interval generation",
            )
            outputs.extend(event.get("outputs", []))
            del pending[batch]
        elif kind == "m4_request_terminal":
            request = event.get("request")
            key = request_key(request)
            require(
                key in bindings and active_ids.get(key[0]) == key, "terminal generation differs"
            )
            terminations[key] = event
            del active_ids[key[0]]
            guards.pop(key[2], None)
            retired.add(key)
            active_previous = {k[0]: k for k in terminations}
        elif kind == "m4_cancel_start":
            for request in event.get("requests", [event.get("request")]):
                if request is None:
                    continue
                requested = event.get("requested_ids")
                wanted = [requested] if type(requested) is str else requested
                if wanted is not None and request["request_id"] not in wanted:
                    continue
                key = request_key(request)
                require(key in bindings, "cancel target was never bound")
                observed = event.get("pending")
                require(type(observed) is list, "cancel pending observation missing")
                require(
                    {p["batch_id"] for p in observed} == set(pending),
                    "cancel pending set differs from raw FIFO",
                )
                cancellations.append({"request": request, "pending": observed, "event": event})
        elif kind == "m4_preempt_success":
            before, after = event.get("before"), event.get("request", event.get("after"))
            require(type(before) is dict, "preemption before-state missing")
            key = request_key(after)
            require(key == request_key(before) and key in bindings, "preemption changed generation")
            require(
                after.get("num_preemptions") == before.get("num_preemptions", -1) + 1
                and after.get("num_computed_tokens") == 0
                and after.get("status") == "PREEMPTED",
                "native preemption transition missing",
            )
            guards.pop(key[2], None)
            preemptions.append({"key": key, "event": event})
        elif kind == "m4_generation_drained":
            require(
                not active_ids and not pending and not guards, "reuse fence precedes causal drain"
            )
            audit_drained(event.get("resources"), allow_fault=fault_case)
            fences.append(event)
    require(schedule_order == consume_order, "native output FIFO missing/reordered")
    require(not pending, "native output FIFO remains pending")
    require(not active_ids and not guards, "active requests/guard not drained")
    require(len(terminations) == len(bindings), "request terminal observations incomplete")
    computed = {}
    for key, values in intervals.items():
        seen, repeated, prompt_tokens = [], 0, 0
        for value in values:
            segment = (
                min(value["start"], value["prompt_length"]),
                min(value["end"], value["prompt_length"]),
            )
            prompt_tokens += segment[1] - segment[0]
            repeated += overlap(segment, seen)
            seen.append(segment)
        computed[key] = {
            "intervals": values,
            "repeated_prompt_tokens": repeated,
            "prompt_tokens": prompt_tokens,
        }
    return {
        "bindings": bindings,
        "dispatch": computed,
        "preemptions": preemptions,
        "notifications": notifications,
        "cancellations": cancellations,
        "fences": fences,
        "terminations": terminations,
        "partial_states": partial_states,
        "batches": len(schedule_order),
        "empty_batches": len(schedule_order)
        - sum(bool(v["intervals"]) for v in events if v["event"] == "m4_schedule"),
        "guard_registrations": len(sequences),
        "outputs": outputs,
    }


def audit_probe(events):
    before, after = one(events, "m4_probe_before"), one(events, "m4_probe_after")
    require(
        before.get("probe_id") == after.get("probe_id") and before["seq"] < after["seq"],
        "probe pairing/order differs",
    )
    for event in (before, after):
        flags = event.get("flags", {})
        require(
            all(flags.get(name) is True for name in ("full", "events", "stats", "metrics")),
            "full events/stats/metrics probe profile not active",
        )
        snapshot = event.get("snapshot")
        require(type(snapshot) is dict and snapshot, "probe storage snapshot missing")
    require(before["snapshot"] == after["snapshot"], "probe changed native storage")
    require(strict_int(after.get("score")), "probe result missing")
    return {"before": location(before), "after": location(after), "score": after["score"]}


def audit_chunk(lifecycle):
    found = []
    for key, record in lifecycle["dispatch"].items():
        portions = [row for row in record["intervals"] if row["start"] < row["prompt_length"]]
        unique = {(row["start"], min(row["end"], row["prompt_length"])) for row in portions}
        partial = any(request_key(state) == key for state in lifecycle["partial_states"])
        terminal = lifecycle["terminations"].get(key, {}).get("request", {})
        if (
            len(unique) >= 2
            and partial
            and terminal.get("status") in {"FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED"}
        ):
            found.append(
                {"request_id": key[0], "generation": key[1], "prefill_intervals": sorted(unique)}
            )
    require(found, "actual multi-chunk partial-prefill completion not observed")
    return found


def audit_preemption(lifecycle):
    found = []
    for preempt in lifecycle["preemptions"]:
        key, event = preempt["key"], preempt["event"]
        record = lifecycle["dispatch"].get(key, {})
        require(
            any(
                row["seq"] > event["seq"] and key[0] in row["preempted_req_ids"]
                for row in lifecycle["notifications"]
            ),
            "native preemption notification never consumed",
        )
        require(
            record.get("repeated_prompt_tokens", 0) > 0,
            "preemption lacks actual repeated prompt interval",
        )
        terminal = lifecycle["terminations"].get(key, {}).get("request", {})
        require(
            terminal.get("status") in {"FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED"},
            "preempted request never completed",
        )
        require(
            any(row["seq"] > event["seq"] for row in record["intervals"]),
            "preempted request never resumed",
        )
        found.append(
            {
                "request_id": key[0],
                "generation": key[1],
                "repeated_prompt_tokens": record["repeated_prompt_tokens"],
                "preempt": location(event),
            }
        )
    require(found, "actual capacity preemption/resume missing")
    return found


def audit_cancellation(lifecycle, scenario):
    found = []
    for cancel in lifecycle["cancellations"]:
        request = cancel["request"]
        key = request_key(request)
        dispatch = lifecycle["dispatch"].get(key, {}).get("intervals", [])
        if scenario == "waiting_cancel_reuse":
            valid = request.get("status") == "WAITING" and not dispatch
        else:
            matched = [
                batch
                for batch in cancel["pending"]
                if any(request_key(interval) == key for interval in batch["intervals"])
            ]
            valid = (
                request.get("status") == "RUNNING"
                and bool(matched)
                and (
                    request.get("num_in_flight_tokens", 0) > 0
                    or request.get("num_output_placeholders", 0) > 0
                )
            )
        terminal = lifecycle["terminations"].get(key)
        if valid and terminal and terminal["request"].get("status") == "FINISHED_ABORTED":
            later = [
                other for other in lifecycle["bindings"] if other[0] == key[0] and other[1] > key[1]
            ]
            require(later, "cancelled ID was not safely reused")
            require(
                any(
                    lifecycle["terminations"][other]["request"]["status"]
                    in {"FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED"}
                    for other in later
                ),
                "request after cancellation/reuse did not complete",
            )
            found.append({"generation": key[1], "cancel": location(cancel["event"])})
    require(found, "required real cancellation state/pending not observed")
    return found


def audit_rounds(events, metadata):
    begins, ends = rows(events, "m4_round_begin"), rows(events, "m4_round_end")
    require(
        [e.get("round") for e in begins] == list(range(100))
        and [e.get("round") for e in ends] == list(range(100)),
        "all 100 long-loop rounds must be individually observed",
    )
    for begin, end in zip(begins, ends, strict=True):
        require(begin["seq"] < end["seq"], "round order differs")
        audit_drained(end.get("resources"))
        alive = end.get("weak_refs_alive")
        require(
            alive == [] or type(alive) is int and alive == 0,
            "historical request weak references remain alive",
        )
        cards = end.get("observer_cardinalities")
        require(type(cards) is dict and cards, "observer storage cardinalities missing")
        require(
            all(
                strict_int(value) and value == 0
                for key, value in cards.items()
                if key not in {"registration_sequence", "next_sequence", "max_pending", "max_live"}
            ),
            "observer metadata grows with historical rounds",
        )
        require(
            cards.get("max_pending", 0) <= 3 and cards.get("max_live", 0) <= 4,
            "long-loop active observer state exceeded fixture bound",
        )
        local = [e for e in events if begin["seq"] < e["seq"] < end["seq"]]
        terminal = rows(local, "m4_request_terminal")
        completed = sum(
            e["request"].get("status") in {"FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED"}
            for e in terminal
        )
        aborted = sum(e["request"].get("status") == "FINISHED_ABORTED" for e in terminal)
        require(completed >= 2 and aborted >= 2, "round lacks complete/cancel/reuse combination")
        cancels = rows(local, "m4_cancel_start")
        states = [request for event in cancels for request in event.get("requests", [])]
        require(any(r.get("status") == "WAITING" for r in states), "round lacks waiting cancel")
        require(
            any(
                r.get("status") == "RUNNING"
                and (
                    r.get("num_in_flight_tokens", 0) > 0 or r.get("num_output_placeholders", 0) > 0
                )
                for r in states
            ),
            "round lacks actual pending cancel",
        )
        require(rows(local, "m4_generation_drained"), "round lacks safe reuse fence")
    return {"seed": metadata["seed"], "rounds": len(ends), "last": location(ends[-1])}


def audit_barrier(events, variant):
    observations = rows(events, "m4_barrier_check")
    require(observations, "barrier raw observation missing")
    for event in observations:
        before, after, candidates = (event.get(key) for key in ("before", "after", "candidates"))
        require(
            type(before) is list and type(after) is list and type(candidates) is list,
            "barrier queue observations missing",
        )
        before_ids = [request_key(r) for r in before]
        after_ids = [request_key(r) for r in after]
        candidate_ids = [request_key(r) for r in candidates]
        require(Counter(before_ids) == Counter(after_ids), "barrier lost/duplicated native object")
        require(
            before_ids[: len(candidate_ids)] == candidate_ids,
            "candidate window crossed non-contiguous barrier",
        )
        require(len(candidates) <= 16, "window exceeded frozen bound")
        require(
            after_ids[len(candidate_ids) :] == before_ids[len(candidate_ids) :],
            "window changed outside suffix",
        )
        if variant == "prepend_failure":
            require(
                event.get("injected") is True
                and event.get("exception")
                and event.get("restored") is True
                and before_ids == after_ids,
                "injected prepend failure did not restore native deque",
            )
        elif variant in {"stable_ties", "window_tail_age"}:
            if variant == "stable_ties":
                require(before_ids == after_ids, "same-score FIFO changed")
            require(
                event.get("guard_before") == event.get("guard_after"),
                "barrier inspection refreshed guard age/sequence",
            )
            if variant == "window_tail_age":
                entered = one(events, "m4_tail_entered")
                require(
                    entered["seq"] > event["seq"], "tail entry precedes outside-window observation"
                )
                tail_keys = before_ids[len(candidate_ids) :]
                require(tail_keys, "window-tail case has no outside objects")
                tail_tokens = {key[2] for key in tail_keys}
                prior_records = [
                    record
                    for record in event["guard_after"]
                    if record.get("object_id") in tail_tokens
                ]
                require(
                    entered.get("records_before") == entered.get("records_after") == prior_records,
                    "tail entry changed original outside-object guard age/sequence",
                )
                windows = [
                    row
                    for row in rows(events, "m4_window")
                    if event["seq"] < row["seq"] < entered["seq"]
                ]
                require(
                    windows
                    and [request_key(r) for r in windows[-1].get("waiting", [])] == tail_keys
                    and [request_key(r) for r in windows[-1].get("candidates", [])]
                    == tail_keys[:16],
                    "tail entry is not bound to earlier outside-window objects",
                )
        else:
            require(len(candidates) < len(before), "barrier never stopped candidate window")
            barrier = before[len(candidates)]
            condition = {
                "blocked_head": barrier.get("status") != "WAITING",
                "blocked_middle": barrier.get("status") != "WAITING",
                "partial_prefill": barrier.get("num_computed_tokens", 0) > 0,
                "in_flight": barrier.get("num_in_flight_tokens", 0) > 0,
                "placeholder": barrier.get("num_output_placeholders", 0) > 0,
                "stale": barrier.get("num_stale_output_tokens", 0) > 0,
                "preempted_zero": barrier.get("num_preemptions", 0) > 0
                and barrier.get("num_computed_tokens") == 0,
            }.get(variant, False)
            require(condition, "declared native barrier state not present")
    return {"checks": len(observations), "first": location(observations[0])}


def source_gate(sources, config, layer, *, root=None):
    require(type(sources) is dict, "source/runtime identity missing")
    commit = sources.get("project_commit")
    require(
        bridge.is_hash(commit, 40) and sources.get("project_dirty") is False,
        "source checkout must be clean/frozen",
    )
    require(
        sources.get("python_version") == "3.12.16"
        and sources.get("packages", {}).get("vllm") == "0.32.0"
        and sources.get("upstream_commit") == UPSTREAM,
        "fixed native runtime/source differs",
    )
    hashes, upstream = sources.get("source_sha256"), sources.get("upstream_sha256")
    if upstream is None and type(sources.get("upstream_modules")) is dict:
        upstream = {name: row.get("sha256") for name, row in sources["upstream_modules"].items()}
    require(
        type(hashes) is dict and hashes and all(bridge.is_hash(v) for v in hashes.values()),
        "project source hashes missing",
    )
    require(
        type(upstream) is dict
        and upstream
        and all(
            upstream.get(key) == value for key, value in config["extra_upstream_sha256"].items()
        ),
        "actual fixed upstream source hashes differ",
    )
    critical = {
        f"src/vllm_prefix_scheduler/{name}.py"
        for name in ("scheduler", "probe", "wait_guard", "config", "m4_contract", "m4_observer")
    }
    require(critical <= set(hashes), "critical M4 sources missing")
    if root is not None:
        frozen_config = subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "blob", f"{commit}:configs/m4-stability.json"]
        )
        require(
            json.loads(frozen_config) == config,
            "supplied configuration differs from tested source freeze",
        )
        names = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-tree",
                "-r",
                "--name-only",
                commit,
                "--",
                "src",
                "scripts",
                "configs",
                "tests",
                "Makefile",
            ],
            text=True,
        ).splitlines()
        require(set(names) == set(hashes), "frozen source catalog differs from Git tree")
        if config.get("production_policy_change") is False:
            for module in ("scheduler", "config", "probe", "wait_guard"):
                name = f"src/vllm_prefix_scheduler/{module}.py"
                baseline_blob = subprocess.check_output(
                    ["git", "-C", str(root), "cat-file", "blob", f"{BASELINE}:{name}"]
                )
                require(
                    hashlib.sha256(baseline_blob).hexdigest() == hashes.get(name),
                    "production changed without expanded same-freeze regression scope",
                )
        for name in names:
            blob = subprocess.check_output(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{name}"]
            )
            require(
                hashlib.sha256(blob).hexdigest() == hashes[name],
                "source differs from frozen Git blob",
            )
    if layer == "gpu":
        model = sources.get("model", {})
        require(
            model.get("revision") == model.get("tokenizer_revision") == REVISION,
            "actual model/tokenizer revision differs",
        )
        files = model.get("files", [])
        require(
            len(files) == len(config["model_files"])
            and {e.get("name") for e in files} == set(config["model_files"]),
            "actual model/tokenizer catalog missing",
        )
        for record in files:
            wanted = config["model_files"][record["name"]]
            require(
                record.get("size") == wanted["size"]
                and record.get("verified_content_id") == wanted["content_id"]
                and bridge.is_hash(record.get("sha256")),
                "actual model content identity differs",
            )
            if wanted["kind"] == "lfs_sha256":
                require(record["sha256"] == wanted["content_id"], "actual model LFS hash differs")
    return commit


def audit_gpu_bridge(events, normalized):
    bridge_rows = [e for e in events if not e["event"].startswith("m4_")]
    schedules = rows(bridge_rows, "schedule_success")
    actual = rows(normalized, "m4_schedule")
    require(
        len(schedules) == len(actual) and schedules, "M4 schedules not bound to complete GPU bridge"
    )
    by_batch = {e["batch_id"]: e for e in schedules}
    for event in actual:
        raw = by_batch.get(event["batch_id"])
        require(raw is not None, "M4 scheduled batch absent from GPU raw bridge")
        for key in ("request_id", "start", "end", "prompt_length"):
            require(
                [row[key] for row in raw["intervals"]] == [row[key] for row in event["intervals"]],
                "M4 generation interval differs from actual native schedule",
            )
        require(
            raw.get("finished_req_ids") == event.get("finished_req_ids"),
            "M4 terminal notification differs from actual native schedule",
        )
    returned = {e["batch_id"]: e for e in rows(bridge_rows, "update_from_output_success")}
    for event in rows(normalized, "m4_consume_success"):
        require(
            event["batch_id"] in returned
            and event.get("outputs") == returned[event["batch_id"]].get("engine_outputs"),
            "M4 outputs differ from actual native GPU return",
        )
    result = bridge.audit_batches(bridge_rows)
    result.pop("dispatch_by_request", None)
    return result


def audit_case(events, metadata, layer):
    begin, end = one(events, "m4_case_begin"), one(events, "m4_case_end")
    require(begin["seq"] < end["seq"], "case order invalid")
    scenario, mode = metadata["scenario"], metadata["mode"]
    fault = scenario == "terminal_contract"
    lifecycle = audit_lifecycle(events, mode, fault_case=fault)
    result = {
        "begin": location(begin),
        "end": location(end),
        "requests": len(lifecycle["bindings"]),
        "batches": lifecycle["batches"],
        "empty_batches": lifecycle["empty_batches"],
        "guard_registrations": lifecycle["guard_registrations"],
    }
    if layer == "gpu":
        require(
            set(begin.get("interventions", []))
            <= {"idle_native_cache_reset", "native_pause_keep", "normal_client_abort"},
            "unsupported GPU intervention",
        )
        require(begin.get("initial_resources") is not None, "independent initial state missing")
        audit_drained(begin["initial_resources"])
        audit_drained(end.get("resources"))
        if scenario == "chunk_complete":
            result["chunk"] = audit_chunk(lifecycle)
        elif scenario == "capacity_preempt_resume":
            result["preemption"] = audit_preemption(lifecycle)
        elif scenario in {"waiting_cancel_reuse", "pending_cancel_reuse"}:
            result["cancellation"] = audit_cancellation(lifecycle, scenario)
        elif scenario == "full_events_probe":
            result["probe"] = audit_probe(events)
    elif scenario == "long_cycle":
        result["long_cycle"] = audit_rounds(events, metadata)
    elif scenario == "async_interleave":
        if metadata["variant"] == "chunk_barrier":
            result["chunk"] = audit_chunk(lifecycle)
        else:
            result["preemption"] = audit_preemption(lifecycle)
    elif scenario == "barriers":
        result["barrier"] = audit_barrier(events, metadata["variant"])
    elif fault:
        terminal = one(events, "m4_terminal_check")
        require(
            terminal.get("retry_rejected") == ["schedule", "add"],
            "fault/closed scheduler accepted retry",
        )
        if metadata["variant"] == "shutdown_closed":
            require(terminal.get("closed") is True, "shutdown did not close guard")
        if metadata["variant"] != "shutdown_closed":
            require(
                type(terminal.get("fault_code")) is str and terminal["fault_code"],
                "expected native lifecycle fault not retained",
            )
        audit_drained(terminal.get("resources"), allow_fault=True)
    return result


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def audit_cpu(config, events, *, partial=False):
    expected = expected_cases(config, "cpu")
    all_samples = rows(events, "m4_cpu_sample")
    samples = [e for e in all_samples if e.get("variant") == "normal"]
    recoveries = [e for e in all_samples if e.get("variant") == "prepend_restore"]
    require(len(samples) + len(recoveries) == len(all_samples), "unknown CPU sample variant")
    require(samples, "CPU samples missing")
    grouped = defaultdict(list)
    for event in samples:
        key = ":".join(
            str(event.get(k)) for k in ("mode", "workload", "window", "outside_length", "repeat")
        )
        require(key in expected, "unexpected CPU matrix sample")
        require(
            event.get("observer_io") is False and event.get("snapshot_io") is False,
            "diagnostic observer IO contaminated CPU samples",
        )
        require(strict_int(event.get("sample")), "CPU sample index missing")
        duration = event.get("duration_ns")
        require(
            type(duration) is dict and duration and all(strict_int(v) for v in duration.values()),
            "CPU duration samples invalid",
        )
        for field in ("calls", "visits"):
            require(
                type(event.get(field)) is dict
                and all(strict_int(v) for v in event[field].values()),
                "CPU raw operation counts missing",
            )
        require(
            set(duration)
            == {"window", "score", "sort_exclusive", "commit", "restore_residual", "total"},
            "CPU phase catalog missing",
        )
        require(
            set(event["calls"])
            == {"window", "score", "sort", "pop", "prepend", "guard_clock", "native_schedule"}
            and set(event["visits"]) == {"candidate", "queue", "hash_lookup", "pop", "prepend"},
            "CPU raw operation catalog missing",
        )
        require(
            event.get("guard_present") is (event["mode"] == "guarded"),
            "CPU protection storage differs",
        )
        if event["mode"] in {"native", "off"}:
            require(
                event["calls"]["native_schedule"] == 1
                and event["calls"]["guard_clock"] == 0
                and event["calls"]["window"] == event["calls"]["score"] == 0,
                "off/native measured baseline read strategy/guard clock",
            )
        else:
            require(
                event["calls"]["window"] == event["calls"]["sort"] == 1
                and event["calls"]["score"] == event["window"]
                and event["visits"]["candidate"] == event["visits"]["queue"] == event["window"]
                and event["calls"]["guard_clock"] == (1 if event["mode"] == "guarded" else 0),
                "CPU candidate/score/clock operations differ from bounded transaction",
            )
        grouped[key].append(event)
    if not partial:
        require(set(grouped) == set(expected), "complete CPU matrix missing")
    summaries = []
    count_profiles = defaultdict(dict)
    for key, records in grouped.items():
        require(
            sorted(e["sample"] for e in records) == list(range(100)),
            "CPU sample count missing/duplicated",
        )
        meta = expected[key]
        durations = {
            phase: [e["duration_ns"].get(phase) for e in records]
            for phase in records[0]["duration_ns"]
        }
        require(
            all(all(strict_int(v) for v in values) for values in durations.values()),
            "CPU phases differ across samples",
        )
        summaries.append(
            {
                **meta,
                "samples": len(records),
                "phases": {
                    phase: {"p50_ns": percentile(values, 0.50), "p95_ns": percentile(values, 0.95)}
                    for phase, values in durations.items()
                },
            }
        )
        profile = tuple(meta[k] for k in ("mode", "workload", "window", "repeat"))
        count_profiles[profile][meta["outside_length"]] = {
            "calls": [e["calls"] for e in records],
            "visits": [e["visits"] for e in records],
        }
    for profile in count_profiles.values():
        if len(profile) == 3:
            require(
                profile[0] == profile[128] == profile[4096],
                "window-outside queue growth increased candidate accesses",
            )
    environment = one(events, "m4_cpu_environment")
    require(
        environment.get("clock", {}).get("name") == "perf_counter_ns"
        and environment["clock"].get("monotonic") is True
        and type(environment.get("machine")) is str
        and environment["machine"]
        and environment.get("unit"),
        "CPU machine/clock/unit missing",
    )
    require(environment.get("empty_wrapper_ns") is not None, "timer/wrapper overhead missing")
    unwrapped = rows(events, "m4_cpu_unwrapped")
    require(
        len(unwrapped) == 108
        and all(
            len(e.get("samples_ns", [])) == 100 and all(strict_int(v) for v in e["samples_ns"])
            for e in unwrapped
        ),
        "unwrapped/native-off schedule baselines missing",
    )
    require(
        len(recoveries) == 90
        and all(
            e.get("restored") is True
            and e["mode"] in {"prefix", "guarded"}
            and e["workload"] == "mixed"
            and e["observer_io"] is False
            and e["snapshot_io"] is False
            for e in recoveries
        ),
        "CPU restore branch matrix missing",
    )
    expected_unwrapped = {
        (mode, workload, window, outside)
        for mode in MODES
        for workload in ("cold", "tied_hot", "mixed")
        for window in (8, 16, 32)
        for outside in (0, 128, 4096)
    }
    require(
        Counter(
            tuple(e[k] for k in ("mode", "workload", "window", "outside_length")) for e in unwrapped
        )
        == Counter({key: 1 for key in expected_unwrapped}),
        "unwrapped CPU baseline duplicated/missing",
    )
    expected_recovery = {
        (mode, "mixed", window, outside, repeat)
        for mode in ("prefix", "guarded")
        for window in (8, 16, 32)
        for outside in (0, 128, 4096)
        for repeat in range(5)
    }
    require(
        Counter(
            tuple(e[k] for k in ("mode", "workload", "window", "outside_length", "repeat"))
            for e in recoveries
        )
        == Counter({key: 1 for key in expected_recovery}),
        "CPU recovery groups duplicated/missing",
    )
    recovery_counts = rows(events, "m4_cpu_restore_counter")
    require(
        Counter(
            tuple(e[k] for k in ("mode", "workload", "window", "outside_length", "repeat"))
            for e in recovery_counts
        )
        == Counter({key: 1 for key in expected_recovery}),
        "CPU recovery operation-count groups missing",
    )
    for event in recovery_counts:
        require(
            event.get("counter_only") is True
            and event.get("timed_sample") is False
            and event.get("counts")
            == {
                "popleft_calls": 1,
                "popleft_entries": 1,
                "extendleft_calls": 1,
                "extendleft_entries": event["window"],
            },
            "native deque recovery counts incomplete or unbounded",
        )
    wrapper = environment["empty_wrapper_ns"]
    require(
        type(wrapper) is dict
        and set(wrapper) == {"bare_call_ns", "timed_wrapper_call_ns", "clock_pair_ns"}
        and all(
            type(values) is list and len(values) == 1000 and all(strict_int(v) for v in values)
            for values in wrapper.values()
        ),
        "actual clock/wrapper cost samples missing",
    )
    return {
        "groups": summaries,
        "environment": environment,
        "scope": "local native CPU only; not end-to-end throughput/TTFT",
    }


def audit(
    config,
    events,
    *,
    layer,
    sources=None,
    resources=None,
    partial=False,
    contract_fixture=False,
    root=None,
    frozen_inputs=None,
    inputs_sha256=None,
):
    config_gate(config)
    normalized = validate_envelopes(events, layer)
    commit = None if contract_fixture else source_gate(sources, config, layer, root=root)
    if layer == "cpu":
        detail = audit_cpu(config, normalized, partial=partial)
        cases = {}
    else:
        cases = matrix_gate(config, normalized, layer, partial=partial)
        detail = {
            case: audit_case(
                [event for event in normalized if event.get("case_id") == case], metadata, layer
            )
            for case, metadata in cases.items()
        }
    result = {
        "task_id": TASK_ID,
        "layer": layer,
        "scope": "contract_fixture"
        if contract_fixture
        else "partial_precheck"
        if partial
        else layer,
        "accepted": not partial and not contract_fixture,
        "complete_m4_acceptance": False,
        "project_commit": commit,
        "case_count": len(cases),
        "details": detail,
    }
    if layer == "native" and not contract_fixture:
        result["native_instances"] = audit_native_instances(normalized, sources, config)
    if layer == "gpu":
        if not contract_fixture:
            result["runtime"] = audit_gpu_runtime(
                events, config, sources, frozen_inputs, inputs_sha256
            )
        result["gpu_bridge"] = audit_gpu_bridge(events, normalized)
        if not partial and not contract_fixture:
            require(
                config.get("profile_status") == "calibrated_frozen",
                "candidate pressure profile cannot pass formal GPU gate",
            )
            require(resources is not None, "formal GPU resources/terminal evidence missing")
            jobs = {str(e.get("campaign_job_id")) for e in events}
            require(
                len(jobs) == 1 and next(iter(jobs)).isdigit(),
                "GPU raw events lack one actual job binding",
            )
            result["resources"] = audit_resources(
                resources,
                config,
                selected_commit=commit,
                selected_job=next(iter(jobs)),
                required_steps=EXECUTION_GROUPS["gpu"],
            )
    return result


def read_reference(reference, label):
    require(
        type(reference) is dict
        and type(reference.get("path")) is str
        and bridge.is_hash(reference.get("sha256")),
        f"{label} original reference missing",
    )
    path = Path(reference["path"])
    require(
        path.is_file() and not path.is_symlink() and sha256(path) == reference["sha256"],
        f"{label} original missing/changed",
    )
    return path


def load_reference(reference, label):
    return json.loads(read_reference(reference, label).read_text())


def audit_resources(
    resources,
    config,
    *,
    selected_commit=None,
    selected_job=None,
    required_steps=None,
    selected_jobs=None,
):
    """M4 independent all-attempt budget; no historical unknown-terminal exception."""
    config_gate(config)
    selection = None
    if selected_jobs is not None:
        require(
            type(selected_jobs) is dict
            and set(selected_jobs) == {"metadata", "gpu"}
            and all(type(job) is str and job.isdigit() for job in selected_jobs.values())
            and len(set(selected_jobs.values())) == 2,
            "final selection requires two distinct metadata/gpu jobs",
        )
        require(
            selected_commit is not None and selected_job is None and required_steps is None,
            "final job selection cannot mix single-job options",
        )
        selection = {job: role for role, job in selected_jobs.items()}
    else:
        required_steps = EXECUTION_GROUPS["all"] if required_steps is None else required_steps
        require(
            type(required_steps) in {list, tuple}
            and list(required_steps) in EXECUTION_GROUPS.values(),
            "required execution steps are not a fixed group",
        )
        required_group = next(
            role for role, steps in EXECUTION_GROUPS.items() if steps == list(required_steps)
        )
        require(
            selected_job is None
            or selected_commit is not None
            and type(selected_job) is str
            and selected_job.isdigit(),
            "selected job lacks valid frozen commit binding",
        )
    require(
        type(resources) is dict and resources.get("task_id") == TASK_ID,
        "resource task identity differs",
    )
    allocations = resources.get("allocations")
    require(
        type(allocations) is list and 1 <= len(allocations) <= 3,
        "M4 allocation budget missing/exceeded",
    )
    ledger = load_reference(resources.get("submission_ledger"), "submission ledger")
    require(ledger.get("task_id") == TASK_ID, "ledger task identity differs")
    attempts = ledger.get("attempts")
    require(type(attempts) is list and attempts, "submission attempt ledger empty")
    require(
        [a.get("attempt_id") for a in attempts]
        == [f"attempt-{i:02d}" for i in range(1, len(attempts) + 1)],
        "submission attempts missing/duplicated/out of order",
    )
    config_hash = resources.get("config_sha256")
    require(
        bridge.is_hash(config_hash) and ledger.get("config_sha256") == config_hash,
        "resource/ledger frozen config binding differs",
    )
    catalog = Path(resources.get("submission_catalog_dir", ""))
    require(
        catalog.is_absolute() and catalog.is_dir() and not catalog.is_symlink(),
        "submission receipt catalog directory missing",
    )
    actual_receipts = {
        (str(path.resolve()), sha256(path)) for path in catalog.glob("*/submission.json")
    }
    declared_receipts = {
        (
            str(read_reference(a.get("submission"), "submission").resolve()),
            a["submission"]["sha256"],
        )
        for a in attempts
    }
    require(actual_receipts == declared_receipts, "omitted/unregistered submission attempt")
    accepted, monitors, requested_minutes = {}, [], 0
    for attempt in attempts:
        receipt = load_reference(attempt["submission"], "submission")
        ready = load_reference(attempt.get("monitor_ready"), "watcher ready")
        exited = load_reference(attempt.get("monitor_exit"), "watcher exit")
        commit = attempt.get("project_commit")
        require(
            bridge.is_hash(commit, 40) and bridge.is_hash(attempt.get("config_sha256")),
            "attempt freeze identity missing",
        )
        require(type(receipt.get("returncode")) is int, "submission exit status missing")
        require(
            type(receipt.get("command")) is list
            and any("sbatch" in value for value in receipt["command"]),
            "submission command missing",
        )
        domain = ready.get("clock_domain")
        require(
            type(domain) is str
            and domain.startswith("monotonic_ns:controller:")
            and receipt.get("clock_domain") == exited.get("clock_domain") == domain,
            "watcher/receipt controller clock domains differ",
        )
        clocks = [
            ready.get("observed_monotonic_ns"),
            receipt.get("started_monotonic_ns"),
            receipt.get("finished_monotonic_ns"),
            exited.get("observed_monotonic_ns"),
        ]
        require(
            all(strict_int(value, 1) for value in clocks)
            and clocks == sorted(clocks)
            and clocks[0] < clocks[1]
            and clocks[2] < clocks[3],
            "watcher was not ready before submission or did not exit after",
        )
        require(
            ready.get("campaign_id") == TASK_ID
            and ready.get("campaign_sha256") == attempt["config_sha256"],
            "watcher readiness task/config differs",
        )
        require(
            ready.get("ready") is True
            and ready.get("attempt_id") == attempt["attempt_id"]
            and ready.get("project_commit") == commit,
            "watcher readiness freeze binding differs",
        )
        require(
            exited.get("pid") == ready.get("pid")
            and exited.get("returncode") == (0 if receipt["returncode"] == 0 else 3)
            and exited.get("owned_pids_remaining") == [],
            "watcher failed to exit cleanly",
        )
        monitors.append((domain, clocks[-1]))
        if receipt["returncode"] != 0:
            require(
                attempt.get("status") == "submission_failed"
                and receipt["returncode"] > 0
                and receipt["returncode"] != 255,
                "ambiguous submission cannot be treated as a failed allocation",
            )
            require(attempt.get("job_id") is None, "failed submission has accepted job identity")
            continue
        raw_job = receipt.get("stdout", "")
        require(
            type(raw_job) is str and re.fullmatch(r"[0-9]+\n?", raw_job),
            "accepted submission has ambiguous job identity",
        )
        job = raw_job.strip()
        require(
            str(attempt.get("job_id")) == job and job not in accepted,
            "accepted job binding missing/duplicated",
        )
        accepted[job] = (attempt, receipt, ready, exited)
    require(
        set(accepted) == {str(a.get("job_id")) for a in allocations}
        and len(allocations) == len(accepted),
        "accepted job omitted/duplicated in allocation catalog",
    )
    intervals, terminal_clocks, selected = [], [], {}
    for allocation in allocations:
        job = str(allocation["job_id"])
        attempt, receipt, ready, exited = accepted[job]
        evidence = allocation.get("evidence", {})
        require(
            set(evidence) == {"submission", "terminal", "checkpoints", "exit", "binding"},
            "allocation terminal/checkpoint originals missing",
        )
        require(
            evidence["submission"] == attempt["submission"], "allocation receipt original differs"
        )
        raw = {name: load_reference(reference, name) for name, reference in evidence.items()}
        terminal = raw["terminal"]
        require(
            terminal.get("returncode") == 0 and type(terminal.get("stdout")) is str,
            "successful authoritative terminal query missing",
        )
        require(
            terminal.get("command", [])[-1:] == [f"scontrol show job {job}"],
            "terminal query does not target owned job",
        )
        fields = dict(re.findall(r"(\w+)=([^\s]+)", terminal["stdout"]))
        state, exit_code = fields.get("JobState"), fields.get("ExitCode")
        require(
            fields.get("JobId") == job
            and state
            in {
                "COMPLETED",
                "FAILED",
                "TIMEOUT",
                "CANCELLED",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
                "REVOKED",
            }
            and re.fullmatch(r"[0-9]+:[0-9]+", exit_code or ""),
            "allocation lacks authoritative terminal/exit code",
        )
        require(
            allocation.get("state") == state and allocation.get("exit_code") == exit_code,
            "declared terminal differs from raw terminal",
        )
        limit = allocation.get("limit_minutes")
        require(
            strict_int(limit, 1) and limit <= 20 and allocation.get("gpus") == 1,
            "allocation exceeds single GPU/20 minute limit",
        )
        require(fields.get("TimeLimit") == f"00:{limit:02d}:00", "raw time limit differs")
        tres = fields.get("AllocTRES", "").split(",")
        require(
            "gres/gpu=1" in tres
            and all(
                value.rsplit("=", 1)[-1] == "1" for value in tres if value.startswith("gres/gpu")
            ),
            "raw allocation is not a single GPU",
        )
        requested_minutes += limit
        start, finish = fields.get("StartTime"), fields.get("EndTime")
        require(
            start not in {None, "Unknown", "None"} and finish not in {None, "Unknown", "None"},
            "server-domain start/end missing",
        )
        start, finish = datetime.fromisoformat(start), datetime.fromisoformat(finish)
        require(start < finish, "server-domain allocation interval invalid")
        intervals.append((start, finish))
        require(
            terminal.get("clock_domain") == ready["clock_domain"]
            and strict_int(terminal.get("observed_monotonic_ns"), 1)
            and receipt["finished_monotonic_ns"]
            < terminal["observed_monotonic_ns"]
            < exited["observed_monotonic_ns"],
            "terminal/watch chronology differs",
        )
        terminal_clocks.append((terminal["clock_domain"], terminal["observed_monotonic_ns"]))
        watch_path = read_reference(attempt.get("monitor_trace"), "watcher trace")
        watch = [json.loads(line) for line in watch_path.read_text().splitlines() if line.strip()]
        require(watch and watch[-1] == terminal, "terminal absent from original watcher trace")
        require(
            all(
                row.get("clock_domain") == ready["clock_domain"]
                and row.get("command", [])[-1:] == [f"scontrol show job {job}"]
                for row in watch
            ),
            "watcher trace job/domain differs",
        )
        checks, exits, binding = raw["checkpoints"], raw["exit"], raw["binding"]
        is_selected = selected_commit is not None and (
            job in selection
            if selection is not None
            else job == selected_job
            if selected_job is not None
            else attempt["project_commit"] == selected_commit
        )
        require(
            str(checks.get("job_id"))
            == str(exits.get("job_id"))
            == str(binding.get("job_id"))
            == job,
            "allocation original job identities differ",
        )
        require(
            binding.get("project_commit")
            == exits.get("project_commit")
            == attempt["project_commit"]
            and binding.get("config_sha256", binding.get("runtime_config_sha256"))
            == attempt["config_sha256"],
            "allocation frozen source/config binding differs",
        )
        require(
            exits.get("owned_pids_remaining") == checks.get("owned_pids_remaining") == [],
            "allocation owned processes remain",
        )
        steps = exits.get("steps")
        require(
            type(steps) is list and checks.get("launcher_steps") == steps,
            "allocation step originals differ",
        )
        require(checks.get("cleanup_errors") == [], "allocation cleanup errors remain")
        require(
            all(type(step) is dict and step.get("remaining") == [] for step in steps),
            "allocation step owned processes remain",
        )
        group, declared = binding.get("execution_group"), binding.get("declared_steps")
        if group is None and declared is None and not is_selected:
            # Old calibration originals remain immutable and cannot be selected.
            group, declared = "all", EXECUTION_GROUPS["all"]
        else:
            require(
                type(group) is str
                and group in EXECUTION_GROUPS
                and declared == EXECUTION_GROUPS[group]
                and exits.get("execution_group") == group
                and exits.get("declared_steps") == declared,
                "allocation execution group/declaration differs",
            )
        require(
            [step.get("name") for step in steps] == declared[: len(steps)],
            "allocation actual steps differ from declared execution group",
        )
        for name in ("gpu_before", "gpu_during", "gpu_after"):
            samples = checks.get(name)
            samples = [samples] if type(samples) is dict else samples
            require(
                type(samples) is list and samples, "GPU before/during/after observation missing"
            )
            for sample in samples:
                bridge.audit_gpu_snapshot(sample)
                require(len(sample["stdout"].strip().splitlines()) == 1, "multiple GPUs observed")
        if is_selected:
            require(attempt["project_commit"] == selected_commit, "selected raw job commit differs")
            require(
                attempt["config_sha256"] == config_hash,
                "selected raw job config differs",
            )
            require(
                binding.get("phase") == exits.get("phase") == attempt.get("phase") == "formal",
                "selected allocation is not a formal run",
            )
            expected_group = selection[job] if selection is not None else required_group
            require(
                group == expected_group
                and attempt.get("execution_group") == group
                and attempt.get("declared_steps") == declared
                and [step.get("name") for step in steps] == EXECUTION_GROUPS[expected_group],
                "selected allocation lacks its exact execution group",
            )
            require(
                state == "COMPLETED"
                and exit_code == "0:0"
                and exits.get("returncode") == 0
                and all(
                    type(step.get("returncode")) is int
                    and step["returncode"] == 0
                    and step.get("status") == "exited"
                    and step.get("remaining") == []
                    and not step.get("timed_out", False)
                    for step in steps
                ),
                "selected frozen allocation did not complete all layers",
            )
            selected[job] = {"execution_group": group, "declared_steps": declared}
    require(requested_minutes <= 60, "M4 requested GPU minutes exceed 60")
    intervals.sort()
    require(
        all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:], strict=False)),
        "M4 allocations overlap",
    )
    if selected_commit is not None:
        require(selected, "selected tested commit absent from complete allocation")
    if selection is not None:
        require(set(selected) == set(selection), "selected metadata/gpu allocation missing")
    query = load_reference(resources.get("active_query_evidence"), "final full queue query")
    require(
        query == resources.get("active_query")
        and query.get("returncode") == 0
        and query.get("command", [])[-1:] == ['squeue -h -o "%i|%T|%j"'],
        "valid final full queue query missing",
    )
    require(
        all(
            query.get("clock_domain") == domain
            and strict_int(query.get("observed_monotonic_ns"), 1)
            and query["observed_monotonic_ns"] > clock
            for domain, clock in terminal_clocks
        ),
        "full queue query precedes terminal or changes clock domain",
    )
    require(
        any(
            query["clock_domain"] == domain and query["observed_monotonic_ns"] < clock
            for domain, clock in monitors
        ),
        "watcher exited before final full queue query",
    )
    require(
        not (set(re.findall(r"\b[0-9]+\b", query.get("stdout", ""))) & set(accepted)),
        "owned job remains in final queue",
    )
    return {
        "allocations": len(allocations),
        "attempts": len(attempts),
        "requested_minutes": requested_minutes,
        "remaining_allocations": 3 - len(allocations),
        "selected_jobs": selected,
    }


def audit_gpu_runtime(events, config, sources, frozen_inputs, inputs_sha256):
    """Actual worker tensors, source identity, and client/native input/output bridge."""
    require(
        type(frozen_inputs) is dict
        and frozen_inputs.get("task_id") == TASK_ID
        and frozen_inputs.get("model_revision")
        == frozen_inputs.get("tokenizer_revision")
        == REVISION,
        "frozen tokenizer inputs missing or revision differs",
    )
    require(bridge.is_hash(inputs_sha256), "frozen input artifact hash missing")
    inputs = frozen_inputs.get("requests", {})
    require(
        set(inputs) == {"long_a", "long_b", "short", "hot", "cold"},
        "frozen input role matrix differs",
    )
    for value in inputs.values():
        tokens = value.get("token_ids")
        require(
            type(tokens) is list
            and tokens
            and all(strict_int(token) for token in tokens)
            and len(tokens) == value.get("length")
            and hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            == value.get("token_ids_sha256"),
            "frozen token IDs/length/hash disagree",
        )
    require(
        inputs["long_a"]["length"]
        == inputs["long_b"]["length"]
        == config["inputs"]["pressure_tokens_each"]
        and inputs["long_a"]["token_ids"][:16] != inputs["long_b"]["token_ids"][:16],
        "capacity pressure input lengths/cold prefixes differ",
    )
    starts, ready = rows(events, "m4_engine_start"), rows(events, "m4_engine_ready")
    require(starts and len(starts) == len(ready), "actual engine startup/ready catalog missing")
    by_engine = {row["engine_id"]: row for row in starts}
    require(len(by_engine) == len(starts), "duplicate engine startup")
    runtime = []
    for row in ready:
        start = by_engine.get(row["engine_id"])
        require(
            start is not None
            and start["mode"] == row["mode"]
            and start.get("project_commit") == sources["project_commit"]
            and start.get("inputs_sha256") == inputs_sha256,
            "engine start/ready input or commit binding differs",
        )
        require(
            row.get("sources") == sources
            and start.get("source_sha256") == sources["source_sha256"]
            and start.get("upstream_sha256") == sources["upstream_sha256"],
            "loaded engine source identity differs",
        )
        requested = start.get("requested_config", {})
        require(
            all(requested.get(k) == v for k, v in config["gpu_profile"].items()),
            "requested engine profile differs from frozen capacity profile",
        )
        workers = row.get("worker")
        require(
            type(workers) is list and len(workers) == 1, "actual single worker snapshot missing"
        )
        snapshot = workers[0]
        require(
            snapshot.get("runner_type") == "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
            and snapshot.get("runner_observer_installed") is True,
            "actual V2 runner/observer missing",
        )
        require(
            snapshot.get("model_parameter_dtypes", {}).keys() == {"torch.bfloat16"}
            and snapshot["model_parameter_dtypes"]["torch.bfloat16"] > 3_500_000_000,
            "actual BF16 model parameter identity missing",
        )
        require(
            snapshot.get("gpu_name") == "NVIDIA GeForce RTX 3090"
            and snapshot.get("compute_capability") == [8, 6]
            and 23 * 1024**3 <= snapshot.get("gpu_total_memory_bytes", 0) <= 25 * 1024**3,
            "actual RTX3090 24GiB hardware differs",
        )
        require(
            snapshot.get("resolved_model_path")
            == snapshot.get("resolved_tokenizer_path")
            == sources["model"]["directory"],
            "actual loaded model/tokenizer differs",
        )
        expected_class = "vllm_prefix_scheduler.m4_observer." + (
            "M4NativeScheduler" if row["mode"] == "native" else "M4PrefixScheduler"
        )
        require(
            snapshot.get("configured_scheduler_type") == expected_class,
            "actual M4 scheduler class differs",
        )
        effective = snapshot.get("effective_config", {})
        for key in (
            "async_scheduling",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "enforce_eager",
            "max_num_seqs",
            "max_num_active_seqs",
            "max_num_batched_tokens",
            "max_model_len",
            "gpu_memory_utilization",
        ):
            require(
                effective.get(key) == config["gpu_profile"][key],
                f"actual engine profile differs: {key}",
            )
        require(
            effective.get("use_v2_model_runner") is True
            and effective.get("max_concurrent_batches", 0) > 1
            and effective.get("executor_backend") == "uni"
            and effective.get("tensor_parallel_size")
            == effective.get("pipeline_parallel_size")
            == 1,
            "actual async single GPU execution absent",
        )
        groups = snapshot.get("kv_cache_groups", [])
        require(
            len(groups) == 1 and len(groups[0].get("layer_names", [])) == 36,
            "actual 36-layer KV capacity missing",
        )
        spec = groups[0].get("spec", {})
        specs = list(spec.get("per_layer_specs", {}).values()) or [spec]
        require(
            all(
                s.get("type") == "vllm.v1.kv_cache_interface.FullAttentionSpec"
                and s.get("dtype") == "torch.bfloat16"
                and s.get("block_size") == 16
                for s in specs
            ),
            "actual full attention BF16 block16 spec differs",
        )
        require(
            snapshot.get("num_gpu_blocks") == config["gpu_profile"]["num_gpu_blocks_override"],
            "actual tensor capacity differs from frozen pressure profile",
        )
        tensors, storages = snapshot.get("kv_tensors", []), {}
        require(tensors, "actual KV tensors missing")
        for tensor in tensors:
            require(
                str(tensor.get("device", "")).startswith("cuda")
                and tensor.get("dtype") == "torch.bfloat16"
                and strict_int(tensor.get("view_bytes"), 1)
                and strict_int(tensor.get("storage_bytes"), 1),
                "actual GPU KV tensor invalid",
            )
            storage = tensor.get("storage_id")
            require(type(storage) is str and storage, "KV tensor storage identity missing")
            require(
                storage not in storages or storages[storage] == tensor["storage_bytes"],
                "aliased KV storage sizes disagree",
            )
            storages[storage] = tensor["storage_bytes"]
        capacity_bytes = snapshot["num_gpu_blocks"] * 16 * 144 * 1024
        require(
            snapshot.get("kv_unique_storage_count") == len(storages)
            and snapshot.get("kv_unique_storage_bytes") == sum(storages.values()) == capacity_bytes,
            "actual 36-layer tensor bytes incompatible with configured capacity",
        )
        for name, module in snapshot.get("modules", {}).items():
            path = name.replace(".", "/") + ".py"
            require(
                module.get("sha256") == sources["upstream_sha256"].get(path),
                "loaded upstream module differs from source freeze",
            )
        require(len(snapshot.get("modules", {})) >= 7, "loaded native module identity missing")
        require(
            snapshot.get("configured_scheduler_module", {}).get("sha256")
            == sources["source_sha256"]["src/vllm_prefix_scheduler/m4_observer.py"],
            "loaded M4 observer differs from source freeze",
        )
        require(snapshot.get("observation_errors") == [], "GPU observer recorded errors")
        runtime.append(
            {
                "mode": row["mode"],
                "kv_blocks": snapshot["num_gpu_blocks"],
                "kv_storage_bytes": capacity_bytes,
                "ready": location(row),
            }
        )
    # Submission order within a process assigns labels to actual generation order,
    # then every input length/hash and actual EngineCore output is checked.
    submits, bounds = defaultdict(list), defaultdict(list)
    for event in rows(events, "m4_client_submit"):
        key = event["engine_id"], event["request_id"]
        submits[key].append(event)
        value = inputs.get(event.get("input_key"))
        require(
            value is not None and event.get("token_ids") == value["token_ids"],
            "actual client input differs from frozen tokenizer output",
        )
    for event in rows(events, "m4_request_bound"):
        request = event["request"]
        bounds[(event["engine_id"], request["request_id"])].append(event)
    require(
        set(submits) == set(bounds) and submits, "client/native request admission catalogs differ"
    )
    generated = defaultdict(list)
    for event in rows(events, "m4_consume_success"):
        interval_keys = {r["request_id"]: request_key(r) for r in event["intervals"]}
        for output in event.get("outputs", []):
            require(
                output.get("request_id") in interval_keys,
                "GPU client output lacks original generation",
            )
            identity = (event["engine_id"], *interval_keys[output["request_id"]])
            generated[identity].extend(output["new_token_ids"])
    for key, client in submits.items():
        client.sort(key=lambda e: e["seq"])
        admitted = sorted(bounds[key], key=lambda e: e["seq"])
        require(len(client) == len(admitted), "client/native generation count differs")
        for submit, bound in zip(client, admitted, strict=True):
            value, request = inputs[submit["input_key"]], bound["request"]
            require(
                request["prompt_length"] == value["length"]
                and request.get("token_ids_sha256") == value["token_ids_sha256"]
                and request.get("prompt_token_ids") == value["token_ids"],
                "actual native input differs from frozen tokenizer output",
            )
            done = [
                e
                for e in rows(events, "m4_client_done")
                if (e["engine_id"], e["request_id"]) == key
                and e.get("generation_label") == submit.get("generation_label")
            ]
            require(
                len(done) == 1 and done[0].get("finished") is True,
                "actual client generation did not finish",
            )
            identity = (bound["engine_id"], *request_key(request))
            require(
                done[0].get("token_ids") == generated[identity],
                "client final tokens differ from actual generation outputs",
            )
    for begin in rows(events, "m4_case_begin"):
        if begin.get("scenario") != "full_events_probe":
            continue
        local = [e for e in events if e.get("case_id") == begin["case_id"]]
        activity = one(local, "m4_probe_activity")
        require(
            activity.get("metric_block_count", 0) > 0 and activity.get("cached_block_count", 0) > 0,
            "full probe metrics/cache activity absent",
        )
        publications = rows(local, "m4_kv_events")
        require(
            any(
                e.get("event_count", 0) > 0
                and e.get("stored_blocks", 0) > 0
                and "BlockStored" in e.get("event_types", [])
                for e in publications
            ),
            "real full KV event publication absent",
        )
        stats = rows(local, "m4_prefix_stats_activity")
        require(
            any(e.get("full_report") is True and e.get("hit_tokens", 0) > 0 for e in stats),
            "real full prefix stats activity absent",
        )
    return runtime


def audit_native_instances(events, sources, config):
    identities = rows(events, "m4_native_identity")
    expected = rows(events, "m4_case_begin")
    require(
        Counter(e.get("case_id") for e in identities)
        == Counter(e.get("case_id") for e in expected),
        "actual native instance identity catalog incomplete",
    )
    upstream = sources.get("upstream_sha256") or {
        name: entry["sha256"] for name, entry in sources["upstream_modules"].items()
    }
    for event in identities:
        expected_class = (
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
            if event["mode"] == "native"
            else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
        )
        require(
            event.get("class") == expected_class
            and "vllm.v1.core.sched.async_scheduler.AsyncScheduler" in event.get("mro", []),
            "actual native AsyncScheduler inheritance differs",
        )
        require(
            event.get("async_scheduling") is True
            and event.get("use_v2_model_runner") is True
            and event.get("apc") is True,
            "native APC/async/V2 disabled",
        )
        loaded = event.get("sources", {})
        native_hash = upstream.get("vllm/v1/core/sched/async_scheduler.py")
        require(
            loaded.get("vllm.v1.core.sched.async_scheduler.AsyncScheduler", {}).get("sha256")
            == native_hash,
            "actual native scheduler source differs",
        )
        if event["mode"] != "native":
            require(
                loaded.get(expected_class, {}).get("sha256")
                == sources["source_sha256"]["src/vllm_prefix_scheduler/scheduler.py"],
                "actual production scheduler source differs",
            )
            require(
                event.get("effective_config", {}).get("window") == config["window"]
                and event["effective_config"].get("max_wait_ms") == config["max_wait_ms"],
                "actual native window/wait config differs",
            )
        fields = event.get("instance_fields", [])
        require(
            ("_wait_guard" in fields) is (event["mode"] == "guarded"),
            "native off/prefix maintains protection table",
        )
    return len(identities)
