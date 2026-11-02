"""Independent standard-library audit of complete M2A native traces.

No vLLM imports, scheduler calls, event drains, or summary-derived acceptance.
"""

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m2a_contract import (  # noqa: E402
    expected_cases,
    require,
    require_assertions,
)

PROBE_NAMES = {"cold", "partial", "full", "exact16", "exact32", "one", "skip_read"}
WINDOW_NAMES = {
    "stable_ties",
    "all_cold",
    "window_one",
    "window_limit",
    "head_barrier",
    "middle_barrier",
    "preempt_history",
    "partial_progress",
    "placeholder",
    "in_flight",
    "stale",
    "resumable",
    "unsupported",
    "prepend_order",
    "score_exception",
    "pop_exception",
    "prepend_exception",
    "repeated_step",
    "other_queues",
}


def rows(events, kind):
    return [event for event in events if event["event"] == kind]


def one(events, kind):
    result = rows(events, kind)
    require(len(result) == 1, f"expected exactly one {kind} event, got {len(result)}")
    return result[0]


def audit_dispatch(events):
    """Recompute interval intersections and overlap sets independently."""
    positions = defaultdict(set)
    counts = defaultdict(
        lambda: {"prefill": 0, "decode": 0, "repeated_prefill": 0, "repeated_decode": 0}
    )
    by_step = {}
    for event in rows(events, "dispatch"):
        step = event["step"]
        require(step not in by_step, "more than one dispatch event per native schedule")
        intervals = event["intervals"]
        require(isinstance(intervals, list), "dispatch intervals must be a list")
        by_step[step] = {}
        for interval in intervals:
            rid = interval["request_id"]
            generation = interval["generation"]
            require(type(generation) is int and generation >= 0, "invalid request generation")
            key = (rid, generation)
            start, end, length = interval["start"], interval["end"], interval["prompt_length"]
            require(
                all(type(v) is int for v in (start, end, length)),
                "interval fields must be strict integers",
            )
            require(0 <= start < end and length > 0, "invalid dispatched interval")
            require(rid not in by_step[step], "duplicate request in one dispatch")
            require(interval["scheduled_tokens"] == end - start, "scheduled interval size mismatch")
            dispatched = set(range(start, end))
            prompt = {p for p in dispatched if p < length}
            decode = dispatched - prompt
            expected = {
                "prefill_tokens": len(prompt),
                "decode_tokens": len(decode),
                "repeated_prefill_tokens": len(prompt & positions[key]),
                "repeated_decode_tokens": len(decode & positions[key]),
            }
            require(
                all(interval.get(k) == v for k, v in expected.items()),
                "interval ledger disagrees with independent position counting",
            )
            for target, source in (
                ("prefill", "prefill_tokens"),
                ("decode", "decode_tokens"),
                ("repeated_prefill", "repeated_prefill_tokens"),
                ("repeated_decode", "repeated_decode_tokens"),
            ):
                counts[key][target] += expected[source]
            positions[key].update(dispatched)
            by_step[step][rid] = end - start
    for event in rows(events, "schedule_after"):
        require(event["step"] in by_step, "schedule output has no independently counted dispatch")
        require(
            event["scheduled"] == by_step[event["step"]],
            "scheduler output and dispatch intervals differ",
        )
    require(
        set(by_step) == {e["step"] for e in rows(events, "schedule_after")}, "orphan dispatch event"
    )
    return {f"{rid}@{generation}": value for (rid, generation), value in counts.items()}


def audit_fifo(events):
    pending = deque()
    maximum = 0
    schedules = set()
    for event in events:
        if event["event"] == "schedule_after":
            step = event["step"]
            require(step not in schedules, "duplicate schedule step")
            schedules.add(step)
            pending.append(step)
            maximum = max(maximum, len(pending))
            require(event["pending_outputs"] == len(pending), "pending schedule count mismatch")
        elif event["event"] == "output_after":
            require(bool(pending), "output without pending schedule")
            require(pending.popleft() == event["output_step"], "outputs not consumed FIFO")
            require(event["pending_outputs"] == len(pending), "pending output count mismatch")
        elif event["event"] == "drained":
            require(
                not pending and event["pending_outputs"] == 0, "drained event with pending work"
            )
    require(not pending, "trace ended with pending outputs")
    return maximum


def audit_calls(events):
    """Validate proper nesting and one matching return for each observed call."""
    stack = []
    pairs = {}
    for index, event in enumerate(events):
        if event["event"] == "call":
            call = event["call"]
            require(call not in pairs, "duplicate native call identity")
            pairs[call] = [index, event]
            stack.append(call)
        elif event["event"] == "return":
            call = event["call"]
            require(stack and stack.pop() == call, "unbalanced native call/return nesting")
            require(len(pairs[call]) == 2, "duplicate native return")
            require(
                event["method"] == pairs[call][1]["method"]
                and event.get("owner") == pairs[call][1].get("owner"),
                "native return identity mismatch",
            )
            pairs[call].extend([index, event])
    require(not stack and all(len(pair) == 4 for pair in pairs.values()), "unfinished native call")
    return pairs


def audit_released(state, *, historical_cancellation=False):
    for name in ("waiting", "kv_holding_waiting", "deferred_waiting", "running", "active_requests"):
        require(not state[name], f"live {name} remain at drain")
    free = state["free_queue"]
    blocks = state["blocks"]
    require(
        len(free) == len(set(free)) == len([b for b in blocks if not b["null"]]),
        "free queue incomplete or duplicated",
    )
    require(all(b["ref_cnt"] == 0 for b in blocks if not b["null"]), "KV references remain")
    require(
        all(not ids for group in state["request_blocks"] for ids in group.values()),
        "request block table remains populated",
    )
    for request in state["requests"].values():
        require(request["status"].startswith("FINISHED"), "historical request did not terminate")
        if historical_cancellation and request["status"] == "FINISHED_ABORTED":
            continue
        require(
            request["in_flight"]
            == request["output_placeholders"]
            == request["stale_output_tokens"]
            == 0,
            "terminal request async counters not drained",
        )


def audit_bc(events, metadata, config, counts, pairs):
    warm = one(events, "A_cached_and_evictable")
    warm_index = events.index(warm)
    state = warm["state"]
    cached = [b for b in state["blocks"] if b["hash"] is not None]
    require(len(cached) == 2, "A did not leave exactly two cache blocks")
    hashes = {b["hash"] for b in cached}
    for block in cached:
        require(
            block["ref_cnt"] == 0
            and block["id"] in state["free_queue"]
            and block["id"] in state["hash_mapping"][block["hash"]],
            "A's cached prefix is not mapped/free/evictable",
        )
    enqueue = rows(events, "bc_enqueued")
    require(
        len(enqueue) == 1 and enqueue[0]["state"]["waiting"] == ["B", "C"],
        "BC must enter in native B,C order",
    )
    actual_enqueues = [e for e in rows(events, "enqueue") if e.get("ids") in (["B"], ["C"])]
    require(
        [e["ids"][0] for e in actual_enqueues] == ["B", "C"],
        "actual enqueue order differs from B,C",
    )
    inputs = {r["id"]: r for r in config["requests"]}
    require(
        all(
            e["tokens"] == inputs[e["ids"][0]]["tokens"]
            and e["arrivals"] == {e["ids"][0]: inputs[e["ids"][0]]["arrival_time"]}
            for e in actual_enqueues
        ),
        "BC arrivals/tokens changed",
    )
    require(not rows(events, "test_only_reorder"), "fixture manual reorder is forbidden")
    lookups = [
        (index, e)
        for index, e in enumerate(events)
        if e["event"] == "return" and e["method"] == "get_computed_blocks" and e.get("owner") == "C"
    ]
    require(bool(lookups), "C lacks a native prefix relookup")
    lookup_index, lookup = lookups[0]
    pressure = metadata["scenario"] == "bc_pressure"
    prefix = metadata["mode"] == "prefix"
    hit = 0 if pressure and not prefix else 32
    require(lookup["result"]["hit_tokens"] == hit, "C native hit does not match causal condition")
    require(counts.get("A@0", {}).get("prefill") == 33, "A warmup dispatch differs")
    require(
        counts.get("B@0", {}).get("prefill") == 49
        and counts.get("C@0", {}).get("prefill") == 33 - hit,
        "B/C prompt dispatch does not establish 82/50 result",
    )
    require(
        all(
            v["decode"] == 2 and v["repeated_prefill"] == v["repeated_decode"] == 0
            for v in counts.values()
        ),
        "unexpected BC decode or repeated work",
    )
    evictions = []
    for begin, call, end, result in pairs.values():
        if call["method"] != "_maybe_evict_cached_block" or not result["result"]:
            continue
        block = next(b for b in call["state"]["blocks"] if b["id"] == call["details"]["block_id"])
        if block["hash"] not in hashes:
            continue
        require(block["ref_cnt"] == 0, "evicted referenced cache block")
        require(
            block["id"] not in result["state"]["hash_mapping"].get(block["hash"], []),
            "eviction did not remove hash mapping",
        )
        require(
            any(
                a < begin < end < z
                and c["method"] == "allocate_slots"
                and c.get("owner") == call.get("owner")
                and r["result"]["success"]
                for a, c, z, r in pairs.values()
            ),
            "eviction is not nested in successful native allocation",
        )
        evictions.append((begin, end, call.get("owner"), block["hash"]))
    if not pressure:
        require(not evictions, "ample control evicted A's prefix")
    elif not prefix:
        require(
            {
                h
                for begin, end, owner, h in evictions
                if warm_index < begin < end < lookup_index and owner == "B"
            }
            == hashes,
            "pressure native/off lacks B-allocation -> eviction -> C-miss causality",
        )
    else:
        require(
            all(begin > lookup_index for begin, _, _, _ in evictions),
            "prefix lost A before C's native lookup",
        )
        touches = [
            e for e in rows(events, "return") if e["method"] == "touch" and e.get("owner") == "C"
        ]
        require(
            any(
                all(
                    any(b["hash"] == h and b["ref_cnt"] > 0 for b in e["state"]["blocks"])
                    for h in hashes
                )
                for e in touches
            ),
            "C did not natively pin its reused prefix",
        )
    return {"B_prefill": 49, "C_prefill": 33 - hit, "BC_prefill": 82 - hit, "C_native_hit": hit}


def assert_trace(trace_path, config=None):
    require_assertions()
    if config is None:
        config = json.loads((ROOT / "configs/native-m2a.json").read_text())
    expected = expected_cases(config)
    cases = defaultdict(list)
    for line_number, line in enumerate(Path(trace_path).read_text().splitlines(), 1):
        require(bool(line.strip()), f"blank trace line {line_number}")
        event = json.loads(line)
        require(
            isinstance(event, dict) and isinstance(event.get("event"), str), "invalid event record"
        )
        require(event.get("case") in expected, "unexpected trace case")
        require(
            event["event"] not in {"skip", "skipped", "failure", "timeout", "test_only_reorder"},
            "trace contains failed/skipped/manual-reorder record",
        )
        cases[event["case"]].append(event)
    require(cases and set(cases) == set(expected), "trace must contain complete nonempty matrix")
    results = {}
    for case, metadata in expected.items():
        events = cases[case]
        begin, end = one(events, "case_begin"), one(events, "case_end")
        require(events[0] is begin and events[-1] is end, "case boundaries are not complete")
        require(
            type(begin.get("repeat")) is int
            and all(begin.get(k) == v for k, v in metadata.items()),
            "case identity differs from frozen matrix",
        )
        require(end.get("result", {}).get("status") == "passed", "case did not finish successfully")
        if metadata["scenario"] == "window":
            grouped = defaultdict(list)
            for event in events[1:-1]:
                require(
                    event.get("subcase") in WINDOW_NAMES,
                    "unexpected or missing window subcase identity",
                )
                grouped[event["subcase"]].append(event)
            require(set(grouped) == WINDOW_NAMES, "window subcase catalog incomplete")
            results[case] = {
                "subcases": {
                    name: audit_window_case(grouped[name], name, config)
                    for name in sorted(WINDOW_NAMES)
                }
            }
        else:
            pairs = audit_calls(events)
            counts = audit_dispatch(events)
            maximum = audit_fifo(events)
            result = audit_scenario(events, metadata, config, counts, pairs, maximum)
            results[case] = {"dispatch": counts, "maximum_pending": maximum, **result}
    audit_equivalence(cases, expected)
    audit_warm_states(cases, expected)
    return {"matrix_verified": True, "case_count": len(results), "cases": results}


NATIVE_CLASS = "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
CUSTOM_CLASS = "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
REJECTION_NAMES = {
    "apc_off",
    "async_off",
    "priority",
    "tensor_parallel",
    "pipeline_parallel",
    "data_parallel",
    "decode_parallel",
    "prefill_parallel",
    "aux_output",
    "hash_block",
    "multiple_groups",
    "noncausal",
    "window1_apc_off",
    "guarded",
    "cuda_only",
    "lora",
    "kv_offload",
    "cpu_offload",
    "prefetch_offload",
    "pooling",
    "expert_parallel",
    "dbo",
    "sliding_window",
    "attention_chunk",
    "speculative",
    "multimodal",
    "kv_connector",
    "ec_connector",
    "encoder_decoder",
    "diffusion",
    "nonfull_spec",
    "external_lb",
    "hybrid_lb",
}


def audit_native_identity(events, metadata, config):
    native = one(events, "native_objects")
    expected_class = NATIVE_CLASS if metadata["mode"] == "native" else CUSTOM_CLASS
    require(
        native["class"] == expected_class and native["mro"][0] == expected_class,
        "wrong actual scheduler class",
    )
    require(
        NATIVE_CLASS in native["mro"]
        and (expected_class == NATIVE_CLASS or native["mro"][1] == NATIVE_CLASS),
        "custom scheduler is not the direct AsyncScheduler subclass",
    )
    for key in ("async_scheduling", "use_v2_model_runner", "log_stats", "kv_events", "kv_metrics"):
        require(native[key] is True, f"required native setting {key} disabled")
    require(native["max_concurrent_batches"] == 2, "wrong effective async concurrency")
    require(
        native["block_size"] == 16 and native["publisher"] == "NullEventPublisher",
        "wrong cache/publisher configuration",
    )
    require(native["apc"] is (metadata["scenario"] != "apc_off"), "APC setting mismatch")
    for cls, relative in (
        (NATIVE_CLASS, "vllm/v1/core/sched/async_scheduler.py"),
        ("vllm.v1.core.sched.scheduler.Scheduler", "vllm/v1/core/sched/scheduler.py"),
    ):
        require(
            native["sources"][cls]["sha256"] == config["source_sha256"][relative],
            "native class source hash mismatch",
        )
    effective = native["effective_config"]
    require(
        effective["parallel"]
        == {
            name: 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        },
        "effective parallel semantics changed",
    )
    require(
        effective["budget"] == (16 if metadata["scenario"] == "chunk" else 64),
        "effective dispatch budget differs from frozen fixture",
    )
    require(
        effective["active"] == (1 if metadata["scenario"] == "chunk" else 2),
        "effective active limit differs from frozen fixture",
    )
    if metadata["mode"] != "native":
        require(effective["max_wait_ms"] == 500, "configuration contract drift")
    return native


def audit_windows(events, mode, window=16):
    for event in rows(events, "schedule_after"):
        counts = event["counts"]
        require(all(type(v) is int and v >= 0 for v in counts.values()), "invalid window counters")
        require(counts["super"] == 1, "schedule must call native super exactly once")
        require(
            counts["window_calls"] <= 1 and counts["sorts"] <= 1 and counts["reorders"] <= 1,
            "more than one window operation per step",
        )
        require(
            counts["checks"] <= window and counts["candidates"] <= counts["checks"],
            "unbounded or invalid candidate scan",
        )
        if mode in {"native", "off"} or window == 1:
            require(
                all(v == 0 for k, v in counts.items() if k != "super"),
                "native/off/window1 executed extension window/probe logic",
            )
        elif counts["candidates"] >= 2:
            require(
                counts["probes"] == counts["candidates"] and counts["sorts"] == 1,
                "candidate scores were omitted or repeated",
            )
        else:
            require(
                counts["probes"] == counts["sorts"] == 0, "unnecessary single-candidate probing"
            )
    for after in rows(events, "window_after"):
        before = after["before"]
        candidates, scores = after["candidates"], after["scores"]
        require(
            candidates == before[: len(candidates)],
            "candidate set crossed a barrier or skipped a queue entry",
        )
        require(len(candidates) <= window, "candidate window exceeds bound")
        if len(candidates) < 2:
            require(not scores and after["after"] == before, "single candidate caused reordering")
        else:
            require(
                [rid for rid, _ in scores] == candidates,
                "scores not computed exactly once in window order",
            )
            require(
                all(type(score) is int and score >= 0 and score % 16 == 0 for _, score in scores),
                "invalid native prefix score",
            )
            expected = [
                rid for rid, _ in sorted(scores, key=lambda item: item[1], reverse=True)
            ] + before[len(candidates) :]
            require(after["after"] == expected, "window sorting unstable or suffix changed")
        require(
            after["counts"]["checks"] <= min(window, len(before)),
            "candidate scan crossed configured bound",
        )
        previous = [e for e in rows(events, "window_before") if e["step"] == after["step"]]
        require(len(previous) == 1, "window missing unique before state")
        for name in ("running", "kv_holding_waiting", "deferred_waiting"):
            require(previous[0]["state"][name] == after["state"][name], f"window modified {name}")


def audit_window_case(events, name, config):
    audit_calls(events)
    audit_dispatch(events)
    audit_fifo(events)
    audit_native_identity(events, {"mode": "prefix", "scenario": "window"}, config)
    check = one(events, "check")
    require(
        check["name"] == name and check["branch_fixture"] is True, "window branch identity mismatch"
    )
    window = 1 if name == "window_one" else 2 if name == "window_limit" else 16
    audit_windows(events, "prefix", window)
    counts = check["counts"]
    require(
        check["window"] == window and counts["checks"] <= window,
        "window configuration or scan differs",
    )
    if name in {"score_exception", "pop_exception", "prepend_exception"}:
        require(
            check["error"] and check["before"] == check["after"] and counts["super"] == 0,
            "pre-super exception did not restore queue",
        )
    elif name == "window_one":
        enters = [e for e in rows(events, "native_schedule_enter") if e["step"] == check["step"]]
        require(
            len(enters) == 1 and enters[0]["state"]["waiting"] == check["before"],
            "window1 modified waiting before native scheduling",
        )
        require(
            all(v == 0 for k, v in counts.items() if k != "super"),
            "window1 performed extension work",
        )
    elif name == "window_limit":
        require(
            check["after"] == ["hot", "cold", "tail"]
            and counts["checks"] == counts["candidates"] == counts["probes"] == 2,
            "window limit crossed or suffix changed",
        )
    elif name == "other_queues":
        require(
            check["after"] == ["hot", "tail", "cold"],
            "other queues suppressed or altered prefix ordering",
        )
        require(
            all(
                check["state"][key] for key in ("running", "kv_holding_waiting", "deferred_waiting")
            ),
            "other-queue control used only empty queues",
        )
    elif name == "repeated_step":
        selected = [
            e
            for e in rows(events, "window_after")
            if e["before"] and e["before"][0] in {"cold", "hot"}
        ]
        require(
            len(selected) == 2 and selected[0]["step"] != selected[1]["step"],
            "repeat-step test did not score two distinct steps",
        )
        require(all(len(e["scores"]) == 3 for e in selected), "scores were cached across steps")
    else:
        require(check["before"] == check["after"], "FIFO/barrier/prepend order changed")
        if name in {
            "head_barrier",
            "preempt_history",
            "partial_progress",
            "placeholder",
            "in_flight",
            "stale",
            "resumable",
            "unsupported",
        }:
            require(
                counts["checks"] == 1 and counts["candidates"] == counts["probes"] == 0,
                "head barrier was crossed",
            )
        elif name == "middle_barrier":
            require(
                counts["checks"] == 2 and counts["candidates"] == 1 and counts["probes"] == 0,
                "middle barrier was crossed",
            )
    drained = rows(events, "drained")
    require(drained and drained[-1]["phase"] == name, "window fixture did not terminate")
    audit_released(drained[-1]["state"], historical_cancellation=True)
    return {"checks": counts, "restored_or_ordered": True}


def typed_values(obj, type_name):
    if isinstance(obj, list):
        for value in obj:
            yield from typed_values(value, type_name)
    elif isinstance(obj, dict):
        if obj.get("type") == type_name:
            yield obj
        for value in obj.values():
            yield from typed_values(value, type_name)


def audit_probe(events):
    before = {e["probe_id"]: e for e in rows(events, "probe_before")}
    after = {e["probe_id"]: e for e in rows(events, "probe_after")}
    require(
        set(before) == set(after) == PROBE_NAMES
        and len(rows(events, "probe_before"))
        == len(rows(events, "probe_after"))
        == len(PROBE_NAMES),
        "probe boundary cases missing or duplicated",
    )
    expected = {
        "cold": 0,
        "partial": 16,
        "full": 32,
        "exact16": 0,
        "exact32": 16,
        "one": 0,
        "skip_read": 0,
    }
    for name in PROBE_NAMES:
        first, last = before[name], after[name]
        require(first["state"] == last["state"], f"probe {name} mutated native state")
        require(last["score"] == expected[name], f"probe {name} violates native boundary rule")
        full = first["state"]["full"]
        required_full = {
            "requests",
            "graph",
            "identities",
            "blocks",
            "hash_mapping_ordered",
            "reverse_hash_mapping",
            "reuse_watchers",
            "single_managers",
            "queue_endpoints",
            "queue_count",
            "pool_flags",
            "stats",
            "events",
            "metrics",
            "manager_fields",
            "coordinator_fields",
            "scheduler_lifecycle",
            "random_state",
        }
        require(required_full <= set(full), "probe full snapshot omits required native state")
        require(
            len(full["blocks"]) == len(first["state"]["blocks"]) and full["single_managers"],
            "probe snapshot omits block/manager details",
        )
        require(
            full["metrics"] is not None and full["stats"] is not None and full["identities"],
            "probe snapshot omitted metrics/statistics/identities",
        )
        require(
            full["pool_flags"]["enable_kv_cache_events"] is True
            and "events" in full
            and "random_state" in full,
            "probe snapshot omitted enabled events/random state",
        )
        fields = full["requests"][name]
        require(fields["kv_cache_report_mode"] == "full", "full KV report mode was not active")
        require(events.index(first) < events.index(last), "probe after precedes before")
    controls = {e["name"]: e for e in rows(events, "check")}
    require(
        set(controls) == {"sensitivity_touch", "sensitivity_event", "sensitivity_stats"},
        "snapshot sensitivity controls missing",
    )
    for name, key in (("sensitivity_event", "events"), ("sensitivity_stats", "stats")):
        require(
            controls[name]["before"]["full"][key] != controls[name]["after"]["full"][key],
            f"snapshot cannot detect {name}",
        )
    require(
        controls["sensitivity_touch"]["before"]["free_queue"]
        != controls["sensitivity_touch"]["after"]["free_queue"],
        "snapshot cannot detect touch",
    )
    start, finish = one(events, "post_probe_begin"), one(events, "post_probe_end")
    start_index, end_index = events.index(start), events.index(finish)
    require(start_index < end_index, "post-probe native lifecycle boundaries reversed")
    require(
        start["state"]["full"]["events"] == [],
        "post-probe native phase starts with unrelated buffered events",
    )
    native_phase = events[start_index + 1 : end_index]
    require(
        all(e.get("event") != "check" for e in native_phase),
        "sensitivity controls contaminated native post-probe evidence",
    )
    published = rows(native_phase, "published")
    outputs = rows(native_phase, "output_stats")
    require(
        published and outputs, "post-probe native phase lacks its own published events/statistics"
    )
    stored = [
        obj
        for event in published
        for obj in typed_values(event["batch"], "vllm.distributed.kv_events.BlockStored")
    ]
    require(
        any(
            obj["fields"]["block_size"] == 16
            and obj["fields"]["token_ids"] == list(range(100, 132))
            for obj in stored
        ),
        "post-probe normal lookup did not publish the reused prefix",
    )
    stats = [
        obj["fields"]
        for event in outputs
        for obj in typed_values(event["outputs"], "vllm.v1.metrics.stats.PrefixCacheStats")
    ]
    require(
        sum(v["requests"] for v in stats) == 1
        and sum(v["queries"] for v in stats) == 33
        and sum(v["hits"] for v in stats) == 32,
        "post-probe native statistics were lost or duplicated",
    )
    normal_dispatch = [
        i
        for e in rows(native_phase, "dispatch")
        for i in e["intervals"]
        if i["request_id"] == "full"
    ]
    require(
        normal_dispatch and sum(i["prefill_tokens"] for i in normal_dispatch) == 1,
        "post-probe request did not use native last-token dispatch",
    )
    require(rows(native_phase, "drained"), "post-probe request did not complete native lifecycle")
    require(
        all(events.index(control) > end_index for control in controls.values()),
        "sensitivity controls ran before the isolated normal phase",
    )
    return {"probe_scores": expected, "exact_state_preserved": True}


def audit_scenario(events, metadata, config, counts, pairs, maximum):
    scenario, mode = metadata["scenario"], metadata["mode"]
    if scenario == "global_rejection":
        checks = rows(events, "check")
        names = [e["name"] for e in checks]
        require(
            set(names) == REJECTION_NAMES and len(names) == len(set(names)),
            "global rejection cases missing or duplicated",
        )
        require(
            all(
                e.get("rejected") is True
                and isinstance(e.get("error"), str)
                and "prefix_scheduler:" in e["error"]
                for e in checks
            ),
            "incompatible global configuration was accepted",
        )
        return {"rejected": names}
    native = audit_native_identity(events, metadata, config)
    audit_windows(events, mode)
    drained = rows(events, "drained")
    require(drained, "case has no real terminal drain")
    audit_released(
        drained[-1]["state"], historical_cancellation=scenario in {"cancellation", "id_reuse"}
    )
    require(rows(events, "output_stats"), "native output statistics not recorded")
    require(maximum == 2, "pending async depth two was not exercised")
    if scenario in {"bc_pressure", "bc_ample"}:
        require(
            native["capacity"]
            == config["capacities"]["pressure" if scenario == "bc_pressure" else "ample"],
            "BC cache capacity changed",
        )
        return audit_bc(events, metadata, config, counts, pairs)
    if scenario == "apc_off":
        require(
            counts["B@0"]["prefill"] == 49 and counts["C@0"]["prefill"] == 33,
            "APC-off unexpectedly reused prompt",
        )
        require(
            all(
                e["result"]["hit_tokens"] == 0
                for e in rows(events, "return")
                if e["method"] == "get_computed_blocks"
            ),
            "APC-off lookup returned a hit",
        )
    elif scenario == "probe":
        return audit_probe(events)
    elif scenario == "chunk":
        dispatched = [
            i
            for e in rows(events, "dispatch")
            for i in e["intervals"]
            if i["request_id"] == "D" and i["prefill_tokens"]
        ]
        require(
            [[i["start"], i["end"]] for i in dispatched]
            == [[0, 16], [16, 32], [32, 48], [48, 64], [64, 65]],
            "real prompt chunk dispatch differs from frozen fixture",
        )
        require(
            native["capacity"] == 10 and native["max_in_flight_tokens"] == 32,
            "chunk geometry/budget changed",
        )
        require(
            counts["D@0"]["prefill"] == 65
            and all(
                "D" not in e["ids"]
                for e in rows(events, "candidates")
                if e["step"] > dispatched[0]["step"]
            ),
            "chunk request was treated as fresh or prompt positions were lost",
        )
        d_steps = [
            e["step"]
            for e in rows(events, "dispatch")
            if any(i["request_id"] == "D" for i in e["intervals"])
        ]
        c_steps = [
            e["step"]
            for e in rows(events, "dispatch")
            if any(i["request_id"] == "C" for i in e["intervals"])
        ]
        require(c_steps and min(c_steps) > max(d_steps), "hot C crossed running chunk priority")
    elif scenario == "preemption":
        preempts = [
            pair
            for pair in pairs.values()
            if pair[1]["method"] == "_preempt_request" and pair[1]["owner"] == "R1"
        ]
        require(preempts, "no actual native preemption call")
        victim = preempts[0][3]["state"]["requests"]["R1"]
        require(
            victim["computed_optimistic"] == 0
            and victim["preemptions"] >= 1
            and victim["stale_output_tokens"] > 0,
            "preemption did not reset progress with pending stale output",
        )
        barrier = [e for e in rows(events, "schedule_before") if e["phase"] == "preempted_barrier"]
        require(
            len(barrier) == 1 and barrier[0]["state"]["waiting"][:2] == ["R1", "H"],
            "natural preemption barrier absent",
        )
        at_step = [e for e in rows(events, "schedule_after") if e["step"] == barrier[0]["step"]]
        require(
            len(at_step) == 1 and "H" not in at_step[0]["scheduled"],
            "hot request crossed natural preemption barrier",
        )
        if mode == "prefix":
            require(
                at_step[0]["counts"]["candidates"] == at_step[0]["counts"]["probes"] == 0,
                "reset progress became a fresh candidate",
            )
        require(
            any(
                i["request_id"] == "R1" and i["preemptions"] > 0
                for e in rows(events, "dispatch")
                for i in e["intervals"]
            ),
            "preempted request never natively resumed",
        )
    elif scenario == "cancellation":
        cancelled = rows(events, "cancel")
        require(
            [e["request_id"] for e in cancelled] == ["waiting_cancel", "running_cancel"],
            "waiting/running cancellation missing",
        )
        require(
            "waiting_cancel@0" not in counts and "running_cancel@0" in counts,
            "cancellation did not cover both pending and waiting paths",
        )
        require(
            events.index(cancelled[-1])
            < next(i for i, e in enumerate(events) if e["event"] == "output_after"),
            "running cancellation had no pending output",
        )
        require(
            counts["survivor@0"]["prefill"] == 33, "scheduler failed to continue after cancellation"
        )
    elif scenario == "id_reuse":
        boundaries = rows(events, "reuse_boundary")
        require(
            len(boundaries) == 2
            and all(
                e["pending_outputs"] == 0 and not e["state"]["active_requests"] for e in boundaries
            ),
            "ID reused before prior generation drained",
        )
        enqueues = rows(events, "enqueue")
        for rid in ("reuse", "cancel_reuse"):
            require(
                [e["generation"] for e in enqueues if e["ids"] == [rid]] == [0, 1],
                "ID reuse generations not observed",
            )
            require(
                all(
                    counts[f"{rid}@{g}"]["prefill"] == 33
                    and counts[f"{rid}@{g}"]["repeated_prefill"] == 0
                    for g in (0, 1)
                ),
                "ID generation accounting leaked across requests",
            )
    elif scenario == "score_eviction":
        scores = [
            (i, e)
            for i, e in enumerate(events)
            if e["event"] == "score" and e["request_id"] == "C" and e["score"] == 32
        ]
        require(scores, "no live hot score before running allocation")
        found = False
        for index, score in scores:
            later = [
                (j, e)
                for j, e in enumerate(events)
                if j > index
                and e["step"] == score["step"]
                and e["event"] == "return"
                and e["method"] == "get_computed_blocks"
                and e["owner"] == "C"
            ]
            if later and later[0][1]["result"]["hit_tokens"] < 32:
                require(
                    any(
                        index < a < z < later[0][0]
                        and c["method"] == "_maybe_evict_cached_block"
                        and c["owner"] == "R"
                        and r["result"] is True
                        for a, c, z, r in pairs.values()
                    ),
                    "score changed without traced running-stage eviction",
                )
                found = True
        require(found, "native relookup did not supersede a stale score")
    else:
        raise ValueError(f"M2A unimplemented scenario auditor: {scenario}")
    return {"scenario_verified": scenario}


CLOCK_FIELDS = {
    "vllm.v1.core.kv_cache_metrics.BlockMetricsState": {
        "birth_time_ns",
        "last_access_ns",
        "access_history",
    },
    "vllm.v1.metrics.stats.KVCacheEvictionEvent": {
        "lifetime_seconds",
        "idle_seconds",
        "reuse_gaps_seconds",
    },
    "vllm.v1.metrics.stats.SchedulerIterationDetails": {"elapsed_ms"},
    "vllm.v1.engine.EngineCoreEvent": {"timestamp"},
    "vllm.v1.engine.EngineCoreOutputs": {"timestamp"},
    "vllm.distributed.kv_events.KVEventBatch": {"ts"},
}


def clock_shape(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return [clock_shape(v) for v in obj]
    if isinstance(obj, dict) and set(obj) == {"ref"}:
        require(
            type(obj["ref"]) is int and obj["ref"] >= 0, "invalid serialized clock-vector alias"
        )
        return obj.copy()
    require(
        type(obj) in (int, float) and math.isfinite(obj) and obj >= 0, "invalid native clock value"
    )
    return 0 if obj == 0 else "<native-clock>"


def normalized(obj):
    """Narrow comparison-only normalization; raw trace remains untouched."""
    if isinstance(obj, list):
        return [normalized(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    output = {}
    typename = obj.get("type")
    typed_clock_fields = CLOCK_FIELDS.get(typename, set()) if type(typename) is str else set()
    for key, val in obj.items():
        if key == "fields" and typed_clock_fields:
            output[key] = {
                name: clock_shape(v) if name in typed_clock_fields else normalized(v)
                for name, v in val.items()
            }
        elif key == "identities":
            # Keep every label and alias relationship; only OS memory addresses vary.
            aliases = {}
            for name, identity in sorted(val.items()):
                aliases.setdefault(identity, name)
            output[key] = {name: aliases[identity] for name, identity in val.items()}
        elif key == "pool_identity":
            output[key] = "<single-native-pool>"
        else:
            output[key] = normalized(val)
    return output


def audit_equivalence(cases, expected):
    kinds = {
        "enqueue",
        "bc_enqueued",
        "A_cached_and_evictable",
        "call",
        "return",
        "dispatch",
        "advance",
        "schedule_before",
        "schedule_after",
        "native_schedule_enter",
        "output_before",
        "output_after",
        "output_stats",
        "published",
        "drained",
        "cancel",
        "reuse_boundary",
    }

    def comparable(events):
        return [
            normalized(
                {
                    k: v
                    for k, v in e.items()
                    if k not in {"case", "scenario", "mode", "repeat", "counts"}
                }
            )
            for e in events
            if e["event"] in kinds
        ]

    for case, meta in expected.items():
        if meta["mode"] != "off":
            continue
        native_case = f"{meta['repeat']}:{meta['scenario']}:native"
        require(native_case in cases, "off has no independent native control")
        native_identity = one(cases[native_case], "native_objects")
        off_identity = one(cases[case], "native_objects")
        require(
            set(off_identity["instance_fields"]) - set(native_identity["instance_fields"])
            == {"prefix_config"}
            and set(native_identity["instance_fields"]) <= set(off_identity["instance_fields"]),
            "off introduced unexpected persistent instance metadata",
        )
        left, right = comparable(cases[native_case]), comparable(cases[case])
        if left != right:
            first = next(
                (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
                min(len(left), len(right)),
            )
            raise ValueError(
                f"M2A off/native behavior differs for {case} at normalized event {first}"
            )


def audit_warm_states(cases, expected):
    """All three modes and repeats start BC from the same full cache state."""
    initial = {}
    for case, meta in expected.items():
        if meta["scenario"] not in {"bc_pressure", "bc_ample"}:
            continue
        state = normalized(one(cases[case], "A_cached_and_evictable")["state"])
        scenario = meta["scenario"]
        if scenario in initial:
            require(state == initial[scenario], f"{scenario} warm state differs across mode/repeat")
        else:
            initial[scenario] = state
    require(
        set(initial) == {"bc_pressure", "bc_ample"},
        "BC capacity controls have no complete initial states",
    )


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native-m2a.json")
    args = parser.parse_args()
    print(json.dumps(assert_trace(args.trace, json.loads(args.config.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
