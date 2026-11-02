"""Independent M3 evidence contract; no imports of inference or strategy code.

Synthetic fixtures exercise this contract only. GPU acceptance requires the same
checks over archived, source-bound events from the real engine processes.
"""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

TASK_ID = "M3-real-model-integration"
UPSTREAM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODES = ("native", "off", "prefix", "guarded")
VARIANTS = ("native", "off", "prefix", "guarded_unexpired", "guarded_expired")
BASELINE = {
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": 16384,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.85,
    "enable_prefix_caching": True,
    "async_scheduling": True,
    "enable_chunked_prefill": True,
    "enforce_eager": True,
    "seed": 42,
    "trust_remote_code": False,
    "disable_log_stats": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(f"M3: {message}")


def require_assertions():
    if not __debug__:
        raise RuntimeError(
            "M3 validation requires assertions enabled; remove -O/-OO/PYTHONOPTIMIZE"
        )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def is_hash(value, length=64):
    return (
        type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)
    )


def case_catalog(repeats=3):
    require(type(repeats) is int and repeats == 3, "exactly three independent repeats are required")
    cases = {
        f"smoke:{mode}": {"scenario": "smoke", "mode": mode, "variant": "production", "repeat": 0}
        for mode in MODES
    }
    for variant in VARIANTS:
        for repeat in range(repeats):
            cases[f"mechanism:{variant}:{repeat}"] = {
                "scenario": "mechanism",
                "mode": "guarded" if variant.startswith("guarded_") else variant,
                "variant": variant,
                "repeat": repeat,
            }
    cases["cancel:guarded"] = {
        "scenario": "cancel",
        "mode": "guarded",
        "variant": "client_abort",
        "repeat": 0,
    }
    return cases


def expected_cases(config):
    require_assertions()
    require(type(config) is dict, "config must be an object")
    require(
        config.get("schema_version") == 1 and type(config["schema_version"]) is int,
        "config schema differs",
    )
    require(config.get("task_id") == TASK_ID, "wrong task identity")
    require(config.get("upstream_commit") == UPSTREAM_COMMIT, "upstream pin differs")
    require(config.get("model_revision") == MODEL_REVISION, "model revision differs")
    require(
        config.get("tokenizer_revision", MODEL_REVISION) == MODEL_REVISION,
        "tokenizer revision differs",
    )
    require(config.get("seed") == 42 and type(config["seed"]) is int, "seed differs")
    require(config.get("window") == 16 and type(config["window"]) is int, "window differs")
    require(
        config.get("max_wait_ms") == 500 and type(config["max_wait_ms"]) is int,
        "real 500ms threshold required",
    )
    require(config.get("baseline") == BASELINE, "baseline profile differs")
    require(
        config.get("functional") == {**BASELINE, "max_num_seqs": 1},
        "functional profile must only change max_num_seqs",
    )
    require(
        config.get("sampling")
        == {"temperature": 0, "max_tokens": 8, "ignore_eos": True, "seed": 42},
        "sampling plan differs",
    )
    requests = config.get("requests")
    require(
        type(requests) is dict and set(requests) == {"A", "B", "C"}, "input role catalog differs"
    )
    require(
        all(
            type(r) is dict and type(r.get("text")) is str and r["text"] for r in requests.values()
        ),
        "nonempty input texts required",
    )
    require(
        requests["A"]["text"] != requests["B"]["text"]
        and requests["B"]["text"] != requests["C"]["text"],
        "cold B must have distinct input",
    )
    timing = config.get("timing")
    require(type(timing) is dict, "timing plan missing")
    require(timing.get("old_wait_s") == 0.75, "calibrated old wait differs")
    for name in ("enqueue_timeout_s", "case_timeout_s", "drain_timeout_s"):
        require(
            type(timing.get(name)) in (int, float) and 0 < timing[name] <= 120,
            "bounded timeout missing",
        )
    return case_catalog(config.get("repeats"))


def rows(events, kind):
    return [e for e in events if e["event"] == kind]


def one(events, kind):
    selected = rows(events, kind)
    require(len(selected) == 1, f"exactly one {kind} event required")
    return selected[0]


def request_case(request_id):
    require(type(request_id) is str and "::" in request_id, "request ID lacks frozen case binding")
    return request_id.rsplit("::", 1)[0]


def request_role(request_id):
    request_case(request_id)
    return request_id.rsplit("::", 1)[1]


def read_events(directory):
    directory = Path(directory)
    paths = (
        [directory]
        if directory.is_file()
        else [directory / "trace.jsonl"]
        if (directory / "trace.jsonl").is_file()
        else sorted(directory.rglob("*.jsonl"))
    )
    require(paths, "no raw JSONL traces")
    events = []
    for path in paths:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            require(line.strip(), f"empty raw trace line {path.name}:{lineno}")
            event = json.loads(line)
            require(type(event) is dict, "trace event must be an object")
            event = {**event, "_source": str(path), "_line": lineno}
            events.append(event)
    validate_envelopes(events)
    return events


def validate_envelopes(events):
    require(type(events) is list and events, "empty raw event set")
    processes = defaultdict(list)
    forbidden = {"synthetic_output", "fake_clock", "manual_reorder", "skipped", "observer_error"}
    for event in events:
        require(type(event) is dict, "event must be object")
        require(event.get("schema") == 1 and type(event["schema"]) is int, "event schema differs")
        for name in ("event", "clock_domain", "engine_id", "mode"):
            require(type(event.get(name)) is str and event[name], f"missing event {name}")
        require(event["mode"] in MODES, "unknown event mode")
        require(event["event"] not in forbidden, "failed/skipped/synthetic event cannot pass")
        for name in ("pid", "seq", "monotonic_ns"):
            require(type(event.get(name)) is int and event[name] >= 0, f"invalid event {name}")
        require(event["pid"] > 0, "event PID must be positive")
        require(event["clock_domain"].startswith("monotonic_ns:"), "unknown clock domain")
        processes[(event["engine_id"], event["pid"])].append(event)
    for stream in processes.values():
        stream.sort(key=lambda e: e["seq"])
        require(len({e["seq"] for e in stream}) == len(stream), "duplicate process event sequence")
        require(len({e["clock_domain"] for e in stream}) == 1, "process clock domain changed")
        require(
            all(
                a["monotonic_ns"] <= b["monotonic_ns"]
                for a, b in zip(stream, stream[1:], strict=False)
            ),
            "process monotonic clock regressed",
        )


def event_location(event):
    return {
        "file": event.get("_source"),
        "line": event.get("_line"),
        "pid": event["pid"],
        "seq": event["seq"],
    }


def case_events(events, case):
    selected = []
    for event in events:
        ids = []
        if "request_id" in event:
            ids.append(event["request_id"])
        if type(event.get("request")) is dict and "request_id" in event["request"]:
            ids.append(event["request"]["request_id"])
        for key in ("request_ids", "req_ids"):
            if type(event.get(key)) is list:
                ids.extend(event[key])
        if type(event.get("num_scheduled_tokens")) is dict:
            ids.extend(event["num_scheduled_tokens"])
        if event.get("case_id") == case or any(request_case(rid) == case for rid in ids):
            selected.append(event)
    return selected


def validate_case_identity(events, config):
    expected = expected_cases(config)
    begins, ends = rows(events, "case_begin"), rows(events, "case_end")
    require(
        Counter(e.get("case_id") for e in begins) == Counter({k: 1 for k in expected}),
        "case begins must contain entire unique frozen matrix",
    )
    require(
        Counter(e.get("case_id") for e in ends) == Counter({k: 1 for k in expected}),
        "case ends must contain entire unique frozen matrix",
    )
    for event in begins + ends:
        meta = expected[event["case_id"]]
        for field, value in meta.items():
            require(
                event.get(field) == value and type(event[field]) is type(value),
                "case metadata mislabeled",
            )
        if event["event"] == "case_end":
            require(event.get("status") == "passed", "case failed or incomplete")
    return expected


def threshold_oracle(milliseconds):
    require(type(milliseconds) in (int, float), "threshold must be a number")
    require(
        milliseconds >= 0 and math.isfinite(milliseconds), "threshold must be finite nonnegative"
    )
    return math.ceil(Fraction(milliseconds) * 1_000_000)


def audit_client(events, case, metadata, config):
    local = case_events(events, case)
    begin, end = one(local, "case_begin"), one(local, "case_end")
    require(
        begin["pid"] == end["pid"] and begin["seq"] < end["seq"],
        "client case boundaries out of order",
    )
    roles = (
        {"S0", "S1", "S2"}
        if metadata["scenario"] == "smoke"
        else {"X"}
        if metadata["scenario"] == "cancel"
        else {"A", "B", "C"}
    )
    inputs = begin.get("inputs")
    require(type(inputs) is dict and set(inputs) == roles, "case input roles incomplete")
    for role, supplied in inputs.items():
        require(type(supplied) is dict and type(supplied.get("text")) is str, "input text missing")
        tokens = supplied.get("token_ids")
        require(
            type(tokens) is list and tokens and all(type(x) is int and x >= 0 for x in tokens),
            "actual tokenizer IDs missing",
        )
        if role in config["requests"]:
            require(supplied["text"] == config["requests"][role]["text"], "frozen text changed")
    expected_sampling = dict(config["sampling"])
    if metadata["scenario"] == "cancel":
        expected_sampling["max_tokens"] = 512
    require(begin.get("sampling") == expected_sampling, "case sampling plan changed")
    submitted, completed = rows(local, "client_submit"), rows(local, "client_done")
    expected_ids = {f"{case}::{role}" for role in roles}
    require(
        Counter(e.get("request_id") for e in submitted)
        == Counter({rid: 1 for rid in expected_ids}),
        "client submissions missing or duplicated",
    )
    require(
        Counter(e.get("request_id") for e in completed)
        == Counter({rid: 1 for rid in expected_ids}),
        "client completions missing or duplicated",
    )
    outputs = rows(local, "client_output")
    result = {}
    for complete in completed:
        rid = complete["request_id"]
        token_ids = complete.get("token_ids")
        require(
            type(token_ids) is list
            and token_ids
            and all(type(x) is int and x >= 0 for x in token_ids),
            "real generated token IDs missing",
        )
        stream = [e for e in outputs if e.get("request_id") == rid]
        require(stream, "request lacks consumed async outputs")
        require(
            all(e["pid"] == complete["pid"] and e["seq"] < complete["seq"] for e in stream),
            "request output/completion ordering invalid",
        )
        submit = next(e for e in submitted if e["request_id"] == rid)
        require(
            submit["pid"] == complete["pid"]
            and begin["seq"] < submit["seq"] < complete["seq"] < end["seq"],
            "request escaped client case boundary",
        )
        if metadata["scenario"] != "cancel":
            require(
                len(token_ids) == config["sampling"]["max_tokens"]
                and complete.get("finish_reason") == "length",
                "normal request did not complete fixed real generation",
            )
            require(
                stream[-1].get("finished") is True and stream[-1].get("token_ids") == token_ids,
                "final async output differs from completion",
            )
        else:
            abort = one(local, "client_abort")
            require(
                abort.get("request_id") == rid
                and stream[0]["seq"] < abort["seq"] < complete["seq"],
                "client abort not after real output and before completion",
            )
            require(
                len(token_ids) < 512 and complete.get("finish_reason") in {"abort", "cancelled"},
                "cancel scenario ran to ordinary completion",
            )
        result[request_role(rid)] = {
            "token_ids": token_ids,
            "finish_reason": complete["finish_reason"],
        }
    return {
        "inputs": inputs,
        "results": result,
        "begin": event_location(begin),
        "end": event_location(end),
    }


def audit_decision(events, case, metadata, config):
    local = case_events(events, case)
    for kind in ("cache_reset_success", "pause_success", "resume_success"):
        one(local, kind)
    enqueues = rows(local, "enqueue")
    ids = {f"{case}::{role}" for role in ("A", "B", "C")}
    require(
        Counter(e.get("request_id") for e in enqueues) == Counter({rid: 1 for rid in ids}),
        "actual EngineCore admission missing or duplicated",
    )
    b, c = f"{case}::B", f"{case}::C"
    eb = next(e for e in enqueues if e["request_id"] == b)
    ec = next(e for e in enqueues if e["request_id"] == c)
    require(eb["pid"] == ec["pid"] and eb["seq"] < ec["seq"], "actual B/C admission order differs")
    candidates = [e for e in rows(events, "decision") if e.get("before") == [b, c]]
    mode = metadata["mode"]
    desired = [c, b] if metadata["variant"] in {"prefix", "guarded_unexpired"} else [b, c]
    if mode in {"prefix", "guarded"}:
        require(candidates, "nontrivial real contiguous B/C decision missing")
        decision = candidates[0]
        require(
            decision["pid"] == eb["pid"] and ec["seq"] < decision["seq"],
            "decision preceded actual admission",
        )
        scores = decision.get("scores")
        require(
            type(scores) is dict and all(type(v) is int and v >= 0 for v in scores.values()),
            "real score observations missing",
        )
        if mode == "prefix":
            require(set(scores) == {b, c}, "prefix scores incomplete")
            require(decision.get("now_ns") is None, "prefix read guard decision clock")
            require(scores[c] > scores[b], "prefix case lacks positive nontrivial cache advantage")
        else:
            records = decision.get("records")
            require(
                type(records) is dict and set(records) >= {b, c},
                "decision original records missing",
            )
            now = decision.get("now_ns")
            threshold = threshold_oracle(config["max_wait_ms"])
            require(
                type(now) is int and decision.get("threshold_ns") == threshold,
                "common real decision time/threshold missing",
            )
            require(
                now <= decision["monotonic_ns"]
                and decision["clock_domain"] == eb["clock_domain"] == ec["clock_domain"],
                "guard decision time outside actual monotonic event/domain",
            )
            protected = {}
            for rid, admitted in ((b, eb), (c, ec)):
                record = records[rid]
                require(
                    record == admitted.get("guard"), "guard enqueue age or sequence was refreshed"
                )
                require(
                    type(record.get("enqueued_ns")) is int and type(record.get("sequence")) is int,
                    "guard record types invalid",
                )
                require(now >= record["enqueued_ns"], "negative real waiting age")
                protected[rid] = now - record["enqueued_ns"] >= threshold
            require(
                records[b]["sequence"] < records[c]["sequence"], "original enqueue sequence changed"
            )
            if metadata["variant"] == "guarded_expired":
                require(
                    protected == {b: True, c: False},
                    "real old-B/new-C 500ms condition did not hold",
                )
                require(
                    set(scores) == {c} and scores[c] > 0,
                    "unprotected hot C score missing or protected B rescored",
                )
            else:
                require(not any(protected.values()), "unexpired guard case actually expired")
                require(
                    set(scores) == {b, c} and scores[c] > scores[b],
                    "unexpired case lacks real C cache advantage",
                )
            expected_order = sorted(
                [b, c],
                key=lambda rid: (
                    (0, records[rid]["sequence"]) if protected[rid] else (1, -scores[rid])
                ),
            )
            require(
                decision.get("after") == expected_order,
                "guard order fails independent clock/sequence/score recomputation",
            )
        require(decision.get("after") == desired, "mechanism decision order differs")
    else:
        require(not candidates, "native/off unexpectedly invokes prefix sorting")
        decision = None
    dispatches = [
        e
        for e in rows(events, "schedule_success")
        if any(rid in e.get("num_scheduled_tokens", {}) for rid in (b, c))
    ]
    require(dispatches, "B/C actual positive dispatch missing")
    first = dispatches[0]
    require(
        first["pid"] == eb["pid"] and ec["seq"] < first["seq"],
        "B/C did not actually wait together before dispatch",
    )
    require(
        first.get("num_scheduled_tokens", {}).get(desired[0], 0) > 0
        and desired[1] not in first["num_scheduled_tokens"],
        "first successful native admission violates single-active decision",
    )
    return {
        "first_order": desired,
        "decision": event_location(decision) if decision else None,
        "first_dispatch": event_location(first),
    }


def audit_batches(events):
    schedules = rows(events, "schedule_success")
    require(schedules, "no real native schedule outputs")
    required = (
        "runner_execute_start",
        "runner_execute_success",
        "update_from_output_start",
        "update_from_output_success",
        "model_output_materialized",
    )
    catalog = {}
    gpu_tokens, padded_tokens = 0, 0
    dispatch_by_request = {}
    descriptor_fields = (
        "num_scheduled_tokens",
        "total_num_scheduled_tokens",
        "new_req_ids",
        "cached_req_ids",
        "finished_req_ids",
    )
    for schedule in schedules:
        batch = schedule.get("batch_id")
        require(
            type(batch) is str and batch and batch not in catalog,
            "missing/duplicate schedule batch identity",
        )
        require(
            all(name in schedule for name in descriptor_fields),
            "native output descriptor incomplete",
        )
        descriptor = {name: schedule[name] for name in descriptor_fields}
        expected_sha = hashlib.sha256(canonical(descriptor).encode()).hexdigest()
        require(
            schedule.get("output_sha256") == expected_sha,
            "schedule output fingerprint differs from raw descriptor",
        )
        counts = schedule.get("num_scheduled_tokens")
        require(
            type(counts) is dict and all(type(v) is int and v > 0 for v in counts.values()),
            "invalid successful schedule token map",
        )
        require(
            type(schedule.get("total_num_scheduled_tokens")) is int
            and schedule["total_num_scheduled_tokens"] == sum(counts.values()),
            "scheduled total differs from request token map",
        )
        intervals = schedule.get("intervals")
        require(
            type(intervals) is list
            and Counter(i.get("request_id") for i in intervals)
            == Counter({rid: 1 for rid in counts}),
            "dispatch intervals do not bind each scheduled request",
        )
        for interval in intervals:
            require(
                all(type(interval.get(key)) is int for key in ("start", "end", "prompt_length")),
                "dispatch range must use actual integer positions",
            )
            require(
                0 <= interval["start"] < interval["end"]
                and interval["end"] - interval["start"] == counts[interval["request_id"]],
                "dispatch interval length differs from native output",
            )
        for interval in intervals:
            rid = interval["request_id"]
            start, end, prompt = interval["start"], interval["end"], interval["prompt_length"]
            record = dispatch_by_request.setdefault(
                rid, {"prompt": 0, "decode": 0, "repeated": 0, "intervals": []}
            )
            record["prompt"] += max(0, min(end, prompt) - min(start, prompt))
            record["decode"] += max(0, end - max(start, prompt))
            prior = sorted(
                (max(start, left), min(end, right))
                for left, right in record["intervals"]
                if max(start, left) < min(end, right)
            )
            covered_end = start
            for left, right in prior:
                record["repeated"] += max(0, right - max(left, covered_end))
                covered_end = max(covered_end, right)
            record["intervals"].append([start, end])
        local = [e for e in events if e.get("batch_id") == batch]
        chain = {name: one(local, name) for name in required}
        require(
            all(
                e["pid"] == schedule["pid"]
                and e["engine_id"] == schedule["engine_id"]
                and e["seq"] > schedule["seq"]
                for e in chain.values()
            ),
            "batch not linked within actual engine/worker process",
        )
        require(
            chain["runner_execute_start"]["seq"]
            < chain["runner_execute_success"]["seq"]
            < chain["model_output_materialized"]["seq"]
            < chain["update_from_output_start"]["seq"]
            < chain["update_from_output_success"]["seq"],
            "batch execution and consumption ordering broken",
        )
        execute, returned, materialized = (
            chain["runner_execute_start"],
            chain["runner_execute_success"],
            chain["model_output_materialized"],
        )
        require(
            all(
                execute.get(name) == schedule[name]
                for name in (*descriptor_fields, "output_sha256")
            ),
            "runner descriptor differs from successful native schedule",
        )
        consumed = chain["update_from_output_start"]
        for event in (materialized, consumed, chain["update_from_output_success"]):
            require(
                event.get("output_type") == "vllm.v1.outputs.ModelRunnerOutput",
                "real native model output type missing",
            )
            require(
                type(event.get("req_ids")) is list and type(event.get("sampled_token_ids")) is list,
                "actual model output missing from native consumption",
            )
            require(
                event["req_ids"] == materialized["req_ids"]
                and event["sampled_token_ids"] == materialized["sampled_token_ids"],
                "native consumed output differs from materialized model return",
            )
        require(
            set(consumed["req_ids"]) == set(counts) and len(consumed["req_ids"]) == len(counts),
            "model return request set differs from scheduled requests",
        )
        require(
            len(consumed["sampled_token_ids"]) == len(consumed["req_ids"]),
            "real sampled token/request rows differ",
        )
        sampled = dict(zip(consumed["req_ids"], consumed["sampled_token_ids"], strict=True))
        require(
            all(
                type(tokens) is list and all(type(token) is int and token >= 0 for token in tokens)
                for tokens in sampled.values()
            ),
            "real sampled output IDs invalid",
        )
        engine_outputs = chain["update_from_output_success"].get("engine_outputs")
        require(
            type(engine_outputs) is list
            and len({output.get("request_id") for output in engine_outputs}) == len(engine_outputs),
            "native output processor request rows missing/duplicated",
        )
        for output in engine_outputs:
            rid, tokens = output.get("request_id"), output.get("new_token_ids")
            require(
                rid in sampled and type(tokens) is list and tokens == sampled[rid][: len(tokens)],
                "native engine output tokens do not derive from real sampled return",
            )
        if counts:
            require(
                returned.get("result_kind") == "none" and returned.get("forward_observed") is True,
                "positive batch lacks real execute/forward observation",
            )
            require(
                returned.get("forward_tokens") == sum(counts.values())
                and type(returned.get("forward_tokens")) is int,
                "actual forward token count differs from dispatched batch",
            )
            require(
                type(returned.get("forward_tokens_padded")) is int
                and returned["forward_tokens_padded"] >= returned["forward_tokens"],
                "actual padded forward tokens missing",
            )
            require(
                set(returned.get("forward_req_ids", [])) == set(counts),
                "actual forward request set differs",
            )
            sample_start, sample = (
                one(local, "runner_sample_start"),
                one(local, "runner_sample_success"),
            )
            require(
                sample_start["pid"] == sample["pid"] == schedule["pid"]
                and sample_start["engine_id"] == sample["engine_id"] == schedule["engine_id"],
                "sampling process/batch differs",
            )
            require(
                returned["seq"] < sample_start["seq"] < sample["seq"] < materialized["seq"]
                and sample.get("result_kind") == "async",
                "positive batch async sampling/materialization chain broken",
            )
            gpu_tokens += returned["forward_tokens"]
            padded_tokens += returned["forward_tokens_padded"]
        else:
            require(
                returned.get("result_kind") == "output"
                and returned.get("forward_observed") is False,
                "empty notification falsely reported as GPU forward",
            )
            require(
                not rows(local, "runner_sample_start") and not rows(local, "runner_sample_success"),
                "empty notification unexpectedly sampled",
            )
        catalog[batch] = {"schedule": schedule, "chain": chain}
    for name in required + ("runner_sample_start", "runner_sample_success"):
        require(
            all(e.get("batch_id") in catalog for e in rows(events, name)),
            "orphan model execution/output event",
        )
    by_engine = defaultdict(list)
    for record in catalog.values():
        by_engine[record["schedule"]["engine_id"]].append(record)
    for stream in by_engine.values():
        scheduled_order = sorted(stream, key=lambda item: item["schedule"]["seq"])
        consumed_order = sorted(
            stream, key=lambda item: item["chain"]["update_from_output_start"]["seq"]
        )
        require(
            [e["schedule"]["batch_id"] for e in scheduled_order]
            == [e["schedule"]["batch_id"] for e in consumed_order],
            "native output FIFO order broken",
        )
    return {
        "batches": len(catalog),
        "positive_batches": sum(
            bool(r["schedule"]["num_scheduled_tokens"]) for r in catalog.values()
        ),
        "scheduled_tokens": sum(
            sum(r["schedule"]["num_scheduled_tokens"].values()) for r in catalog.values()
        ),
        "runner_forward_tokens": gpu_tokens,
        "runner_forward_tokens_padded": padded_tokens,
        "physical_gpu_work": "runner input token counts observed; FLOPs and speedup unmeasured",
        "dispatch_by_request": dispatch_by_request,
    }


EMPTY_LIST_RESOURCES = (
    "requests",
    "waiting",
    "running",
    "holding",
    "deferred",
    "finished_req_ids",
    "reset_preempted_req_ids",
    "inflight_prefills",
    "pending_batches",
    "guard_records",
    "observation_errors",
)
ZERO_RESOURCES = (
    "deferred_frees",
    "request_blocks",
    "non_null_ref_sum",
    "inflight_tokens",
    "stale_tokens",
    "output_placeholders",
)


def audit_drained(resources):
    require(type(resources) is dict, "native resource checkpoint missing")
    for name in EMPTY_LIST_RESOURCES:
        require(
            resources.get(name) == [] and type(resources[name]) is list,
            f"native resource not empty: {name}",
        )
    for name in ZERO_RESOURCES:
        require(
            type(resources.get(name)) is int and resources[name] == 0,
            f"native resource not zero: {name}",
        )
    total = resources.get("total_non_null_blocks")
    require(type(total) is int and total > 0, "actual KV capacity missing")
    require(
        all(
            type(resources.get(name)) is int and resources[name] == total
            for name in ("free_count", "free_unique_count", "free_traversed_count")
        ),
        "free queue incomplete or duplicated",
    )
    require(resources.get("guard_fault_code") is None, "guard faulted during normal drain")
    require(
        type(resources.get("cached_block_count")) is int and resources["cached_block_count"] >= 0,
        "cache hash state unobserved",
    )


def audit_guard(events, engine_id):
    events = sorted(
        [e for e in events if e["engine_id"] == engine_id], key=lambda e: (e["pid"], e["seq"])
    )
    current, seen_objects, first_dispatch = {}, set(), set()
    next_sequence, last_enqueue_ns = 0, None

    def record_map(value):
        require(type(value) is list, "guard storage observation missing")
        mapping = {}
        for record in value:
            require(type(record) is dict, "guard storage record invalid")
            rid = record.get("request_id")
            require(type(rid) is str and rid not in mapping, "guard storage duplicated")
            mapping[rid] = record
        return mapping

    for event in events:
        kind = event["event"]
        if kind == "enqueue":
            rid, record = event["request_id"], event.get("guard")
            require(
                type(record) is dict
                and record.get("request_id") == rid
                and record.get("object_id") == event.get("object_id"),
                "guard lacks actual-object enqueue binding",
            )
            require(
                record.get("sequence") == next_sequence and type(record["sequence"]) is int,
                "guard original sequence differs",
            )
            require(
                type(record.get("enqueued_ns")) is int
                and record["enqueued_ns"] <= event["monotonic_ns"],
                "guard enqueue time not real monotonic domain",
            )
            require(
                record["object_id"] not in seen_objects and rid not in current,
                "guard object/request duplicate",
            )
            require(
                last_enqueue_ns is None or record["enqueued_ns"] >= last_enqueue_ns,
                "real guard enqueue clock regressed",
            )
            last_enqueue_ns = record["enqueued_ns"]
            next_sequence += 1
            seen_objects.add(record["object_id"])
            current[rid] = record
        elif kind == "schedule_success":
            require(
                record_map(event.get("guard_before")) == current,
                "guard changed before successful dispatch",
            )
            for rid, tokens in event["num_scheduled_tokens"].items():
                if tokens > 0 and rid not in first_dispatch:
                    require(rid in current, "first positive dispatch lacked original wait record")
                    del current[rid]
                    first_dispatch.add(rid)
            require(
                record_map(event.get("guard_after")) == current,
                "first successful dispatch did not clear only its object metadata",
            )
        elif kind == "request_terminal":
            require(
                record_map(event.get("guard_before")) == current,
                "terminal guard before-state differs",
            )
            current.pop(event["request_id"], None)
            require(
                record_map(event.get("guard_after")) == current,
                "terminal object metadata retained or unrelated metadata changed",
            )
        elif kind == "scheduler_checkpoint":
            require(
                record_map(event["resources"]["guard_records"]) == current,
                "guard checkpoint differs from lifecycle replay",
            )
    require(not current, "guard records remain before shutdown")
    require(first_dispatch, "guarded first positive dispatch lifecycle unexercised")
    return {"registered": next_sequence, "first_dispatch_cleared": len(first_dispatch)}


def audit_native_requests(events, case, metadata, client):
    local = case_events(events, case)
    admitted, terminals = rows(local, "enqueue"), rows(local, "request_terminal")
    expected_ids = {f"{case}::{role}" for role in client["inputs"]}
    require(
        Counter(e.get("request_id") for e in admitted) == Counter({rid: 1 for rid in expected_ids}),
        "actual enqueue catalog incomplete",
    )
    require(
        Counter(e.get("request_id") for e in terminals)
        == Counter({rid: 1 for rid in expected_ids}),
        "actual native terminal catalog incomplete",
    )
    bindings = {e["request_id"]: e for e in admitted}
    for event in rows(events, "schedule_success"):
        for interval in event.get("intervals", []):
            rid = interval.get("request_id")
            if rid in expected_ids:
                require(
                    interval.get("object_id") == bindings[rid].get("object_id")
                    and interval.get("prompt_length") == len(bindings[rid]["prompt_token_ids"]),
                    "dispatch range does not bind actual request object/prompt geometry",
                )
    generated = defaultdict(list)
    for event in sorted(rows(events, "update_from_output_success"), key=lambda e: e["seq"]):
        for output in event.get("engine_outputs", []):
            if output.get("request_id") in expected_ids:
                generated[output["request_id"]].extend(output["new_token_ids"])
    for enqueue in admitted:
        rid = enqueue["request_id"]
        require(
            enqueue.get("bound") is True and enqueue.get("status") == "WAITING",
            "native add did not bind and enqueue ordinary request",
        )
        require(enqueue.get("num_computed_tokens") == 0, "request was not fresh at admission")
        require(
            enqueue.get("prompt_token_ids") == client["inputs"][request_role(rid)]["token_ids"],
            "EngineCore prompt IDs differ from actual tokenizer input",
        )
        terminal = next(e for e in terminals if e["request_id"] == rid)
        require(
            terminal.get("object_id") == enqueue.get("object_id")
            and terminal["pid"] == enqueue["pid"]
            and terminal["seq"] > enqueue["seq"],
            "native lifecycle object binding broken",
        )
        tokens = client["results"][request_role(rid)]["token_ids"]
        if metadata["scenario"] == "cancel":
            require(
                terminal.get("status") == "FINISHED_ABORTED",
                "client cancellation lacks actual native aborted terminal",
            )
            require(
                generated[rid][: len(tokens)] == tokens,
                "cancelled client tokens not prefix of real native outputs",
            )
        else:
            require(
                terminal.get("status") == "FINISHED_LENGTH_CAPPED",
                "normal native terminal reason differs",
            )
            require(
                generated[rid] == tokens,
                "client completion not reproduced by real native output consumption",
            )


def audit_checkpoints(events, case, scenario):
    names = [f"{case}:drained"]
    if scenario == "mechanism":
        names = [f"{case}:initial", f"{case}:warm_drained", *names]
    for name in names:
        selected = [e for e in rows(events, "scheduler_checkpoint") if e.get("label") == name]
        require(selected, "case boundary checkpoint missing")
        require(len({e["pid"] for e in selected}) == 1, "case checkpoint process changed")
        checkpoint = max(selected, key=lambda e: e["seq"])
        audit_drained(checkpoint.get("resources"))
        if name.endswith(":initial"):
            require(
                checkpoint["resources"]["cached_block_count"] == 0,
                "independent repeat initial cache was not reset",
            )
        if name.endswith(":warm_drained"):
            require(
                checkpoint["resources"]["cached_block_count"] > 0,
                "real A warmup left no reusable cache state",
            )
    return names


def audit_case_timeline(events, case, scenario):
    local = case_events(events, case)
    begin, end = one(local, "case_begin"), one(local, "case_end")
    client_pid = begin["pid"]

    def client_event(kind, rid=None, label=None):
        candidates = [
            e
            for e in local
            if e["event"] == kind
            and (rid is None or e.get("request_id") == rid)
            and (label is None or e.get("label") == label)
        ]
        event = one(candidates, kind)
        require(event["pid"] == client_pid, "client lifecycle migrated process")
        return event

    reset = client_event("cache_reset_success")
    initial = client_event("client_drain", label="initial")
    final = client_event("client_drain", label="drained")
    chain = [begin, reset, initial]
    if scenario == "mechanism":
        a, b, c = (f"{case}::{role}" for role in ("A", "B", "C"))
        chain.extend(
            [
                client_event("client_submit", a),
                client_event("client_done", a),
                client_event("client_drain", label="warm_drained"),
                client_event("pause_success"),
                client_event("client_submit", b),
                client_event("client_enqueue_confirmed", b),
                client_event("client_submit", c),
                client_event("client_enqueue_confirmed", c),
                client_event("resume_success"),
            ]
        )
        for rid in (b, c):
            ack = client_event("client_enqueue_confirmed", rid)
            enqueue = one(
                [e for e in local if e["event"] == "enqueue" and e.get("request_id") == rid],
                "enqueue",
            )
            require(
                ack.get("engine_pid") == enqueue["pid"] and ack.get("engine_seq") == enqueue["seq"],
                "client acknowledgement does not reference actual successful EngineCore admission",
            )
        ec = one(
            [e for e in local if e["event"] == "enqueue" and e.get("request_id") == c], "enqueue"
        )
        require(
            ec.get("waiting") == [b, c],
            "actual EngineCore B/C waiting window not simultaneously present",
        )
        resume = chain[-1]
        require(
            all(
                resume["seq"] < client_event("client_done", rid)["seq"] < final["seq"]
                for rid in (b, c)
            ),
            "B/C completion occurred outside resumed interval",
        )
    chain.extend([final, end])
    require(
        all(a["seq"] < b["seq"] for a, b in zip(chain, chain[1:], strict=False)),
        "controlled case lifecycle order invalid",
    )
    for drain in rows(local, "client_drain"):
        require(drain.get("unfinished") == 0, "client output processor retained active request")
        snapshots = drain.get("snapshots")
        require(
            type(snapshots) is list and len(snapshots) == 1,
            "successful client drain lacks actual worker snapshot",
        )
        snapshot = snapshots[0]
        require(
            snapshot.get("runner_request_ids") == []
            and snapshot.get("runner_execute_state_pending") is False
            and snapshot.get("observation_errors") == [],
            "worker/runner async resources not drained",
        )
        if scenario != "smoke":
            audit_drained(snapshot.get("scheduler"))
            label = f"{case}:{drain['label']}"
            actual = [
                e
                for e in rows(events, "scheduler_checkpoint")
                if e.get("label") == label
                and e["pid"] == snapshot.get("pid")
                and e.get("resources") == snapshot["scheduler"]
            ]
            require(actual, "client drain snapshot not bound to actual scheduler checkpoint")
    if scenario == "smoke":
        return
    enqueues, terminals = rows(local, "enqueue"), rows(local, "request_terminal")
    scheduler_pid = enqueues[0]["pid"]
    require(
        all(e["pid"] == scheduler_pid for e in enqueues + terminals),
        "native request lifecycle process changed",
    )

    def checkpoint(label):
        selected = [
            e for e in rows(events, "scheduler_checkpoint") if e.get("label") == f"{case}:{label}"
        ]
        require(selected, "native case checkpoint missing")
        result = max(selected, key=lambda e: e["seq"])
        require(result["pid"] == scheduler_pid, "checkpoint not from actual case scheduler")
        return result

    require(
        checkpoint("initial")["seq"] < min(e["seq"] for e in enqueues),
        "initial cache checkpoint relabeled from later lifecycle",
    )
    require(
        checkpoint("drained")["seq"] > max(e["seq"] for e in terminals),
        "native drain checkpoint preceded actual request terminals",
    )
    request_ids = {e["request_id"] for e in enqueues}
    outputs = [
        e
        for e in rows(events, "update_from_output_success")
        if set(e.get("req_ids", [])) & request_ids
    ]
    require(
        outputs and checkpoint("drained")["seq"] > max(e["seq"] for e in outputs),
        "native drain checkpoint preceded actual output consumption",
    )
    if scenario == "mechanism":
        a_terminal = next(e for e in terminals if e["request_id"].endswith("::A"))
        b_enqueue = next(e for e in enqueues if e["request_id"].endswith("::B"))
        require(
            a_terminal["seq"] < checkpoint("warm_drained")["seq"] < b_enqueue["seq"],
            "A warmup drain did not separate B/C initial state",
        )


def audit_scheduler_identity(events, engine_id, mode):
    local = [e for e in events if e["engine_id"] == engine_id]
    identity = one(local, "scheduler_init")
    effective = identity.get("effective_config", {})
    expected_type = (
        "vllm_prefix_scheduler.m3_observer.ObserverNativeScheduler"
        if mode == "native"
        else "vllm_prefix_scheduler.m3_observer.ObserverPrefixScheduler"
    )
    require(
        identity.get("actual_type") == expected_type, "actual EngineCore scheduler class differs"
    )
    mro = identity.get("mro")
    require(
        type(mro) is list and "vllm.v1.core.sched.async_scheduler.AsyncScheduler" in mro,
        "actual scheduler lost native AsyncScheduler ancestry",
    )
    if mode != "native":
        require(
            "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler" in mro,
            "diagnostic class lacks production scheduler ancestor",
        )
        require(
            effective.get("window") == 16 and effective.get("max_wait_ms") == 500,
            "effective policy parameters differ",
        )
    require(effective.get("mode") == mode, "actual scheduler mode differs")
    for key in (
        "async_scheduling",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "use_v2_model_runner",
    ):
        require(effective.get(key) is True, f"required runtime feature disabled: {key}")
    require(effective.get("dtype") in {"torch.bfloat16", "bfloat16"}, "actual model is not BF16")
    for key, value in (
        ("max_num_seqs", 1),
        ("max_num_active_reqs", 1),
        ("max_model_len", 16384),
        ("max_num_batched_tokens", 2048),
    ):
        require(
            type(effective.get(key)) is int and effective[key] == value,
            f"functional runtime parameter differs: {key}",
        )
    require(
        effective.get("block_size") == effective.get("hash_block_size")
        and type(effective.get("num_gpu_blocks")) is int
        and effective["num_gpu_blocks"] > 1,
        "real KV geometry missing",
    )
    require(
        effective.get("kv_spec_types") == ["vllm.v1.kv_cache_interface.FullAttentionSpec"],
        "actual KV group is not full attention",
    )
    modules = identity.get("modules")
    require(
        type(modules) is dict
        and set(modules)
        >= {"scheduler", "config", "probe", "wait_guard", "observer", "native_async_scheduler"},
        "production/native module sources missing",
    )
    require(
        all(
            type(m) is dict and is_hash(m.get("sha256")) and type(m.get("path")) is str
            for m in modules.values()
        ),
        "module source fingerprints invalid",
    )
    return {
        "actual_type": identity["actual_type"],
        "location": event_location(identity),
        "effective_config": effective,
        "module_sha256": {k: v["sha256"] for k, v in modules.items()},
    }


def audit_sources(start, ready, config):
    sources = ready.get("sources")
    require(type(sources) is dict, "real environment/model source evidence missing")
    require(
        is_hash(sources.get("project_commit"), 40) and sources.get("project_dirty") is False,
        "source checkout was not clean and frozen",
    )
    require(
        sources["project_commit"] == start.get("project_commit"),
        "engine/project commit binding differs",
    )
    require(
        sources.get("python_version") == "3.12.16"
        and sources.get("packages", {}).get("vllm") == "0.32.0",
        "fixed Python/vLLM dependency differs",
    )
    require(sources.get("upstream_commit") == UPSTREAM_COMMIT, "upstream source commit differs")
    for name in ("source_sha256", "upstream_sha256"):
        values = sources.get(name)
        require(
            type(values) is dict
            and values
            and all(type(k) is str and is_hash(v) for k, v in values.items()),
            "actual module source hashes missing",
        )
        require(values == start.get(name), "startup/ready source fingerprint differs")
    required_sources = {
        "src/vllm_prefix_scheduler/" + name + ".py"
        for name in (
            "scheduler",
            "config",
            "probe",
            "wait_guard",
            "m3_contract",
            "m3_events",
            "m3_observer",
            "m3_worker",
        )
    }
    require(
        required_sources <= set(sources["source_sha256"]),
        "critical production/observer/auditor source absent",
    )
    require(
        set(config["extra_upstream_sha256"]) <= set(sources["upstream_sha256"]),
        "critical upstream engine/runner source absent",
    )
    require(
        all(sources["upstream_sha256"][k] == v for k, v in config["extra_upstream_sha256"].items()),
        "fixed engine/runner upstream content differs",
    )
    model = sources.get("model", {})
    require(
        model.get("revision") == model.get("tokenizer_revision") == MODEL_REVISION
        and is_hash(model.get("tree_sha256")),
        "real model/tokenizer snapshot identity missing",
    )
    files = model.get("files")
    require(type(files) is list and files, "real model/tokenizer file manifest missing")
    names = [f.get("name") for f in files]
    require(len(set(names)) == len(names), "model source file duplicated")
    require(
        {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors.index.json",
        }
        <= set(names),
        "model/tokenizer manifest incomplete",
    )
    weights = [
        f for f in files if type(f.get("name")) is str and f["name"].endswith(".safetensors")
    ]
    require(
        weights and sum(f.get("size", 0) for f in weights) > 7_000_000_000,
        "Qwen3-4B BF16 real weight files missing",
    )
    require(
        set(names) == set(config["model_files"]),
        "model source manifest differs from frozen official file catalog",
    )
    for entry in files:
        require(
            type(entry.get("size")) is int and entry["size"] > 0 and is_hash(entry.get("sha256")),
            "model file content identity missing",
        )
        require(
            entry.get("revision") == MODEL_REVISION
            and (
                is_hash(entry.get("verified_content_id"), 64)
                or is_hash(entry.get("verified_content_id"), 40)
            ),
            "model content not bound to snapshot tree",
        )
        official = config["model_files"][entry["name"]]
        require(
            entry["size"] == official["size"]
            and entry["verified_content_id"] == official["content_id"],
            "model source differs from official fixed snapshot identity",
        )
        if official["kind"] == "lfs_sha256":
            require(
                entry["sha256"] == entry["verified_content_id"],
                "LFS content differs from upstream identity",
            )
    requested = start.get("requested_config", {})
    require(
        requested.get("model") == requested.get("tokenizer") == model.get("directory"),
        "requested model/tokenizer path differs from verified snapshot",
    )
    return sources


def audit_worker(snapshot, mode, phase, sources):
    require(
        snapshot.get("resolved_model_path")
        == snapshot.get("resolved_tokenizer_path")
        == sources["model"]["directory"],
        "actual loaded model/tokenizer path differs from verified snapshot",
    )
    require(
        snapshot.get("runner_type") == "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "active real V2 runner missing",
    )
    require(
        snapshot.get("model_parameter_dtypes", {}).keys() == {"torch.bfloat16"},
        "actual model parameters not exclusively BF16",
    )
    require(
        snapshot["model_parameter_dtypes"]["torch.bfloat16"] > 3_500_000_000,
        "real 4B model parameter observation missing",
    )
    require(
        snapshot.get("gpu_name") == "NVIDIA GeForce RTX 3090"
        and snapshot.get("compute_capability") == [8, 6],
        "fixed RTX3090 hardware differs",
    )
    require(
        23 * 1024**3 <= snapshot.get("gpu_total_memory_bytes", 0) <= 25 * 1024**3,
        "24GiB device capacity missing",
    )
    require(
        snapshot.get("runner_observer_installed") is (phase == "diagnostic"),
        "production smoke/observer separation violated",
    )
    expected_cls = (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        if mode == "native"
        else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
    )
    if phase == "diagnostic":
        expected_cls = (
            "vllm_prefix_scheduler.m3_observer.ObserverNativeScheduler"
            if mode == "native"
            else "vllm_prefix_scheduler.m3_observer.ObserverPrefixScheduler"
        )
    require(
        snapshot.get("configured_scheduler_type") == expected_cls,
        "worker configured scheduler identity differs",
    )
    effective = snapshot.get("effective_config", {})
    for key in (
        "async_scheduling",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "use_v2_model_runner",
        "enforce_eager",
    ):
        require(effective.get(key) is True, "worker runtime feature disabled")
    limit = 4 if phase == "smoke" else 1
    require(
        effective.get("max_num_seqs") == limit
        and effective.get("max_num_active_seqs") in {None, limit},
        "actual sequence capacity differs",
    )
    require(
        effective.get("max_concurrent_batches", 0) > 1
        and effective.get("executor_backend") == "uni",
        "native asynchronous batch queue path absent",
    )
    require(
        effective.get("max_model_len") == 16384
        and effective.get("max_num_batched_tokens") == 2048
        and effective.get("gpu_memory_utilization") == 0.85,
        "actual baseline configuration differs",
    )
    require(
        effective.get("tensor_parallel_size") == effective.get("pipeline_parallel_size") == 1,
        "single GPU parallel configuration differs",
    )
    tensors = snapshot.get("kv_tensors")
    require(
        type(tensors) is list
        and tensors
        and type(snapshot.get("num_gpu_blocks")) is int
        and snapshot["num_gpu_blocks"] > 1,
        "real KV tensors or capacity missing",
    )
    storages = {}
    for tensor in tensors:
        require(
            type(tensor.get("shape")) is list
            and tensor["shape"]
            and tensor.get("numel", 0) > 0
            and tensor.get("view_bytes", 0) > 0,
            "invalid KV tensor observation",
        )
        require(str(tensor.get("device", "")).startswith("cuda"), "KV tensor not on GPU")
        storage = tensor.get("storage_id")
        require(
            type(storage) is str and tensor.get("storage_bytes", 0) > 0,
            "KV tensor physical storage observation missing",
        )
        require(
            storage not in storages or storages[storage] == tensor["storage_bytes"],
            "aliased KV storage sizes disagree",
        )
        storages[storage] = tensor["storage_bytes"]
    require(
        snapshot.get("kv_unique_storage_count") == len(storages)
        and snapshot.get("kv_unique_storage_bytes") == sum(storages.values()),
        "KV physical storage accounting differs",
    )
    groups = snapshot.get("kv_cache_groups")
    require(type(groups) is list and len(groups) == 1, "single full-attention KV group missing")
    spec = groups[0].get("spec", {})
    if spec.get("type") == "vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs":
        layers = spec.get("per_layer_specs", {})
        require(
            set(layers) == set(groups[0].get("layer_names", [])) and layers,
            "uniform KV wrapper layer set differs",
        )
        require(
            all(
                layer.get("type") == "vllm.v1.kv_cache_interface.FullAttentionSpec"
                and layer.get("dtype") == "torch.bfloat16"
                for layer in layers.values()
            ),
            "uniform KV wrapper contains non-BF16/non-full-attention layer",
        )
    else:
        require(
            spec.get("type") == "vllm.v1.kv_cache_interface.FullAttentionSpec"
            and spec.get("dtype") == "torch.bfloat16",
            "real KV spec/dtype differs",
        )
    require(
        groups[0].get("layer_names") and len(groups[0]["layer_names"]) > 1,
        "real multi-layer KV geometry missing",
    )
    modules = snapshot.get("modules", {})
    for name in (
        "vllm.v1.engine.core",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.worker.gpu_worker",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.async_utils",
        "vllm.v1.executor.uniproc_executor",
    ):
        path = name.replace(".", "/") + ".py"
        require(
            modules.get(name, {}).get("sha256") == sources["upstream_sha256"].get(path)
            and path in sources["upstream_sha256"],
            "loaded native module differs from verified source",
        )
    for name, filename in (("events", "m3_events"), ("worker", "m3_worker")):
        require(
            snapshot.get("observer_modules", {}).get(name, {}).get("sha256")
            == sources["source_sha256"][f"src/vllm_prefix_scheduler/{filename}.py"],
            "loaded observer differs from frozen checkout",
        )
    if mode != "native":
        for name in ("scheduler", "config", "probe", "wait_guard"):
            require(
                snapshot.get("production_modules", {}).get(name, {}).get("sha256")
                == sources["source_sha256"][f"src/vllm_prefix_scheduler/{name}.py"],
                "loaded production smoke/diagnostic module differs from freeze",
            )
    require(snapshot.get("observation_errors") == [], "observer recorded a suppressed failure")
    return {
        "kv_blocks": snapshot["num_gpu_blocks"],
        "kv_storage_bytes": snapshot["kv_unique_storage_bytes"],
        "model_parameter_count": snapshot["model_parameter_dtypes"]["torch.bfloat16"],
    }


def audit_gpu_snapshot(snapshot):
    require(
        type(snapshot) is dict
        and snapshot.get("returncode") == 0
        and type(snapshot.get("returncode")) is int,
        "GPU query failed or status missing",
    )
    require(
        type(snapshot.get("monotonic_ns")) is int and type(snapshot.get("stdout")) is str,
        "GPU query raw output/clock missing",
    )
    lines = snapshot["stdout"].strip().splitlines()
    require(lines, "GPU query returned empty data")
    for line in lines:
        columns = [item.strip() for item in line.split(",")]
        require(
            len(columns) == 5 and columns[0] == "NVIDIA GeForce RTX 3090",
            "GPU query device/schema differs",
        )
        require(
            all(value.isdigit() for value in columns[1:]),
            "GPU memory/utilization query is not numeric",
        )
        total, used, free, utilization = map(int, columns[1:])
        require(
            23 * 1024 <= total <= 25 * 1024
            and 0 <= used <= total
            and 0 <= free <= total
            and 0 <= utilization <= 100,
            "GPU MiB/utilization observation invalid",
        )


def audit_resources(resources):
    require(type(resources) is dict, "allocation release evidence missing")
    allocations = resources.get("allocations")
    require(
        type(allocations) is list and 1 <= len(allocations) <= 4,
        "allocation budget missing/exceeded",
    )
    ids, intervals = set(), []
    for allocation in allocations:
        job = allocation.get("job_id")
        require(
            type(job) in (str, int) and str(job).isdigit() and str(job) not in ids,
            "allocation identity missing/duplicated",
        )
        ids.add(str(job))
        require(
            type(allocation.get("limit_minutes")) is int
            and 0 < allocation["limit_minutes"] <= 20
            and type(allocation.get("gpus")) is int
            and allocation["gpus"] == 1,
            "single-GPU/time allocation budget exceeded",
        )
        require(
            allocation.get("state") in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"},
            "Slurm allocation lacks terminal state",
        )
        require(
            type(allocation.get("exit_code")) is str
            and re.fullmatch(r"[0-9]+:[0-9]+", allocation["exit_code"]),
            "Slurm terminal exit code missing",
        )
        if allocation["state"] == "COMPLETED":
            require(allocation["exit_code"] == "0:0", "successful Slurm state has nonzero exit")
        for name in ("gpu_before", "gpu_during", "gpu_after"):
            observations = allocation.get(name)
            observations = [observations] if type(observations) is dict else observations
            require(type(observations) is list and observations, "GPU resource observation missing")
            for observation in observations:
                audit_gpu_snapshot(observation)
        require(allocation.get("owned_pids_remaining") == [], "task process resources remain")
        raw = allocation.get("slurm_final")
        require(
            type(raw) is dict and raw.get("returncode") == 0 and type(raw.get("stdout")) is str,
            "raw successful Slurm final query missing",
        )
        fields = dict(re.findall(r"(\w+)=([^\s]+)", raw["stdout"]))
        require(
            fields.get("JobId") == str(job)
            and fields.get("JobState") == allocation["state"]
            and fields.get("ExitCode") == allocation["exit_code"],
            "Slurm declaration differs from raw final query",
        )
        require(
            fields.get("TimeLimit") in {"00:20:00", "00:15:00", "00:10:00", "00:05:00"},
            "raw Slurm allocation limit outside frozen bounded profiles",
        )
        start, end = fields.get("StartTime"), fields.get("EndTime")
        require(
            type(start) is str
            and type(end) is str
            and start not in {"Unknown", "None"}
            and end not in {"Unknown", "None"},
            "raw Slurm allocation interval missing",
        )
        began, ended = datetime.fromisoformat(start), datetime.fromisoformat(end)
        require(began < ended, "Slurm interval invalid")
        intervals.append((began, ended))
    intervals.sort()
    require(
        all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:], strict=False)),
        "GPU allocations overlapped",
    )
    require(
        resources.get("active_query_success") is True and resources.get("active_owned_jobs") == [],
        "final valid owned-job query missing or active allocation remains",
    )
    active = resources.get("active_query")
    require(
        type(active) is dict
        and active.get("returncode") == 0
        and type(active.get("stdout")) is str,
        "raw final active-job query missing",
    )
    require(
        not (set(re.findall(r"\b[0-9]+\b", active["stdout"])) & ids),
        "raw active-job query still contains owned allocation",
    )
    artifacts = resources.get("raw_artifacts")
    require(
        type(artifacts) is list
        and artifacts
        and all(
            type(a) is dict and type(a.get("path")) is str and is_hash(a.get("sha256"))
            for a in artifacts
        ),
        "resource raw artifact provenance missing",
    )
    for artifact in artifacts:
        require(
            Path(artifact["path"]).is_file() and sha256(artifact["path"]) == artifact["sha256"],
            "resource raw artifact missing or source hash differs",
        )
    return {"allocations": len(allocations), "released_job_ids": sorted(ids)}


def audit_trace(
    events,
    config,
    *,
    resources=None,
    engine_id=None,
    config_sha256=None,
    frozen_inputs=None,
    inputs_sha256=None,
):
    require_assertions()
    validate_envelopes(events)
    require(is_hash(config_sha256), "actual frozen config file hash required")
    expected = expected_cases(config)
    require(
        is_hash(inputs_sha256) and type(frozen_inputs) is dict,
        "frozen real tokenizer input artifact/hash missing",
    )
    require(
        frozen_inputs.get("model_revision")
        == frozen_inputs.get("tokenizer_revision")
        == MODEL_REVISION,
        "frozen tokenizer input source differs",
    )
    texts = {r["text"] for r in config["requests"].values()} | {
        f"Give one short fact about {subject}." for subject in ("GPUs", "oceans", "planets")
    }
    require(
        type(frozen_inputs.get("by_text")) is dict and set(frozen_inputs["by_text"]) == texts,
        "frozen tokenizer input catalog differs",
    )
    require(
        all(
            type(ids) is list and ids and all(type(token) is int and token >= 0 for token in ids)
            for ids in frozen_inputs["by_text"].values()
        ),
        "frozen actual integer tokenizer IDs missing",
    )
    partial = engine_id is not None
    if partial:
        events = [e for e in events if e["engine_id"] == engine_id]
        require(events, "requested engine trace missing")
        starts = rows(events, "case_begin")
        require(starts, "partial engine has no completed cases")
        names = [e.get("case_id") for e in starts]
        require(
            len(set(names)) == len(names) and set(names) <= set(expected),
            "partial case identity duplicate/unexpected",
        )
        expected = {name: expected[name] for name in names}
        ends = rows(events, "case_end")
        require(
            Counter(e.get("case_id") for e in ends) == Counter({k: 1 for k in expected}),
            "partial case ends incomplete/duplicated",
        )
        for event in starts + ends:
            require(
                all(event.get(k) == v for k, v in expected[event["case_id"]].items()),
                "partial case metadata mislabeled",
            )
            if event["event"] == "case_end":
                require(event.get("status") == "passed", "partial case failed or incomplete")
    else:
        validate_case_identity(events, config)
    engines = {e["engine_id"] for e in rows(events, "engine_start")}
    expected_engines = {f"{phase}-{mode}" for phase in ("smoke", "diagnostic") for mode in MODES}
    require(
        engines == ({engine_id} if partial else expected_engines),
        "fresh engine mode/phase catalog differs",
    )
    engine_results, sources_all, diagnostics = {}, [], []
    for name in sorted(engines):
        local = [e for e in events if e["engine_id"] == name]
        start, ready = one(local, "engine_start"), one(local, "engine_ready")
        phase, mode = name.split("-", 1)
        require(start["phase"] == phase and start["mode"] == mode, "engine identity mislabeled")
        require(
            start.get("inputs_sha256") == inputs_sha256,
            "engine input artifact hash differs from frozen input file",
        )
        sources = audit_sources(start, ready, config)
        sources_all.append(sources)
        worker = ready.get("worker")
        require(
            type(worker) is list and len(worker) == 1, "single real worker startup evidence missing"
        )
        capacity = audit_worker(worker[0], mode, phase, sources)
        require(
            start.get("config_sha256")
            == config_sha256
            == sources["source_sha256"].get("configs/m3-real-model.json"),
            "actual config differs from frozen source/startup hash",
        )
        prod = (
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
            if mode == "native"
            else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
        )
        require(start.get("production_class") == prod, "original production class name differs")
        if phase == "smoke":
            require(
                start.get("scheduler_class") == prod and not rows(local, "scheduler_init"),
                "production smoke replaced by observer",
            )
        else:
            identity = audit_scheduler_identity(local, name, mode)
            for key in ("scheduler", "config", "probe", "wait_guard"):
                require(
                    identity["module_sha256"][key]
                    == sources["source_sha256"][f"src/vllm_prefix_scheduler/{key}.py"],
                    "actual production module differs from frozen source",
                )
            diagnostics.extend(local)
            before, after = (
                one(local, "scheduler_shutdown_before"),
                one(local, "scheduler_shutdown_after"),
            )
            audit_drained(before.get("resources"))
            audit_drained(after.get("resources"))
            require(
                before["pid"] == after["pid"] and before["seq"] < after["seq"],
                "native scheduler shutdown ordering broken",
            )
            if mode == "guarded":
                require(
                    before["resources"].get("guard_closed") is False
                    and after["resources"].get("guard_closed") is True,
                    "guard closed before drain or failed to close",
                )
        completed, shutdown = one(local, "engine_completed"), one(local, "engine_shutdown")
        require(
            completed["pid"] == shutdown["pid"] and completed["seq"] < shutdown["seq"],
            "engine shutdown preceded completion",
        )
        require(
            shutdown.get("client_unfinished") == 0 and shutdown.get("observation_errors") == [],
            "client output processor/observer not drained",
        )
        exited = one(local, "process_exit")
        require(
            exited.get("exitcode") == 0
            and type(exited.get("exitcode")) is int
            and exited.get("owned_pids_remaining") == []
            and exited.get("owned_pids_before_cleanup") == []
            and exited.get("timed_out") is False,
            "engine/worker owned processes did not exit successfully",
        )
        engine_results[name] = {
            **capacity,
            "project_commit": sources["project_commit"],
            "smoke_direct_scheduler_object": False if phase == "smoke" else True,
        }
    require(
        len({s["project_commit"] for s in sources_all}) == 1
        and len({canonical(s["source_sha256"]) for s in sources_all}) == 1,
        "formal evidence mixes project commits/source conditions",
    )
    require(
        len({canonical(s["upstream_sha256"]) for s in sources_all}) == 1
        and len({canonical(s["model"]["files"]) for s in sources_all}) == 1,
        "formal evidence mixes upstream/model sources",
    )
    require(
        len({e["config_sha256"] for e in rows(events, "engine_start")}) == 1,
        "formal evidence mixes frozen configurations",
    )
    cases = {}
    for case, meta in expected.items():
        local = case_events(events, case)
        begin = one(local, "case_begin")
        require(
            begin["engine_id"]
            == f"{'smoke' if meta['scenario'] == 'smoke' else 'diagnostic'}-{meta['mode']}",
            "case executed in wrong engine/mode",
        )
        client = audit_client(events, case, meta, config)
        require(
            all(
                value["token_ids"] == frozen_inputs["by_text"].get(value["text"])
                for value in client["inputs"].values()
            ),
            "actual request token IDs differ from frozen tokenizer output",
        )
        cases[case] = client
        audit_case_timeline(events, case, meta["scenario"])
        if meta["scenario"] != "smoke":
            audit_native_requests(events, case, meta, client)
            audit_checkpoints(events, case, meta["scenario"])
            if meta["scenario"] == "mechanism":
                cases[case]["mechanism"] = audit_decision(events, case, meta, config)
    batch_results = audit_batches(diagnostics) if diagnostics else {"batches": 0}
    guard_results = {
        name: audit_guard(events, name) for name in engines if name == "diagnostic-guarded"
    }
    comparisons = []
    if not partial:
        require(
            cases["smoke:native"]["inputs"] == cases["smoke:off"]["inputs"],
            "native/off smoke inputs differ",
        )
        require(
            cases["smoke:native"]["results"] == cases["smoke:off"]["results"],
            "native/off greedy smoke outputs differ; investigate retained evidence",
        )
        common_inputs = [
            value["inputs"] for case, value in cases.items() if case.startswith("mechanism:")
        ]
        require(
            len({canonical(value) for value in common_inputs}) == 1,
            "mechanism modes/repeats have differing real input IDs",
        )
        require(
            len(
                {
                    value["kv_blocks"]
                    for name, value in engine_results.items()
                    if name.startswith("diagnostic-")
                }
            )
            == 1,
            "functional modes have unequal actual KV capacities",
        )
        for repeat in range(3):
            native, off = (f"mechanism:{mode}:{repeat}" for mode in ("native", "off"))
            require(
                cases[native]["results"] == cases[off]["results"],
                "native/off greedy functional outputs differ; investigate retained evidence",
            )
            prefix, guard = (
                f"mechanism:{mode}:{repeat}" for mode in ("prefix", "guarded_unexpired")
            )
            require(
                cases[prefix]["mechanism"]["first_order"][0].endswith("::C")
                and cases[guard]["mechanism"]["first_order"][0].endswith("::C"),
                "unexpired guarded/prefix actual decision mismatch",
            )
            comparisons.append(
                {
                    "repeat": repeat,
                    "native_off_exact_tokens": True,
                    "unexpired_prefix_decision": True,
                }
            )
        release = audit_resources(resources)
    else:
        release = None
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "scope": "partial" if partial else "complete",
        "accepted": not partial,
        "project_commit": sources_all[0]["project_commit"],
        "case_count": len(cases),
        "cases": cases,
        "engines": engine_results,
        "batch_audit": batch_results,
        "guard_lifecycle": guard_results,
        "comparisons": comparisons,
        "resource_release": release,
        "limitations": [
            "Controlled pause and waiting interventions; no natural queue fairness or performance conclusion.",
            "Smoke scheduler object selection is supported by fixed construction source and configuration; direct object/MRO observation belongs to diagnostics.",
            "Scheduled token intervals and runner execution do not establish FLOPs or speedup.",
        ],
    }
