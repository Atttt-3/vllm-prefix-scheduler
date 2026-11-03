"""Independent M5 request, runner and paired statistics audit (stdlib only)."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from vllm_prefix_scheduler.m5_campaign import MODES, config_matrix, read_json
from vllm_prefix_scheduler.m5_campaign import TASK as TASK_ID
from vllm_prefix_scheduler.m5_campaign import digest as sha256

STATUSES = ("completed", "failed", "cancelled", "timeout", "unfinished")
CLOCK_KEYS = ("host", "boot_id", "time_namespace", "clock")
PHASE_KEYS = (
    "load_start_ns",
    "load_end_ns",
    "warmup_start_ns",
    "warmup_end_ns",
    "measure_start_ns",
    "arrival_end_ns",
    "drain_end_ns",
)


class AuditError(ValueError):
    """An evidence contract check failed."""


def require(condition, reason):
    if not condition:
        raise AuditError(reason)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def require_dict(value, label):
    require(type(value) is dict, f"{label}: object required")
    return value


def require_list(value, label):
    require(type(value) is list, f"{label}: array required")
    return value


def require_ns(value, label):
    require(integer(value, 1), f"{label}: positive timestamp required")
    return value


def token_ids(value):
    return type(value) is list and all(integer(token) for token in value)


def quantile(values, percent):
    """Nearest rank, with an exact integral index for integer percentages."""
    return sorted(values)[(percent * len(values) + 99) // 100 - 1] if values else None


def distribution(values):
    return {
        "n": len(values),
        "p50": quantile(values, 50),
        "p95": quantile(values, 95),
        "p99": quantile(values, 99),
        "max": max(values, default=None),
    }


def clock_gate(client, engine, gates):
    a = require_dict(client.get("clock"), "client clock")
    b = require_dict(engine.get("clock"), "engine clock")
    require(
        all(type(a.get(k)) is str and a[k] and a[k] == b.get(k) for k in CLOCK_KEYS)
        and a["clock"] == "CLOCK_MONOTONIC"
        and integer(a.get("pid"), 1)
        and integer(b.get("pid"), 1),
        "clock domain differs",
    )
    handshake = require_dict(client.get("clock_check"), "clock check")
    send, remote, recv = (
        require_ns(handshake.get(k), f"clock {k}") for k in ("send_ns", "engine_ns", "recv_ns")
    )
    require(
        send <= remote <= recv and recv - send <= gates["max_clock_rtt_ns"],
        "clock handshake outside bound",
    )


def workload_gate(inputs, config, pair, phase):
    require(inputs.get("schema_version") == 1, "input schema differs")
    require(
        inputs.get("tokenizer_revision") == config["tokenizer_revision"],
        "input tokenizer revision differs",
    )
    if pair:
        require(
            (inputs.get("scenario"), inputs.get("seed")) == (pair["scenario"], pair["seed"]),
            "input pair/scenario/seed differs",
        )
    require(
        type(inputs.get("scenario")) is str and integer(inputs.get("seed")),
        "input identity missing",
    )
    scenario = inputs["scenario"]
    require(scenario in config["scenarios"], "workload scenario differs")
    block = config["block_size"]
    require(
        inputs.get("task_id") == TASK_ID and inputs.get("block_size") == block,
        "workload task or block size differs",
    )
    rate, interval = inputs.get("rate_rps"), inputs.get("interval_ms")
    if scenario == "serial":
        require(
            rate is None
            and type(interval) in (int, float)
            and math.isfinite(interval)
            and interval > 0,
            "serial offered interval invalid",
        )
        spacing = Fraction(str(interval)) * 1_000_000
    elif scenario == "mechanism":
        require(
            rate is None and interval is None and phase == "diagnostic",
            "mechanism must be a diagnostic with controlled arrival",
        )
        spacing = 0
    else:
        require(
            interval is None and type(rate) in (int, float) and math.isfinite(rate) and rate > 0,
            "offered rate invalid",
        )
        spacing = 1_000_000_000 / Fraction(str(rate))
    sampling = require_dict(inputs.get("sampling"), "sampling")
    require(
        sampling
        == {**config["sampling"], "max_tokens": config["scenarios"][scenario]["output_tokens"]},
        "sampling differs from frozen workload",
    )
    rows = require_list(inputs.get("requests"), "planned requests")
    require(
        rows and (phase != "formal" or len(rows) == config["request_count"]),
        "planned request count differs",
    )
    expected = {}
    previous = 0
    jitter_rng = random.Random(inputs["seed"] ^ 0x4D350001)
    jitter_bound_ns = int(spacing / 8)
    for i, row in enumerate(rows):
        require_dict(row, "planned request")
        identity = row.get("request_id")
        offset = row.get("offset_ns")
        ids = row.get("prompt_token_ids")
        require(
            type(identity) is str and identity and identity not in expected,
            "invalid or duplicate planned request",
        )
        require(integer(offset, previous), "planned offsets must be nondecreasing")
        if scenario == "burst":
            burst = config["scenarios"][scenario]["burst_size"]
            group, within = divmod(i, burst)
            planned_offset = int(
                group * burst * spacing + within * config["scenarios"][scenario]["burst_gap_ns"]
            )
        else:
            planned_offset = int(i * spacing)
            if scenario == "mixed" and 0 < i < len(rows) - 1:
                planned_offset += jitter_rng.randint(-jitter_bound_ns, jitter_bound_ns)
        require(offset == planned_offset, "arrival offsets differ from absolute rate")
        require(row.get("class") in {"hot", "cold"}, "frozen hot/cold class missing")
        require(token_ids(ids) and ids, "frozen prompt tokens missing")
        expected[identity] = row
        previous = offset
    family_keys = {}
    block_owner = {}
    measured_warmup = require_list(inputs.get("warmup"), "warmup workload")
    all_ids = set()
    prefix_length = config["scenarios"][scenario]["prefix_tokens"]
    for row in [*measured_warmup, *rows]:
        require_dict(row, "frozen input row")
        identity = row.get("request_id")
        require(
            type(identity) is str and identity and identity not in all_ids,
            "warmup/measurement request identity differs",
        )
        all_ids.add(identity)
        category, family = row.get("class"), row.get("family")
        salt, ids = row.get("cache_salt"), row.get("prompt_token_ids")
        require(
            category in {"hot", "cold"}
            and type(salt) is str
            and salt
            and token_ids(ids)
            and len(ids) >= block,
            "input class/prefix evidence missing",
        )
        first_block = tuple(ids[:block])
        if category == "hot":
            require(
                type(family) is str and family and len(ids) > prefix_length,
                "hot request lacks reusable prefix family",
            )
            reusable = (salt, tuple(ids[:prefix_length]))
            require(
                family_keys.setdefault(family, reusable) == reusable,
                "frozen token prefixes differ within hot family",
            )
            owner = ("family", family)
        else:
            require(family is None, "cold request incorrectly assigned a hot family")
            owner = ("cold", identity)
        require(
            block_owner.setdefault(first_block, owner) == owner,
            "unrelated requests share first cache block",
        )
    return expected


def client_requests(client, inputs, expected, gates, phase, *, rate_pilot=False):
    phases = require_dict(client.get("phases"), "client phases")
    times = [require_ns(phases.get(k), k) for k in PHASE_KEYS]
    require(times == sorted(times), "client phases out of order")
    start, arrivals, end = times[-3:]
    require(
        arrivals == start + max(row["offset_ns"] for row in expected.values()),
        "arrival horizon differs from frozen offsets",
    )
    deadline = gates["deadline_ns"]
    cutoff = client.get("drain_deadline_ns")
    require(
        client.get("deadline_ns") == deadline and cutoff == arrivals + deadline,
        "round deadline differs",
    )
    rows = require_list(client.get("requests"), "request ledger")
    require(len(rows) == len(expected), "request ledger denominator differs")
    by_id = {}
    lag = []
    for row in rows:
        require_dict(row, "request ledger row")
        request_id = row.get("request_id")
        require(
            request_id in expected and request_id not in by_id,
            "request ledger missing/duplicate/unknown identity",
        )
        missing = {
            "planned_ns",
            "submit_ns",
            "first_token_ns",
            "last_token_ns",
            "end_ns",
            "status",
            "token_ids",
            "cached_tokens",
            "chunk_tokens",
        } - row.keys()
        require(not missing, "request ledger required field missing")
        plan = expected[request_id]
        planned = start + plan["offset_ns"]
        require(row["planned_ns"] == planned, "planned timestamp differs from frozen offset")
        status = row["status"]
        require(status in STATUSES, "request terminal status differs")
        submitted, first, last, terminal = (
            row[k] for k in ("submit_ns", "first_token_ns", "last_token_ns", "end_ns")
        )
        if submitted is None:
            require(status in {"timeout", "unfinished"}, "unsubmitted request terminal differs")
        else:
            require(
                integer(submitted, planned) and submitted <= end,
                "submission timestamp outside round",
            )
            if phase != "diagnostic":
                lag.append(submitted - planned)
        if terminal is None:
            require(status == "unfinished", "terminal timestamp missing")
        else:
            require(
                integer(terminal, submitted if submitted is not None else planned)
                and terminal <= end,
                "terminal timestamp outside round",
            )
        if status == "completed":
            require(terminal <= planned + deadline, "completed request past deadline")
        if status == "timeout":
            require(terminal >= planned + deadline, "timeout before request deadline")
        cached = row["cached_tokens"]
        require(
            cached is None or integer(cached) and cached <= len(plan["prompt_token_ids"]),
            "cached token count invalid",
        )
        chunks = require_list(row["chunk_tokens"], "client token chunks")
        delivered = []
        chunk_times = []
        for chunk in chunks:
            require_dict(chunk, "client token chunk")
            at = chunk.get("at_ns")
            tokens = chunk.get("token_ids")
            require(
                submitted is not None
                and integer(at, submitted)
                and at <= (terminal if terminal is not None else end)
                and (not chunk_times or at >= chunk_times[-1]),
                "client token chunk outside request",
            )
            require(token_ids(tokens) and tokens, "client chunk has no real generated tokens")
            chunk_times.append(at)
            delivered.extend(tokens)
        require(
            row["token_ids"] == delivered
            and first == (chunk_times[0] if chunk_times else None)
            and last == (chunk_times[-1] if chunk_times else None),
            "client token times or IDs differ from chunks",
        )
        if status == "completed":
            require(
                delivered and len(delivered) <= inputs["sampling"]["max_tokens"],
                "completed request output missing/over limit",
            )
            if inputs["sampling"].get("ignore_eos") is True:
                require(
                    len(delivered) == inputs["sampling"]["max_tokens"],
                    "completed request output shorter than frozen generation length",
                )
        by_id[request_id] = row
    submit_lag = distribution(lag)
    if phase != "diagnostic" and not rate_pilot:
        require(
            lag
            and submit_lag["p99"] <= gates["submit_p99_ns"]
            and submit_lag["max"] <= gates["submit_max_ns"],
            "submission lag exceeds gate",
        )
        require(len(lag) == len(rows), "submission backlog left planned requests unsubmitted")
    return by_id, submit_lag


def add_interval(previous, start, end):
    """Record a successfully executed interval and return its overlap with prior execution."""
    if start >= end:
        return 0
    overlap = sum(max(0, min(right, end) - max(left, start)) for left, right in previous)
    combined = []
    for left, right in sorted([*previous, (start, end)]):
        if combined and left <= combined[-1][1]:
            combined[-1] = combined[-1][0], max(combined[-1][1], right)
        else:
            combined.append((left, right))
    previous[:] = combined
    return overlap


def engine_batches(engine, plan, warmup, requests, tier, mode, phases, config, native_end):
    expected_tokens = defaultdict(list)
    enqueues = {}
    all_inputs = {**warmup, **plan}
    seen_generations = set()
    for row in require_list(engine.get("enqueues"), "engine enqueues"):
        require_dict(row, "engine enqueue")
        identity = row.get("request_id")
        require(
            identity in all_inputs
            and identity not in enqueues
            and integer(row.get("generation"))
            and row["generation"] not in seen_generations,
            "enqueue identity/generation differs",
        )
        seen_generations.add(row["generation"])
        enqueue_time = require_ns(row.get("enqueued_ns"), "enqueue timestamp")
        require(
            not enqueues or enqueue_time >= next(reversed(enqueues.values()))["enqueued_ns"],
            "enqueue time goes backwards",
        )
        ids = all_inputs[identity]["prompt_token_ids"]
        prompt_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        require(
            row.get("prompt_sha256") == prompt_hash,
            "actual native prompt differs from frozen token IDs",
        )
        require(
            type(row.get("block_hashes")) is list
            and len(row["block_hashes"]) == len(ids) // config["block_size"]
            and all(type(value) is str and value for value in row["block_hashes"]),
            "native block hash evidence missing",
        )
        if identity in requests:
            client = requests[identity]
            require(
                client["submit_ns"] is not None
                and enqueue_time >= client["submit_ns"]
                and enqueue_time <= native_end,
                "engine enqueue outside client request",
            )
        enqueues[identity] = row

    first = {}
    previous = defaultdict(list)
    work = Counter()
    batches = {}
    schedule_times = []
    prior_batch_ns = 0
    for row in require_list(engine.get("batches"), "engine batches"):
        require_dict(row, "engine batch")
        batch_id = row.get("batch_id")
        require(
            type(batch_id) is int and batch_id == len(batches) + 1,
            "missing, duplicate or invalid batch ID",
        )
        at = require_ns(row.get("scheduled_ns"), "schedule timestamp")
        require(at >= prior_batch_ns, "schedule time goes backwards")
        prior_batch_ns = at
        duration = row.get("schedule_ns")
        require(integer(duration), "schedule duration missing")
        scheduled = require_list(row.get("scheduled"), "scheduled intervals")
        require(row.get("consumed") is True, "batch consume proof missing")
        scheduled_by_id = {}
        for interval in scheduled:
            require_dict(interval, "dispatch interval")
            identity = interval.get("request_id")
            start = interval.get("start")
            end = interval.get("end")
            length = interval.get("prompt_length")
            require(
                identity in enqueues
                and identity not in scheduled_by_id
                and enqueues[identity]["generation"] == interval.get("generation"),
                "dispatch request generation differs",
            )
            require(
                integer(start)
                and integer(end, start + 1)
                and integer(length, 1)
                and integer(interval.get("preemptions")),
                "dispatch interval invalid",
            )
            require(
                length == len(all_inputs[identity]["prompt_token_ids"]),
                "scheduled prompt length differs from frozen tokens",
            )
            require(at >= enqueues[identity]["enqueued_ns"], "dispatch precedes enqueue")
            scheduled_by_id[identity] = interval
            first.setdefault(identity, (at, batch_id))

        in_measurement = phases["measure_start_ns"] <= at <= native_end
        if row.get("stage") != "measure":
            require(not set(scheduled_by_id) & set(plan), "measured request outside measure stage")
        if in_measurement:
            schedule_times.append(duration)
        runner = row.get("runner")
        if not scheduled or tier == "endpoint":
            require(runner is None, "unexpected runner evidence for endpoint/empty batch")
        else:
            require(row.get("materialized") is True, "positive batch materialization proof missing")
            runner = require_dict(runner, "successful runner input")
            runner_rows = require_list(runner.get("rows"), "actual runner positions")
            observed = {}
            for actual in runner_rows:
                require_dict(actual, "actual runner row")
                identity = actual.get("request_id")
                require(
                    identity in scheduled_by_id and identity not in observed,
                    "runner request missing, duplicate or unscheduled",
                )
                dispatch = scheduled_by_id[identity]
                require(
                    all(actual.get(key) == dispatch[key] for key in ("generation", "start", "end"))
                    and integer(actual.get("prefill_length"), dispatch["prompt_length"])
                    and actual.get("position_source") == "runner_positions_d2h",
                    "runner positions differ from actual scheduled request generation",
                )
                observed[identity] = actual
            require(set(observed) == set(scheduled_by_id), "positive batch runner rows missing")
            effective = sum(r["end"] - r["start"] for r in runner_rows)
            require(
                runner.get("effective_tokens") == effective
                and integer(runner.get("padded_tokens"), effective),
                "runner effective/padded tokens disagree",
            )
            if in_measurement and set(observed) & set(plan):
                require(set(observed) <= set(plan), "warmup and measured requests share a batch")
                work["effective_tokens"] += effective
                work["padded_tokens"] += runner["padded_tokens"]
            for identity, actual in observed.items():
                if identity not in plan:
                    continue
                require(in_measurement, "measured execution outside measurement")
                prompt = scheduled_by_id[identity]["prompt_length"]
                start, end = actual["start"], actual["end"]
                work["prefill_tokens"] += max(0, min(end, prompt) - start)
                work["decode_tokens"] += max(0, end - max(start, prompt))
                work["recomputed_prefill_tokens"] += add_interval(
                    previous[identity], start, min(end, prompt)
                )
        delivered = set()
        for output in require_list(row.get("outputs"), "engine consumed token outputs"):
            require_dict(output, "engine token output")
            identity = output.get("request_id")
            tokens = output.get("token_ids")
            require(
                identity in scheduled_by_id
                and identity not in delivered
                and output.get("generation") == scheduled_by_id[identity]["generation"]
                and token_ids(tokens)
                and tokens,
                "engine generated output generation differs",
            )
            delivered.add(identity)
            expected_tokens[identity].extend(tokens)
        batches[batch_id] = (at, scheduled_by_id)

    require(
        work["effective_tokens"] == work["prefill_tokens"] + work["decode_tokens"],
        "effective tokens differ from prefill/decode accounting",
    )
    require(set(warmup) <= set(first), "warmup request lacks native execution")
    for identity, row in requests.items():
        observed = row["token_ids"]
        authoritative = expected_tokens[identity]
        require(
            observed == authoritative[: len(observed)]
            and (row["status"] != "completed" or observed == authoritative),
            "client tokens differ from consumed native outputs",
        )
        first_at = first.get(identity)
        if row["token_ids"] or row["status"] == "completed":
            require(
                identity in enqueues and first_at is not None,
                "output request lacks enqueue/positive dispatch",
            )
            require(first_at[0] <= row["first_token_ns"], "first token precedes positive dispatch")
    wait = fairness(engine, plan, enqueues, requests, batches, first, mode, config)
    stage_samples = {name: [] for name in ("window", "score", "sort_exclusive", "queue_commit")}
    for row in require_list(engine.get("overhead"), "stage overhead"):
        require_dict(row, "stage timing row")
        require(
            tier == "profile" and row.get("batch_id") in batches,
            "stage timing outside profile tier or batch",
        )
        window_ns, score, sorting, commit, reorder = (
            row.get(k)
            for k in ("window_ns", "score_ns", "sort_inclusive_ns", "commit_ns", "reorder_ns")
        )
        require(
            all(integer(v) for v in (window_ns, score, sorting, commit, reorder))
            and sorting >= score
            and reorder >= window_ns + sorting + commit,
            "invalid stage timing",
        )
        if row.get("stage") == "measure":
            for key, value in zip(
                stage_samples, (window_ns, score, sorting - score, commit), strict=True
            ):
                stage_samples[key].append(value)
    return (
        work,
        {
            "schedule": distribution(schedule_times),
            **{key: distribution(values) for key, values in stage_samples.items()},
        },
        enqueues,
        first,
        wait,
    )


def fairness(engine, plan, enqueues, requests, batches, first, mode, config):
    whole = Counter()
    window = Counter()
    by_hot = Counter()
    window_by_hot = Counter()
    competing = 0
    valid_steps = {}
    protected_windows = 0
    for row in require_list(engine.get("decisions"), "candidate decisions"):
        require_dict(row, "candidate decision")
        batch_id = row.get("batch_id")
        require(
            batch_id in batches and batch_id not in valid_steps,
            "candidate decision batch missing/duplicate",
        )
        candidates, ordered, protected = (
            require_list(row.get(key), key) for key in ("candidates", "ordered", "protected")
        )
        require(
            len(candidates) <= config["window"]
            and len(candidates) == len(set(candidates))
            and Counter(candidates) == Counter(ordered)
            and set(candidates) <= set(enqueues)
            and set(protected) <= set(candidates)
            and len(protected) == len(set(protected)),
            "candidate window membership differs",
        )
        at = require_ns(row.get("decision_ns"), "candidate decision timestamp")
        require(
            at <= batches[batch_id][0]
            and all(
                enqueues[i]["enqueued_ns"] <= at and (i not in first or first[i][1] >= batch_id)
                for i in candidates
            ),
            "candidate decision timestamp or eligibility invalid",
        )
        shadow = mode in {"native", "off"}
        require(
            row.get("window_kind") == ("shadow" if shadow else "production"),
            "candidate window observer label differs",
        )
        if shadow:
            require(ordered == candidates, "FCFS control reordered candidate window")
        if mode == "guarded" and len(candidates) >= 2:
            timestamps = require_dict(row.get("guard_enqueued_ns"), "native guard enqueue clocks")
            require(
                set(timestamps) == set(candidates)
                and all(
                    integer(value, 1)
                    and value <= min(at, enqueues[i]["enqueued_ns"])
                    and (i not in requests or value >= requests[i]["submit_ns"])
                    for i, value in timestamps.items()
                ),
                "native guard enqueue clocks do not match candidates",
            )
            enqueue_order = {identity: n for n, identity in enumerate(enqueues)}
            threshold_ns = config["max_wait_ms"] * 1_000_000
            expected = sorted(
                (i for i in candidates if at - timestamps[i] >= threshold_ns),
                key=lambda i: (timestamps[i], enqueue_order[i]),
            )
            require(
                protected == expected and ordered[: len(expected)] == expected,
                "guarded window protection order/threshold violated",
            )
        else:
            require(
                not protected and "guard_enqueued_ns" not in row,
                "decision without guarded sorting claims protection",
            )
        valid_steps[batch_id] = set(candidates)
        if {plan[i]["class"] for i in candidates if i in plan} == {"cold", "hot"}:
            competing += 1
        if set(protected) & set(plan):
            protected_windows += 1

    order = {identity: i for i, identity in enumerate(enqueues)}
    for younger, (dispatch, batch) in first.items():
        if younger not in plan:
            continue
        for older, enqueue in enqueues.items():
            if older not in plan or order[older] >= order[younger]:
                continue
            older_first = first.get(older)
            if (
                enqueue["enqueued_ns"] > dispatch
                or older_first is not None
                and older_first[1] <= batch
            ):
                continue
            terminal = requests[older]["end_ns"]
            if terminal is not None and terminal <= dispatch:
                continue
            whole[older] += 1
            young_hot = plan[younger]["class"] == "hot"
            by_hot[older] += young_hot
            candidates = valid_steps.get(batch, set())
            if older in candidates and younger in candidates:
                window[older] += 1
                window_by_hot[older] += young_hot
    return whole, window, by_hot, window_by_hot, competing, protected_windows


def stratum_metrics(ids, requests, enqueues, first, whole, window, by_hot, window_by_hot, deadline):
    selected = [requests[identity] for identity in ids]
    completed = [row for row in selected if row["status"] == "completed"]
    counts = {status: sum(row["status"] == status for row in selected) for status in STATUSES}
    counts.update(
        planned=len(selected),
        submitted=sum(row["submit_ns"] is not None for row in selected),
        timely_first_token=sum(
            row["first_token_ns"] is not None
            and row["first_token_ns"] <= row["planned_ns"] + deadline
            for row in selected
        ),
    )
    dispatched_ids = [identity for identity in ids if identity in first]
    tpots = [
        (row["last_token_ns"] - row["first_token_ns"]) / (len(row["token_ids"]) - 1)
        for row in completed
        if len(row["token_ids"]) >= 2
    ]
    return {
        "counts": counts,
        "completion_rate": counts["completed"] / counts["planned"] if selected else None,
        "timely_first_token_rate": counts["timely_first_token"] / counts["planned"]
        if selected
        else None,
        "ttft_ns": distribution([r["first_token_ns"] - r["planned_ns"] for r in completed]),
        "submit_ttft_ns": distribution([r["first_token_ns"] - r["submit_ns"] for r in completed]),
        "tpot_ns": distribution(tpots),
        "enqueue_to_first_dispatch_ns": distribution(
            [first[i][0] - enqueues[i]["enqueued_ns"] for i in dispatched_ids]
        ),
        "whole_queue_overtakes": distribution([whole[i] for i in ids]),
        "window_overtakes": distribution([window[i] for i in ids]),
        "whole_queue_hot_overtakes": distribution([by_hot[i] for i in ids]),
        "window_hot_overtakes": distribution([window_by_hot[i] for i in ids]),
        "cached_tokens": distribution(
            [r["cached_tokens"] for r in selected if r["cached_tokens"] is not None]
        ),
    }


def boundary_gate(engine, client, inputs, tier):
    labels = ("pre-reset", "cache-reset", "measure-start", "measure-drained")
    raw = require_list(engine.get("boundaries"), "native boundary checks")
    selected = {}
    previous = 0
    for row in raw:
        require_dict(row, "native boundary")
        label = row.get("label")
        at = require_ns(row.get("at_ns"), "boundary timestamp")
        require(at >= previous, "native boundary time goes backwards")
        previous = at
        if label in {f"{base}-poll" for base in labels}:
            continue
        require(label in labels and label not in selected, "unknown or duplicate boundary")
        selected[label] = row
        require(
            row.get("runner_request_ids") == [] and row.get("runner_pending") is False,
            "native runner residual at boundary",
        )
        if tier != "endpoint":
            lists = (
                "requests",
                "waiting",
                "running",
                "holding",
                "deferred",
                "finished_req_ids",
                "reset_preempted_req_ids",
                "inflight_prefills",
                "guard_records",
                "observation_errors",
            )
            counts = (
                "deferred_frees",
                "request_blocks",
                "non_null_ref_sum",
                "pending_batches",
                "inflight_tokens",
                "stale_tokens",
                "output_placeholders",
                "m5_pending",
            )
            require(
                all(row.get(key) == [] for key in lists)
                and all(type(row.get(key)) is int and row[key] == 0 for key in counts)
                and "guard_fault_code" in row
                and row["guard_fault_code"] is None,
                "native scheduler residual at boundary",
            )
            capacity = engine["profile"]["num_gpu_blocks"] - 1
            require(
                all(
                    row.get(key) == capacity
                    for key in (
                        "free_count",
                        "free_unique_count",
                        "free_traversed_count",
                        "total_non_null_blocks",
                    )
                ),
                "native free KV blocks not restored at boundary",
            )
            require(
                integer(row.get("cached_block_count")) and row["cached_block_count"] <= capacity,
                "cached boundary block count missing",
            )
            if label == "cache-reset":
                require(row["cached_block_count"] == 0, "cache reset left native APC entries")
    require(tuple(selected) == labels, "required native drain boundary missing")
    phases = client["phases"]
    require(
        phases["warmup_start_ns"] <= selected["pre-reset"]["at_ns"]
        and selected["measure-start"]["at_ns"] <= phases["warmup_end_ns"]
        and selected["measure-drained"]["at_ns"] >= phases["drain_end_ns"],
        "native boundaries do not bind the client phases",
    )
    warm = require_list(client.get("warmup"), "client warmup")
    primer = require_dict(client.get("primer"), "runtime primer")
    require(
        primer.get("request_id") == "runtime-primer"
        and primer.get("status") == "completed"
        and [r["request_id"] for r in warm] == [r["request_id"] for r in inputs["warmup"]]
        and all(r.get("status") == "completed" for r in warm),
        "warmup workload incomplete",
    )
    return selected["measure-drained"]["at_ns"]


def pressure_metrics(engine, config, plan, enqueues, first, requests, native_end, reusable_blocks):
    evictions = require_list(engine.get("evictions"), "KV evictions")
    allocations = require_list(engine.get("allocations"), "KV allocations")
    preemptions = require_list(engine.get("preemptions"), "preemptions")
    batch_times = {row["batch_id"]: row["scheduled_ns"] for row in engine["batches"]}
    block_size = config["block_size"]
    hot_blocks = {}
    for identity, row in enqueues.items():
        if identity not in plan or plan[identity]["class"] != "hot":
            continue
        terminal = requests[identity]["end_ns"]
        cutoff = first[identity][0] if identity in first else terminal or native_end
        for cache_key in row["block_hashes"][:reusable_blocks]:
            hot_blocks[cache_key] = max(cutoff, hot_blocks.get(cache_key, 0))

    known_allocations = set()
    for row in allocations:
        require_dict(row, "KV allocation")
        identity, batch = row.get("request_id"), row.get("batch_id")
        require(
            identity in enqueues
            and batch in batch_times
            and row.get("generation") == enqueues[identity]["generation"]
            and integer(row.get("at_ns"), enqueues[identity]["enqueued_ns"])
            and row["at_ns"] <= batch_times[batch]
            and integer(row.get("free_before"))
            and integer(row.get("free_after"))
            and integer(row.get("hit_tokens"))
            and row["hit_tokens"] % block_size == 0
            and type(row.get("success")) is bool,
            "allocation event unbound to native batch",
        )
        if row["success"]:
            known_allocations.add((batch, identity))

    cached, reusable = 0, 0
    for row in evictions:
        require_dict(row, "KV eviction")
        if row.get("stage") != "measure":
            continue
        identity, batch = row.get("request_id"), row.get("batch_id")
        require(
            (batch, identity) in known_allocations
            and row.get("reusable") is True
            and integer(row.get("block_id"), 1)
            and integer(row.get("at_ns"), enqueues[identity]["enqueued_ns"])
            and row["at_ns"] <= batch_times[batch]
            and type(row.get("cache_key")) is str
            and row["cache_key"],
            "KV eviction unbound",
        )
        cached += 1
        reusable += row["at_ns"] < hot_blocks.get(row["cache_key"], 0)
    count = 0
    for row in preemptions:
        require_dict(row, "preemption")
        identity, batch = row.get("request_id"), row.get("batch_id")
        require(
            identity in enqueues
            and batch in batch_times
            and row.get("generation") == enqueues[identity]["generation"]
            and integer(row.get("preemptions"), 1)
            and integer(row.get("at_ns"), enqueues[identity]["enqueued_ns"])
            and row["at_ns"] <= batch_times[batch],
            "preemption event unbound",
        )
        count += row.get("stage") == "measure"
    return {"cached_evictions": cached, "reusable_evictions": reusable, "preemptions": count}


def audit_run(config_path, inputs_path, directory, *, pair_id=None, mode=None, phase="formal"):
    config = require_dict(read_json(config_path), "M5 config")
    inputs = require_dict(read_json(inputs_path), "M5 inputs")
    client = require_dict(read_json(directory / "client.json"), "client evidence")
    engine = require_dict(read_json(directory / "engine.json"), "engine evidence")
    gates = require_dict(config.get("gates"), "M5 gates")
    require(
        client.get("task_id") == config.get("task_id") == TASK_ID
        and client.get("schema_version") == engine.get("schema_version") == 1,
        "task or trace schema differs",
    )
    config_hash = sha256(config_path)
    inputs_hash = sha256(inputs_path)
    require(
        client.get("input_sha256") == inputs_hash and client.get("config_sha256") == config_hash,
        "input/config source hashes differ",
    )
    require(
        client.get("phase") == phase and phase in {"formal", "calibration", "diagnostic"},
        "run phase differs",
    )
    mode = client.get("mode") if mode is None else mode
    require(mode in MODES and client.get("mode") == mode, "mode label differs")
    pair_id = client.get("pair_id") if pair_id is None else pair_id
    require(
        type(pair_id) is str and pair_id and client.get("pair_id") == pair_id, "pair label differs"
    )
    pair = config_matrix(config).get(pair_id) if phase == "formal" else None
    require(phase != "formal" or pair is not None, "formal pair not in frozen matrix")
    tier = client.get("tier")
    require(
        tier in {"light", "endpoint", "profile"} and (phase != "formal" or tier == "light"),
        "invalid observation tier for phase",
    )
    plan = workload_gate(inputs, config, pair, phase)
    require(
        (client.get("scenario"), client.get("seed")) == (inputs["scenario"], inputs["seed"]),
        "client input scenario/seed differs",
    )
    require(engine.get("errors") == [], "engine observer errors or errors field missing")
    clock_gate(client, engine, gates)
    rate_pilot = (
        phase == "calibration"
        and mode == "native"
        and tier == "light"
        and inputs["scenario"] in ("burst", "mixed")
        and pair_id
        in {
            f"{inputs['scenario']}-{rate}rps"
            for rate in config["calibration"]["candidate_rates"]
            if inputs["rate_rps"] == rate
        }
    )
    requests, submit_lag = client_requests(
        client, inputs, plan, gates, phase, rate_pilot=rate_pilot
    )
    warmup = {
        row["request_id"]: row for row in require_list(inputs.get("warmup"), "warmup workload")
    }
    warmup["runtime-primer"] = next(iter(plan.values()))
    require(not set(warmup) & set(plan), "warmup request IDs collide with measurement")
    phases = client["phases"]
    native_end = boundary_gate(engine, client, inputs, tier)
    if tier == "endpoint":
        require(
            all(
                engine.get(key) == []
                for key in (
                    "enqueues",
                    "batches",
                    "decisions",
                    "overhead",
                    "evictions",
                    "allocations",
                    "preemptions",
                )
            ),
            "endpoint trace contains instrumented batches or decisions",
        )
        work, stage_costs, enqueues, first = None, {"schedule": distribution([])}, {}, {}
        wait = Counter(), Counter(), Counter(), Counter(), 0, 0
    else:
        work, stage_costs, enqueues, first, wait = engine_batches(
            engine, plan, warmup, requests, tier, mode, phases, config, native_end
        )
    whole, window, by_hot, window_by_hot, competing, protected = wait
    profile = require_dict(engine.get("profile"), "actual engine profile")
    require(integer(profile.get("num_gpu_blocks"), 2), "actual physical KV blocks missing")
    reusable_blocks = (
        config["scenarios"][inputs["scenario"]]["prefix_tokens"] // config["block_size"]
    )
    pressure = pressure_metrics(
        engine, config, plan, enqueues, first, requests, native_end, reusable_blocks
    )
    strata = {"all": list(plan)}
    for label in ("hot", "cold"):
        strata[label] = [identity for identity, row in plan.items() if row["class"] == label]
    deadline = gates["deadline_ns"]
    split = {
        label: stratum_metrics(
            ids, requests, enqueues, first, whole, window, by_hot, window_by_hot, deadline
        )
        for label, ids in strata.items()
    }
    start, end = phases["measure_start_ns"], phases["drain_end_ns"]
    observed_ns = end - start
    require(observed_ns > 0, "nonpositive observed throughput window")
    horizon = client["drain_deadline_ns"]
    completed = sum(
        r["status"] == "completed" and r["end_ns"] <= horizon for r in requests.values()
    )
    generated = sum(
        len(chunk["token_ids"])
        for r in requests.values()
        for chunk in r["chunk_tokens"]
        if chunk["at_ns"] <= horizon
    )
    last_terminal = max(
        (r["end_ns"] for r in requests.values() if r["end_ns"] is not None), default=end
    )
    last_window = last_terminal - start
    active, max_outstanding = 0, 0
    transitions = []
    for row in requests.values():
        if row["submit_ns"] is not None:
            transitions.append((row["submit_ns"], 0, 1))
            transitions.append((row["end_ns"] if row["end_ns"] is not None else end, 1, -1))
    for _, _, change in sorted(transitions):
        active += change
        max_outstanding = max(max_outstanding, active)
    if inputs["scenario"] == "serial":
        require(max_outstanding <= 1 and competing == 0, "serial control has competing requests")
    throughput = {
        "window_ns": observed_ns,
        "completed_requests": completed,
        "generated_tokens": generated,
        "requests_per_s": completed * 1_000_000_000 / observed_ns,
        "tokens_per_s": generated * 1_000_000_000 / observed_ns,
        "fixed_window_ns": horizon - start,
        "fixed_requests_per_s": completed * 1_000_000_000 / (horizon - start),
        "fixed_tokens_per_s": generated * 1_000_000_000 / (horizon - start),
        "last_terminal_requests_per_s": completed * 1_000_000_000 / last_window
        if last_window
        else None,
        "last_terminal_tokens_per_s": generated * 1_000_000_000 / last_window
        if last_window
        else None,
    }
    metrics = {
        "strata": split,
        "throughput": throughput,
        "duration_ns": {
            "load": phases["load_end_ns"] - phases["load_start_ns"],
            "warmup": phases["warmup_end_ns"] - phases["warmup_start_ns"],
            "arrival": phases["arrival_end_ns"] - start,
            "drain": end - phases["arrival_end_ns"],
            "measure_through_drain": observed_ns,
            "last_terminal_from_start": last_window,
            "max_submit_to_end": max(
                (
                    r["end_ns"] - r["submit_ns"]
                    for r in requests.values()
                    if r["status"] == "completed"
                ),
                default=None,
            ),
        },
        "submit_lag_ns": submit_lag,
        "kv_blocks": profile["num_gpu_blocks"],
        "max_outstanding": max_outstanding,
        "pressure": {
            "competing_windows": competing,
            "protected_windows": protected,
            **pressure,
        },
        "schedule_ns": stage_costs["schedule"],
        "stage_costs_ns": stage_costs,
        "work": None
        if work is None
        else {
            key: work[key]
            for key in (
                "prefill_tokens",
                "decode_tokens",
                "recomputed_prefill_tokens",
                "effective_tokens",
                "padded_tokens",
            )
        },
    }
    return {
        "status": "valid",
        "phase": phase,
        "tier": tier,
        "pair_id": pair_id,
        "mode": mode,
        "scenario": inputs["scenario"],
        "seed": inputs["seed"],
        "input_sha256": inputs_hash,
        "config_sha256": config_hash,
        "client_sha256": sha256(directory / "client.json"),
        "engine_sha256": sha256(directory / "engine.json"),
        "metrics": metrics,
    }


def comparable(result):
    metrics = result["metrics"]
    values = {
        "request_throughput": metrics["throughput"]["requests_per_s"],
        "token_throughput": metrics["throughput"]["tokens_per_s"],
        "schedule_p95_ns": metrics["schedule_ns"]["p95"],
        **metrics["work"],
    }
    for stratum, report in metrics["strata"].items():
        for key in ("completion_rate", "timely_first_token_rate"):
            values[f"{stratum}.{key}"] = report[key]
        for key in (
            "ttft_ns",
            "submit_ttft_ns",
            "tpot_ns",
            "enqueue_to_first_dispatch_ns",
            "whole_queue_overtakes",
            "window_overtakes",
            "whole_queue_hot_overtakes",
            "window_hot_overtakes",
        ):
            values[f"{stratum}.{key}.p95"] = report[key]["p95"]
    return values


def difference(a, b):
    return {key: None if a[key] is None or b[key] is None else a[key] - b[key] for key in a}


def paired_summary(rounds):
    result = {}
    for key in rounds[0]:
        observed = [row[key] for row in rounds if row[key] is not None]
        result[key] = {
            "n": len(observed),
            "median": median(observed) if observed else None,
            "min": min(observed, default=None),
            "max": max(observed, default=None),
            "positive": sum(v > 0 for v in observed),
            "negative": sum(v < 0 for v in observed),
            "equal": sum(v == 0 for v in observed),
        }
    return result


def audit_matrix(config_path, index):
    config = require_dict(read_json(config_path), "M5 config")
    pairs = config_matrix(config)
    catalog = require_list(require_dict(index, "matrix index").get("runs"), "matrix runs")
    expected = [(pair_id, mode) for pair_id, pair in pairs.items() for mode in pair["order"]]
    actual = [(row.get("pair_id"), row.get("mode")) for row in catalog]
    require(
        actual == expected, "formal matrix missing, duplicate, mislabeled or out of frozen order"
    )
    require(len({row.get("run") for row in catalog}) == len(catalog), "formal run directory reused")
    groups = defaultdict(dict)
    for row in catalog:
        pair_id, mode = row["pair_id"], row["mode"]
        groups[pair_id][mode] = audit_run(
            config_path, Path(row["inputs"]), Path(row["run"]), pair_id=pair_id, mode=mode
        )
    results = []
    by_scenario = defaultdict(lambda: defaultdict(list))
    for pair_id, rounds in groups.items():
        require(
            len({run["input_sha256"] for run in rounds.values()}) == 1,
            "pair arrival/input table differs",
        )
        require(
            len({run["metrics"]["kv_blocks"] for run in rounds.values()}) == 1,
            "pair physical KV capacity differs",
        )
        scenario = pairs[pair_id]["scenario"]
        gates = config["gates"]
        pressure = rounds["native"]["metrics"]["pressure"]
        if scenario in {"burst", "mixed"}:
            require(
                pressure["competing_windows"] >= gates["min_competing_windows"]
                and pressure["reusable_evictions"] >= gates["min_reusable_evictions"],
                "native key-scenario KV pressure or competing windows below frozen gate",
            )
        elif scenario == "ample":
            require(
                all(run["metrics"]["pressure"]["cached_evictions"] == 0 for run in rounds.values()),
                "ample control experienced KV eviction",
            )
        elif scenario == "all-cold":
            require(
                all(
                    run["metrics"]["strata"]["all"]["cached_tokens"]["max"] == 0
                    for run in rounds.values()
                ),
                "all-cold control has observed cache hits/missing hits",
            )
        values = {mode: comparable(run) for mode, run in rounds.items()}
        contrasts = {
            f"{mode}-native": difference(values[mode], values["native"])
            for mode in ("off", "prefix", "guarded")
        }
        contrasts["guarded-prefix"] = difference(values["guarded"], values["prefix"])
        for name, delta in contrasts.items():
            by_scenario[scenario][name].append(delta)
        results.append(
            {"pair_id": pair_id, "scenario": scenario, "runs": rounds, "contrasts": contrasts}
        )
    return {
        "status": "valid",
        "pairs": results,
        "scenarios": {
            scenario: {name: paired_summary(deltas) for name, deltas in contrasts.items()}
            for scenario, contrasts in by_scenario.items()
        },
    }
