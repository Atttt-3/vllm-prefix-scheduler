"""M4 bounded native metadata fixtures; model outputs alone are synthetic.

Native hashing, queues, allocation, preemption and update_from_output are real.
Only explicitly named barriers/fault cases modify native fields or inject errors.
"""

import gc
import json
import os
import random
import time
import weakref
from collections import deque
from pathlib import Path

import m2a_harness as historical

from vllm_prefix_scheduler.m4_observer import WeakIdentityRegistry, guard_state, request_state


def require(value, message):
    if not value:
        raise RuntimeError(message)


def native_config(config, seed):
    baseline = json.loads((Path(__file__).parents[2] / "configs/native-m2b.json").read_text())
    return {
        **baseline,
        "seed": seed,
        "window": config["window"],
        "max_wait_ms": config["max_wait_ms"],
        "max_steps_per_phase": config["native_matrix"]["max_steps_per_phase"],
    }


class Driver:
    def __init__(self, config, scheduler, emit):
        self.config, self.scheduler, self.emit = config, scheduler, emit
        self.registry = WeakIdentityRegistry()
        self.pending = deque()
        self.sequence = 0
        self.patches = []
        self.refs = []
        self.max_pending = 0
        self.max_live = 0
        self.install(scheduler, "_preempt_request", self.preempt)
        self.install(scheduler, "_free_request", self.terminal)
        if hasattr(scheduler, "_candidate_window"):
            self.install(scheduler, "_candidate_window", self.candidates)
        self.event("m4_native_identity", **historical.source_record(scheduler))

    def event(self, event, **fields):
        self.emit({"event": event, **fields})

    def states(self):
        return [
            request_state(r, self.registry, self.scheduler)
            for r in self.scheduler.requests.values()
        ]

    def pending_rows(self):
        return [row[3] for row in self.pending]

    def resources(self):
        s = self.scheduler
        pool = s.kv_cache_manager.block_pool
        q = pool.free_block_queue
        block = q.fake_free_list_head.next_free_block
        free, seen = [], set()
        while block is not q.fake_free_list_tail:
            require(block is not None and id(block) not in seen, "corrupt native free queue")
            seen.add(id(block))
            free.append(block.block_id)
            block = block.next_free_block
        guard = getattr(s, "_wait_guard", None)
        return {
            "requests": sorted(s.requests),
            "waiting": [r.request_id for r in s.waiting],
            "running": [r.request_id for r in s.running],
            "holding": [r.request_id for r in s.kv_holding_waiting],
            "deferred": sorted(r.request_id for r in s.deferred_waiting),
            "finished_req_ids": sorted(s.finished_req_ids),
            "reset_preempted_req_ids": sorted(s.reset_preempted_req_ids),
            "inflight_prefills": sorted(r.request_id for r in s._inflight_prefills),
            "deferred_frees": len(s.deferred_frees),
            "request_blocks": sum(
                len(m.req_to_blocks) for m in s.kv_cache_manager.coordinator.single_type_managers
            ),
            "non_null_ref_sum": sum(b.ref_cnt for b in pool.blocks if not b.is_null),
            "free_count": q.num_free_blocks,
            "free_unique_count": len(set(free)),
            "free_traversed_count": len(free),
            "total_non_null_blocks": sum(not b.is_null for b in pool.blocks),
            "cached_block_count": sum(b._block_hash is not None for b in pool.blocks),
            "pending_batches": [r[3]["batch_id"] for r in self.pending],
            "inflight_tokens": sum(r.num_in_flight_tokens for r in s.requests.values()),
            "stale_tokens": sum(r.num_stale_output_tokens for r in s.requests.values()),
            "output_placeholders": sum(r.num_output_placeholders for r in s.requests.values()),
            "guard_records": guard_state(s, self.registry),
            "guard_closed": None if guard is None else guard.closed,
            "guard_fault_code": None if guard is None else guard.fault_code,
            "observation_errors": [],
        }

    def assert_drained(self, allow_notices=False):
        state = self.resources()
        keys = (
            "requests",
            "waiting",
            "running",
            "holding",
            "deferred",
            "inflight_prefills",
            "deferred_frees",
            "request_blocks",
            "non_null_ref_sum",
            "pending_batches",
            "inflight_tokens",
            "stale_tokens",
            "output_placeholders",
            "guard_records",
        )
        if not allow_notices:
            keys += ("finished_req_ids", "reset_preempted_req_ids")
        require(all(not state[k] for k in keys), f"native resource residue: {state}")
        require(
            len(
                {
                    state[k]
                    for k in (
                        "free_count",
                        "free_unique_count",
                        "free_traversed_count",
                        "total_non_null_blocks",
                    )
                }
            )
            == 1,
            "free queue did not recover",
        )
        return state

    def install(self, obj, name, factory):
        own = name in getattr(obj, "__dict__", {})
        original = getattr(obj, name)
        self.patches.append((obj, name, original, own))
        setattr(obj, name, factory(original))

    def preempt(self, original):
        def wrapped(request, *args, **kwargs):
            before = request_state(request, self.registry, self.scheduler)
            self.event("m4_preempt_start", request=before, pending=self.pending_rows())
            result = original(request, *args, **kwargs)
            self.event(
                "m4_preempt_success",
                before=before,
                request=request_state(request, self.registry, self.scheduler),
                pending=self.pending_rows(),
            )
            return result

        return wrapped

    def terminal(self, original):
        def wrapped(request, *args, **kwargs):
            result = original(request, *args, **kwargs)
            self.event(
                "m4_request_terminal", request=request_state(request, self.registry, self.scheduler)
            )
            return result

        return wrapped

    def candidates(self, original):
        def wrapped():
            result = original()
            self.event(
                "m4_window",
                candidates=[request_state(r, self.registry, self.scheduler) for r in result],
                waiting=[
                    request_state(r, self.registry, self.scheduler) for r in self.scheduler.waiting
                ],
            )
            return result

        return wrapped

    def add(self, rid, tokens, max_tokens=3):
        request = historical.request({**self.config, "max_tokens": max_tokens}, rid, tokens)
        self.scheduler.add_request(request)
        self.refs.append(weakref.ref(request))
        self.max_live = max(self.max_live, len(self.scheduler.requests))
        self.event(
            "m4_request_bound",
            bound=self.scheduler.requests.get(rid) is request,
            request={
                **request_state(request, self.registry, self.scheduler),
                "prompt_token_ids": list(request.prompt_token_ids),
            },
        )
        return request

    def schedule(self):
        from vllm.v1.outputs import ModelRunnerOutput

        require(len(self.pending) < 2, "pending FIFO depth exceeded")
        before = guard_state(self.scheduler, self.registry)
        output = self.scheduler.schedule()
        self.sequence += 1
        batch_id = str(self.sequence)
        ids = list(output.num_scheduled_tokens)
        intervals = []
        for rid, count in output.num_scheduled_tokens.items():
            r = self.scheduler.requests[rid]
            intervals.append(
                {
                    "request_id": rid,
                    **self.registry.binding(r),
                    "start": r.num_computed_tokens - count,
                    "end": r.num_computed_tokens,
                    "prompt_length": r.num_prompt_tokens,
                }
            )
        synthetic = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[
                [] if self.scheduler.requests[rid].is_prefill_chunk else [9001] for rid in ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        row = {
            "batch_id": batch_id,
            "intervals": intervals,
            "finished_req_ids": sorted(output.finished_req_ids),
            "preempted_req_ids": sorted(output.preempted_req_ids or ()),
        }
        self.pending.append((output, synthetic, batch_id, row))
        self.max_pending = max(self.max_pending, len(self.pending))
        self.event(
            "m4_schedule",
            **row,
            num_scheduled_tokens=dict(output.num_scheduled_tokens),
            states=self.states(),
            guard_before=before,
            guard_after=guard_state(self.scheduler, self.registry),
        )
        return output

    def consume(self):
        output, synthetic, batch_id, row = self.pending.popleft()
        self.event(
            "m4_consume_start", batch_id=batch_id, intervals=row["intervals"], states=self.states()
        )
        result = self.scheduler.update_from_output(output, synthetic)
        self.event(
            "m4_consume_success",
            batch_id=batch_id,
            intervals=row["intervals"],
            states=self.states(),
            outputs=[
                {
                    "request_id": o.request_id,
                    "new_token_ids": list(o.new_token_ids),
                    "finish_reason": None if o.finish_reason is None else o.finish_reason.name,
                }
                for group in result.values()
                for o in group.outputs
            ],
        )

    def cancel(self, ids):
        from vllm.v1.request import RequestStatus

        self.event(
            "m4_cancel_start",
            requests=self.states(),
            requested_ids=ids,
            pending=self.pending_rows(),
        )
        result = self.scheduler.finish_requests(ids, RequestStatus.FINISHED_ABORTED)
        self.event(
            "m4_cancel_success",
            requests=[request_state(r, self.registry, self.scheduler) for r in result],
            guard_after=guard_state(self.scheduler, self.registry),
            pending=self.pending_rows(),
        )

    def drain(self, label):
        for _ in range(self.config["max_steps_per_phase"]):
            if not self.scheduler.has_requests() and not self.pending:
                resources = self.assert_drained()
                self.event("m4_generation_drained", label=label, resources=resources)
                return resources
            if self.scheduler.has_requests() and len(self.pending) < 2:
                output = self.schedule()
                if output.total_num_scheduled_tokens and len(self.pending) < 2:
                    continue
            if self.pending:
                self.consume()
        raise RuntimeError(f"native phase step limit: {label}")

    def close(self):
        try:
            self.event("m4_shutdown_before", resources=self.resources())
            self.scheduler.shutdown()
            self.event("m4_shutdown_after", resources=self.resources())
        finally:
            for obj, name, original, own in reversed(self.patches):
                if own:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)


def _long_cycle(driver, config, case):
    rng = random.Random(case["seed"])
    for round_index in range(config["native_rounds_per_seed"]):
        driver.event("m4_round_begin", round=round_index, resources=driver.assert_drained())
        base = rng.randrange(1000, 100000)
        driver.add("complete", range(base, base + 33), max_tokens=2)
        driver.drain("complete")
        driver.add("reused", range(base + 100, base + 133))
        driver.cancel("reused")
        driver.cancel(["reused", "unknown", "unknown"])
        driver.drain("waiting_cancel")
        driver.add("reused", range(base + 200, base + 265))
        driver.schedule()
        request = driver.scheduler.requests["reused"]
        require(
            request.is_prefill_chunk and request.num_in_flight_tokens > 0,
            "missing real partial pending cancel",
        )
        del request
        driver.cancel("reused")
        driver.drain("pending_cancel")
        driver.add("reused", range(base + 300, base + 333), max_tokens=2)
        driver.drain("reuse_complete")
        refs = driver.refs
        driver.refs = []
        gc.collect()
        alive = sum(ref() is not None for ref in refs)
        require(alive == 0, "historical native Request retained after fixture ledger release")
        require(
            len(driver.registry) == 0 and not driver.pending, "observer metadata grew with history"
        )
        driver.event(
            "m4_round_end",
            round=round_index,
            resources=driver.assert_drained(),
            weak_refs_alive=alive,
            observer_cardinalities={
                "weak_registry": len(driver.registry),
                "pending": len(driver.pending),
                "strong_request_ledger": 0,
                "max_pending": driver.max_pending,
                "max_live": driver.max_live,
            },
        )


def _async_case(driver, variant):
    if variant == "chunk_barrier":
        r = driver.add("chunk", range(1000, 1065))
        driver.schedule()
        require(r.is_prefill_chunk and r.num_in_flight_tokens > 0, "actual partial prefill missing")
        driver.add("tail", range(2000, 2033))
        driver.drain("chunk_tail")
    else:
        first = driver.add("R0", range(1000, 1032))
        second = driver.add("R1", range(2000, 2032))
        driver.schedule()
        require(len(driver.scheduler.running) == 2, "pressure did not jointly admit requests")
        driver.schedule()
        require(
            second.num_preemptions > 0 and second.num_stale_output_tokens > 0,
            "native capacity preemption with pending stale share missing",
        )
        driver.consume()
        driver.add("tail", [*range(1000, 1032), 3000])
        driver.drain("preempt_resume")
        require(first.is_finished() and second.is_finished(), "capacity requests did not finish")


def _barrier_case(driver, variant):
    from vllm.v1.request import RequestStatus

    s = driver.scheduler
    if variant == "prepend_failure":
        driver.add("warm", range(1000, 1033), max_tokens=1)
        driver.drain("warm")
        driver.add("cold", range(2000, 2033))
        driver.add("hot", [*range(1000, 1032), 9999])
    else:
        count = 18 if variant == "window_tail_age" else 3
        for index in range(count):
            driver.add(f"q{index}", range(2000 + index * 100, 2033 + index * 100))
    before = [request_state(r, driver.registry, s) for r in s.waiting]
    guard_before = guard_state(s, driver.registry)
    old = None
    mutated = None
    original_prepend = None
    failure = None
    if variant in {"blocked_head", "blocked_middle"}:
        mutated = list(s.waiting)[0 if variant == "blocked_head" else 1]
        old = mutated.status
        mutated.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        s.deferred_waiting.add(mutated)
    fields = {
        "partial_prefill": ("num_computed_tokens", 1),
        "in_flight": ("num_in_flight_tokens", 1),
        "placeholder": ("num_output_placeholders", 1),
        "stale": ("num_stale_output_tokens", 1),
        "preempted_zero": ("num_preemptions", 1),
    }
    if variant in fields:
        mutated = list(s.waiting)[1]
        field, changed = fields[variant]
        old = getattr(mutated, field)
        setattr(mutated, field, changed)
    observed_before = [request_state(r, driver.registry, s) for r in s.waiting]
    candidates = [request_state(r, driver.registry, s) for r in s._candidate_window()]
    if variant == "prepend_failure":
        original_prepend = s.waiting.prepend_requests

        def failed_prepend(items):
            iterator = iter(items)
            s.waiting.prepend_request(next(iterator))
            raise RuntimeError("M4 injected prepend after partial mutation")

        s.waiting.prepend_requests = failed_prepend
    try:
        s._reorder_waiting()
    except RuntimeError as error:
        if variant != "prepend_failure":
            raise
        failure = str(error)
    finally:
        if original_prepend is not None:
            del s.waiting.prepend_requests
    after = [request_state(r, driver.registry, s) for r in s.waiting]
    require(
        [r["object_id"] for r in before] == [r["object_id"] for r in after],
        "barrier/tie/restoration order changed",
    )
    if variant == "prepend_failure":
        require(failure is not None, "prepend failure not triggered")
    driver.event(
        "m4_barrier_check",
        variant=variant,
        injected=variant not in {"window_tail_age", "stable_ties"},
        before=observed_before,
        after=after,
        candidates=candidates,
        guard_before=guard_before,
        guard_after=guard_state(s, driver.registry),
        restored=True,
        exception=failure,
    )
    if mutated is not None:
        if variant in {"blocked_head", "blocked_middle"}:
            mutated.status = old
            s.deferred_waiting.discard(mutated)
        else:
            setattr(mutated, fields[variant][0], old)
    if variant == "window_tail_age":
        # Remove the current first window by native terminal handling; tail ages must be unchanged.
        driver.cancel([f"q{i}" for i in range(16)])
        tail_before = guard_state(s, driver.registry)
        s._reorder_waiting()
        require(
            tail_before == guard_state(s, driver.registry), "outside-window age refreshed on entry"
        )
        driver.event(
            "m4_tail_entered",
            records_before=tail_before,
            records_after=guard_state(s, driver.registry),
        )
    driver.cancel(None)
    driver.drain("barrier_cleanup")


def _terminal_case(driver, variant):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    s = driver.scheduler
    original = None
    target = None
    if variant == "shutdown_closed":
        s.shutdown()
    else:
        target, name = (
            (AsyncScheduler, "schedule")
            if variant == "native_schedule_fault"
            else (Scheduler, "_free_request")
        )
        original = getattr(target, name)

        def failure(*args, **kwargs):
            raise RuntimeError("M4 injected native failure")

        setattr(target, name, failure)
        try:
            try:
                if variant == "native_schedule_fault":
                    s.schedule()
                else:
                    # A terminal object with no native allocation tests the error latch;
                    # it is explicitly a native branch fixture, not a real cancellation.
                    r = historical.request(driver.config, "fault_object", [1000])
                    from vllm.v1.request import RequestStatus

                    r.status = RequestStatus.FINISHED_ABORTED
                    s._free_request(r)
            except RuntimeError:
                pass
            else:
                raise RuntimeError("injected fault did not occur")
        finally:
            setattr(target, name, original)
    rejected = []
    for label, action in (
        ("schedule", s.schedule),
        ("add", lambda: s.add_request(historical.request(driver.config, "rejected", [1]))),
    ):
        try:
            action()
        except RuntimeError:
            rejected.append(label)
    require(rejected == ["schedule", "add"], "fault/closed retries not rejected")
    resources = driver.assert_drained()
    driver.event(
        "m4_terminal_check",
        variant=variant,
        injected=True,
        retry_rejected=rejected,
        fault_code=s._wait_guard.fault_code,
        closed=s._wait_guard.closed,
        resources=resources,
    )


def run_case(config, case, model_path, emit):
    """Execute one frozen leaf; raises on invalid precondition or any leak."""
    require(__debug__, "M4 native refuses optimized interpreter")
    case = dict(case)
    case.setdefault("repeat", config["seeds"].index(case["seed"]))
    local = native_config(config, case["seed"])
    scenario, variant = case["scenario"], case["variant"]
    capacity = 5 if variant == "capacity_preempt_resume" else 10
    budget = 64 if variant == "capacity_preempt_resume" else 16
    from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

    clock_reads = [0]
    original_clock = PrefixAsyncScheduler._wait_clock_ns

    def forbidden_clock(instance):
        clock_reads[0] += 1
        raise RuntimeError("off/prefix read the protection clock")

    if case["mode"] in {"off", "prefix"}:
        PrefixAsyncScheduler._wait_clock_ns = forbidden_clock
    try:
        scheduler = historical.create_scheduler(
            local, case["mode"], model_path, capacity=capacity, budget=budget, active=2
        )
    finally:
        PrefixAsyncScheduler._wait_clock_ns = original_clock
    sequence = 0
    case_id = f"{scenario}:{variant}:{case['mode']}:{case['repeat']}"

    def send(row):
        nonlocal sequence
        sequence += 1
        emit(
            {
                "schema": 1,
                "task_id": "M4-scheduler-stability",
                "layer": "native",
                "case_id": case_id,
                "engine_id": case_id,
                "pid": os.getpid(),
                "clock_domain": "native-process-monotonic",
                "seq": sequence,
                "monotonic_ns": time.monotonic_ns(),
                **case,
                **row,
            }
        )

    if case["mode"] in {"native", "off", "prefix"}:
        require(
            not hasattr(scheduler, "_wait_guard"), "off/prefix unexpectedly created guard metadata"
        )
    driver = Driver(local, scheduler, send)
    if case["mode"] in {"off", "prefix"}:
        driver.install(
            scheduler, "_wait_clock_ns", lambda original: lambda: forbidden_clock(scheduler)
        )
    driver.event(
        "m4_case_begin",
        native_model_outputs="synthetic",
        resources=driver.assert_drained(),
        initial_resources=driver.assert_drained(),
        interventions=["synthetic_model_runner_output"]
        + (
            ["native_branch_field_or_fault_injection"]
            if scenario in {"barriers", "terminal_contract"}
            else []
        ),
    )
    try:
        driver.event("m4_generation_drained", label="initial", resources=driver.assert_drained())
        if scenario == "long_cycle":
            _long_cycle(driver, config, case)
        elif scenario == "async_interleave":
            _async_case(driver, variant)
        elif scenario == "barriers":
            _barrier_case(driver, variant)
        elif scenario == "terminal_contract":
            _terminal_case(driver, variant)
        else:
            raise ValueError(f"unknown frozen native scenario {scenario}")
        final = driver.assert_drained()
        driver.event("m4_generation_drained", label="final", resources=final)
        driver.event(
            "m4_case_end",
            resources=final,
            status="passed",
            guard_present=hasattr(scheduler, "_wait_guard"),
            protection_clock_reads=clock_reads[0]
            if case["mode"] in {"native", "off", "prefix"}
            else None,
        )
        return {"case_id": case_id, "resources": final}
    finally:
        driver.close()
