"""Pinned native M2B replay with controlled *private* protection time.

Native allocation, hashing, async accounting, output updates and termination are
never replaced. ModelRunnerOutput is synthetic, as in the accepted M1/M2A layer.
Branch fault injection is explicitly named and separated from natural lifetimes.
"""

import copy
import functools
import json
from pathlib import Path

import m2a_harness as historical
from m2b_snapshot import ObjectRegistry, guard_snapshot, snapshot

from vllm_prefix_scheduler.m2b_contract import expected_cases, require, require_assertions


class ControlledClock:
    def __init__(self, value=0):
        self.value = value
        self.error = None
        self.purpose = None
        self.reads = 0
        self.emit = None

    def __call__(self):
        self.reads += 1
        event = {
            "event": "clock",
            "read_index": self.reads,
            "purpose": self.purpose,
            "value": self.value,
            "error": self.error,
        }
        if self.emit is not None:
            self.emit(event)
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.value


def create_scheduler(config, mode, model_path, clock, **kwargs):
    from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

    original = PrefixAsyncScheduler._wait_clock_ns
    PrefixAsyncScheduler._wait_clock_ns = lambda self: clock()
    try:
        return historical.create_scheduler(config, mode, model_path, **kwargs)
    finally:
        PrefixAsyncScheduler._wait_clock_ns = original


class Driver(historical.Driver):
    def __init__(self, config, scheduler, emit, clock):
        self.registry = ObjectRegistry()
        self.clock = clock
        self.bound_objects = {}
        self.shutdown_called = False
        self.pending_intervals = []
        super().__init__(config, scheduler, emit)
        clock.emit = lambda event: self.event(event.pop("event"), **event)
        from vllm.v1.core.sched.scheduler import Scheduler

        from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

        self.install(Scheduler, "add_request", self.add_native_wrapper)
        self.install(Scheduler, "shutdown", self.shutdown_native_wrapper)
        # Also catch a forbidden off/prefix call made after constructor return.
        self.install(
            PrefixAsyncScheduler, "_wait_clock_ns", lambda original: lambda instance: clock()
        )
        self.wrap_native(scheduler, "_free_request")
        guard = getattr(scheduler, "_wait_guard", None)
        if guard is not None:
            for name in ("register", "discard", "close", "fail", "read_clock"):
                self.install(
                    guard, name, lambda original, name=name: self.guard_wrapper(name, original)
                )
        self.event("fixture_initial", state=self.state(full=True))

    def event(self, event, **data):
        self.emit(
            {
                "event": event,
                "step": self.step,
                "guard": guard_snapshot(self.scheduler, self.registry),
                **data,
            }
        )

    def state(self, full=False):
        return snapshot(self.scheduler, self.requests, self.generations, self.registry, full)

    def super_wrapper(self, original):
        driver = self

        @functools.wraps(original)
        def wrapped(instance, *args, **kwargs):
            if instance is not driver.scheduler:
                return original(instance, *args, **kwargs)
            driver.counts()["super"] += 1
            driver.event(
                "native_schedule_enter", state=driver.state(), counts=driver.counts().copy()
            )
            try:
                result = original(instance, *args, **kwargs)
            except BaseException as error:
                driver.event(
                    "native_schedule_raise",
                    error_type=type(error).__name__,
                    error=str(error),
                    state=driver.state(),
                )
                raise
            driver.event(
                "native_schedule_return",
                scheduled=result.num_scheduled_tokens,
                objects={
                    rid: driver.registry.token(instance.requests[rid])
                    for rid in result.num_scheduled_tokens
                },
                state=driver.state(),
            )
            intervals = []
            for raw in driver.pending_intervals:
                interval = {k: v for k, v in raw.items() if k not in {"start", "count"}}
                interval.update(
                    driver.ledger.record(
                        f"{raw['request_id']}@{raw['generation']}",
                        raw["start"],
                        raw["count"],
                        raw["prompt_length"],
                    )
                )
                intervals.append(interval)
                driver.dispatches.append(interval)
            driver.event("dispatch", intervals=intervals, state=driver.state())
            driver.pending_intervals = []
            return result

        return wrapped

    def advance_wrapper(self, original):
        def wrapped(output):
            intervals = []
            for rid, count in output.num_scheduled_tokens.items():
                request = self.scheduler.requests[rid]
                generation = self.generations[rid]
                interval = {
                    "request_id": rid,
                    "generation": generation,
                    "object": self.registry.token(request),
                    "prompt_length": request.num_prompt_tokens,
                    "step": self.step,
                    "preemptions": request.num_preemptions,
                    "start": request.num_computed_tokens,
                    "count": count,
                }
                intervals.append(interval)
            self.pending_intervals = intervals
            self.event("schedule_update", intervals=intervals, state=self.state())
            result = original(output)
            self.event("advance", state=self.state())
            return result

        return wrapped

    def at(self, tick):
        self.clock.value = tick
        self.event("clock_set", value=tick)

    def shutdown_native_wrapper(self, original):
        driver = self

        @functools.wraps(original)
        def wrapped(instance, *args, **kwargs):
            if instance is not driver.scheduler:
                return original(instance, *args, **kwargs)
            driver.event("native_shutdown_call", state=driver.state())
            try:
                result = original(instance, *args, **kwargs)
            except BaseException as error:
                driver.event(
                    "native_shutdown_raise",
                    error_type=type(error).__name__,
                    error=str(error),
                    state=driver.state(),
                )
                raise
            driver.event("native_shutdown_return", state=driver.state())
            return result

        return wrapped

    def schedule(self, phase="main"):
        try:
            return super().schedule(phase)
        except BaseException as error:
            self.event(
                "schedule_raise",
                phase=phase,
                error_type=type(error).__name__,
                error=str(error),
                state=self.state(full=True),
            )
            raise

    def window_wrapper(self, original):
        def wrapped():
            self.counts()["window_calls"] += 1
            before = [r.request_id for r in self.scheduler.waiting]
            self.event("window_before", waiting=before, state=self.state(full=True))
            result = original()
            after = [r.request_id for r in self.scheduler.waiting]
            self.counts()["reorders"] += int(before != after)
            row = {
                "before": before,
                "after": after,
                "counts": self.counts().copy(),
                "candidates": self.step_candidates.get(self.step, []),
                "scores": self.step_scores.get(self.step, []),
            }
            self.windows.append(row)
            self.event("window_after", **row, state=self.state(full=True))
            return result

        return wrapped

    def guard_wrapper(self, name, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            request = args[0] if name in {"register", "discard"} else None
            details = {"operation": name}
            if request is not None:
                details.update(object=self.registry.token(request), request_id=request.request_id)
            if name == "discard":
                details["reason"] = args[1] if len(args) > 1 else kwargs.get("reason")
            if name == "read_clock":
                self.clock.purpose = args[0] if args else kwargs.get("reason", "enqueue")
                details["reason"] = self.clock.purpose
            if name == "fail":
                details["code"] = args[0]
            self.event("guard_before", **details)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                self.event(
                    "guard_raise", **details, error_type=type(error).__name__, error=str(error)
                )
                raise
            self.event("guard_after", **details, result=result if type(result) is bool else None)
            return result

        return wrapped

    def bind(self, request):
        rid = request.request_id
        token = self.registry.token(request)
        if self.bound_objects.get(rid) is request:
            return self.generations[rid]
        generation = self.generations.get(rid, -1) + 1
        self.generations[rid] = generation
        key = rid if generation == 0 else f"{rid}@{generation}"
        self.requests[key] = request
        self.current_keys[rid] = key
        self.bound_objects[rid] = request
        self.event("binding", request_id=rid, object=token, generation=generation)
        return generation

    def add_native_wrapper(self, original):
        driver = self

        @functools.wraps(original)
        def wrapped(instance, request, *args, **kwargs):
            if instance is not driver.scheduler:
                return original(instance, request, *args, **kwargs)
            before = instance.requests.get(request.request_id)
            driver.event(
                "native_add_call",
                request_id=request.request_id,
                object=driver.registry.token(request),
                previous_object=driver.registry.token(before) if before is not None else None,
            )
            try:
                result = original(instance, request, *args, **kwargs)
            except BaseException as error:
                driver.event(
                    "native_add_raise",
                    request_id=request.request_id,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise
            actual = instance.requests.get(request.request_id)
            generation = driver.bind(actual) if actual is not None else None
            driver.event(
                "native_add_return",
                request_id=request.request_id,
                object=driver.registry.token(request),
                actual_object=driver.registry.token(actual) if actual is not None else None,
                previous_object=driver.registry.token(before) if before is not None else None,
                generation=generation,
                actual_fresh=driver.scheduler._fresh_request(actual)
                if actual is not None and hasattr(driver.scheduler, "_fresh_request")
                else None,
                actual_status=actual.status.name if actual is not None else None,
                actual_class=f"{type(actual).__module__}.{type(actual).__qualname__}"
                if actual is not None
                else None,
                state=driver.state(full=True),
            )
            return result

        return wrapped

    def add(self, request):
        rid = request.request_id
        self.event(
            "enqueue_call",
            request_id=rid,
            object=self.registry.token(request),
            arrival_time=request.arrival_time,
            tokens=list(request.prompt_token_ids or ()),
        )
        try:
            result = self.scheduler.add_request(request)
        except BaseException as error:
            self.event(
                "enqueue_raise",
                request_id=rid,
                object=self.registry.token(request),
                error_type=type(error).__name__,
                error=str(error),
                state=self.state(),
            )
            raise
        actual = self.scheduler.requests.get(rid)
        self.event(
            "enqueue",
            ids=[rid],
            generation=self.generations.get(rid),
            object=self.registry.token(request),
            actual_object=self.registry.token(actual) if actual is not None else None,
            arrivals={rid: request.arrival_time},
            tokens=list(request.prompt_token_ids or ()),
            state=self.state(),
        )
        return result

    def wrap_native(self, obj, name):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                old_owner = self.owner
                request = (
                    args[0]
                    if name
                    in {
                        "get_computed_blocks",
                        "allocate_slots",
                        "cache_blocks",
                        "free",
                        "_preempt_request",
                        "_free_request",
                    }
                    else None
                )
                if request is not None:
                    self.owner = request.request_id
                self.call_id += 1
                call = self.call_id
                details = {}
                if request is not None:
                    details["object"] = self.registry.token(request)
                if name == "_maybe_evict_cached_block":
                    details["block_id"] = args[0].block_id
                if name == "get_new_blocks":
                    details["num_blocks"] = args[0]
                if name == "allocate_slots":
                    details.update(
                        num_new_tokens=kwargs.get(
                            "num_new_tokens", args[1] if len(args) > 1 else None
                        ),
                        num_new_computed_tokens=kwargs.get("num_new_computed_tokens", 0),
                    )
                self.event(
                    "call",
                    method=name,
                    call=call,
                    owner=self.owner,
                    details=details,
                    state=self.state(),
                )
                try:
                    result = original(*args, **kwargs)
                    rendered = None
                    if name == "get_computed_blocks":
                        rendered = {"blocks": result[0].get_block_ids(), "hit_tokens": result[1]}
                    elif name == "allocate_slots":
                        rendered = {
                            "success": result is not None,
                            "blocks": result.get_block_ids() if result is not None else None,
                        }
                    elif name == "get_new_blocks":
                        rendered = [b.block_id for b in result]
                    elif name == "_maybe_evict_cached_block":
                        rendered = result
                    self.event(
                        "return",
                        method=name,
                        call=call,
                        owner=self.owner,
                        result=rendered,
                        state=self.state(),
                    )
                    return result
                except BaseException as error:
                    self.event(
                        "raise",
                        method=name,
                        call=call,
                        owner=self.owner,
                        error_type=type(error).__name__,
                        error=str(error),
                        state=self.state(),
                    )
                    raise
                finally:
                    self.owner = old_owner

            return wrapped

        self.install(obj, name, factory)

    def sort_wrapper(self, original):
        def wrapped(candidates):
            self.counts()["sorts"] += 1
            self.event(
                "decision_before",
                objects=[self.registry.token(r) for r in candidates],
                request_ids=[r.request_id for r in candidates],
                state=self.state(full=True),
            )
            result = original(candidates)
            self.event(
                "decision_after",
                objects=[self.registry.token(r) for r in result],
                request_ids=[r.request_id for r in result],
                state=self.state(),
            )
            return result

        return wrapped

    def finish(self):
        guard = getattr(self.scheduler, "_wait_guard", None)
        active = guard is None or (not guard.closed and guard.fault_code is None)
        if (
            active
            and not self.scheduler.requests
            and not self.pending
            and (self.scheduler.finished_req_ids or self.scheduler.reset_preempted_req_ids)
        ):
            # Deliver terminal notices through native output; closed/faulted
            # instances must never bypass their latch just to empty these IDs.
            self.schedule("terminal_notifications")
            self.consume()
        result = super().finish()
        require(
            guard is None or not guard.records, "protection records survived complete native drain"
        )
        require(
            not self.scheduler._inflight_prefills, "native in-flight prefill set retained objects"
        )
        require(not self.scheduler.deferred_frees, "native deferred KV frees did not drain")
        require(
            all(not group for group in result["state"]["request_blocks"]),
            "native request block table retained historical IDs",
        )
        notices = {
            "finished_request_ids": sorted(self.scheduler.finished_req_ids),
            "reset_preempted_request_ids": sorted(self.scheduler.reset_preempted_req_ids),
            "inflight_prefills": len(self.scheduler._inflight_prefills),
            "deferred_frees": len(self.scheduler.deferred_frees),
        }
        if active:
            require(
                not notices["finished_request_ids"] and not notices["reset_preempted_request_ids"],
                "active native terminal notices did not drain",
            )
        result["native_auxiliary"] = notices
        self.event("native_auxiliary_state", active=active, **notices)
        self.event("fixture_drained", state=self.state(full=True))
        return result

    def close(self):
        try:
            if not self.shutdown_called:
                self.scheduler.shutdown()
            self.event("shutdown_after", state=self.state(full=True))
        finally:
            for obj, name, original, own in reversed(self.originals):
                if own:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
            self.clock.emit = None


def req(config, rid, tokens, *, max_tokens=None, **kwargs):
    local = config if max_tokens is None else {**config, "max_tokens": max_tokens}
    return historical.request(local, rid, tokens, **kwargs)


def expect_error(d, operation, label, *, expected=Exception):
    try:
        operation()
    except expected as error:
        d.event(
            "expected_error",
            name=label,
            error_type=type(error).__name__,
            error=str(error),
            state=d.state(full=True),
        )
        return error
    raise RuntimeError(f"Expected failure did not occur: {label}")


def verify_fault_retries(d, config, variant):
    from vllm_prefix_scheduler.wait_guard import WaitGuardError

    reads = d.clock.reads
    expect_error(
        d,
        lambda: d.schedule(f"{variant}_retry"),
        f"{variant}_retry_schedule",
        expected=WaitGuardError,
    )
    expect_error(
        d,
        lambda: d.add(req(config, "rejected_after_fault", [9000])),
        f"{variant}_retry_add",
        expected=WaitGuardError,
    )
    require(d.clock.reads == reads, "latched retries unexpectedly read protection clock")


def cancel_all(d):
    from vllm.v1.request import RequestStatus

    result = d.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    d.event("cancel_all", objects=[d.registry.token(r) for r in result], state=d.state(full=True))
    while d.pending:
        d.consume()


def bc_case(d, config, variant):
    historical._warm(d, config)
    requests = historical._abc(config)
    d.add(requests["B"])
    d.add(requests["C"])
    d.event("bc_enqueued", ids=["B", "C"], state=d.state(full=True))
    if variant == "protected":
        d.at(500_000_000)
    d.drain("BC")


def hot_flow_case(d, config):
    historical._warm(d, config)
    d.at(0)
    d.add(req(config, "B", range(300, 349), arrival=9000))
    for index, tick in enumerate((100_000_000, 200_000_000, 300_000_000, 500_000_000), 1):
        d.at(tick)
        d.add(req(config, f"H{index}", [*range(100, 132), 200 + index], arrival=-index))
        output = d.schedule(f"hot_round_{index}")
        require(
            len(output.num_scheduled_tokens) == 1,
            "hot flow must admit exactly one request per round",
        )
        d.event(
            "hot_round",
            round=index,
            tick=tick,
            scheduled=output.num_scheduled_tokens,
            state=d.state(full=True),
        )
        d.consume()
        d.event("hot_round_consumed", round=index, state=d.state(full=True))
    d.drain("hot_flow_end")


def allocation_retry_case(d, config):
    running = req(config, "R", range(1000, 1032), max_tokens=3)
    d.add(running)
    d.schedule("retry_running_start")
    d.consume()
    d.at(0)
    cold = req(config, "B", range(2000, 2049), max_tokens=1)
    d.add(cold)
    d.add(req(config, "C", range(3000, 3049), max_tokens=1))
    initial = guard_snapshot(d.scheduler, d.registry)
    for index, tick in enumerate((100_000_000, 500_000_000), 1):
        d.at(tick)
        output = d.schedule(f"allocation_failure_{index}")
        require(
            "B" not in output.num_scheduled_tokens, "resource-starved B unexpectedly dispatched"
        )
        require(running.num_preemptions == 0, "retry fixture unexpectedly preempted R")
        d.event("allocation_retry", attempt=index, first_guard=initial, state=d.state(full=True))
        d.consume()
    require(running.is_finished(), "R did not naturally release resources after its third output")
    d.at(500_000_001)
    output = d.schedule("retry_first_success")
    require(
        output.num_scheduled_tokens.get("B", 0) > 0,
        "B did not dispatch after R released its blocks",
    )
    d.drain("retry_end")


def threshold_case(d, config, variant):
    plan = config["clock_plans"][variant]
    d.at(plan["enqueue_ns"])
    if variant.startswith("fault_"):
        if variant == "fault_provider_add":
            d.clock.error = "injected private enqueue clock failure"
            expect_error(d, lambda: d.add(req(config, "B", range(300, 333))), variant)
        else:
            d.add(req(config, "B", range(300, 333)))
            d.add(req(config, "C", range(400, 433)))
            if variant == "fault_provider_decision":
                d.clock.error = "injected private decision clock failure"
            else:
                d.at(plan["invalid_value"])
            expect_error(d, lambda: d.schedule(variant), variant)
        require(d.scheduler._wait_guard.fault_code is not None, "clock fault did not latch")
        d.clock.error = None
        d.clock.value = plan["enqueue_ns"] + 1_000_000_000
        expect_error(d, lambda: d.schedule("fault_retry_schedule"), "fault_retry_schedule")
        expect_error(d, lambda: d.add(req(config, "blocked_after_fault", [42])), "fault_retry_add")
        cancel_all(d)
        return
    historical._warm(d, config)
    cold = req(config, "B", range(300, 349), arrival=9999)
    d.add(cold)
    d.at(plan["decision_ns"])
    d.add(req(config, "C", [*range(100, 132), 201], arrival=-9999))
    output = d.schedule("threshold_decision")
    d.event(
        "threshold_observation",
        plan=plan,
        scheduled=output.num_scheduled_tokens,
        state=d.state(full=True),
    )
    d.drain("threshold_end")


def protected_order_case(d, config, variant):
    historical._warm(d, config)
    running = req(config, "R", range(1000, 1032), max_tokens=3)
    d.add(running)
    d.schedule("order_running_start")
    d.consume()
    d.at(0)
    for index, (rid, tokens) in enumerate(
        (
            ("B", list(range(300, 349))),
            ("C", [*range(100, 132), 201]),
            ("D", [*range(100, 132), 202]),
        )
    ):
        if variant == "prior_reorder":
            d.at(index * 1_000_000)
        d.add(req(config, rid, tokens, arrival={"B": 5, "C": -1, "D": 0}[rid]))
    d.at(100_000_000)
    d.schedule("prefix_reordered_no_admission")
    require(
        [r.request_id for r in d.scheduler.waiting] == ["C", "D", "B"],
        "fixture did not establish prior prefix ordering",
    )
    d.at(502_000_000 if variant == "prior_reorder" else 500_000_000)
    d.schedule("protected_original_sequence")
    require(
        [r.request_id for r in d.scheduler.waiting] == ["B", "C", "D"],
        "protected order did not restore original enqueue sequence",
    )
    d.event("protected_sequence", variant=variant, state=d.state(full=True))
    d.drain("order_end")


def window_age_case(d, config):
    historical._warm(d, config)
    d.add(req(config, "R", range(1000, 1032), max_tokens=3))
    d.schedule("age_running_start")
    d.consume()
    d.at(0)
    for rid in ("head0", "head1", "B"):
        d.add(req(config, rid, range(2000 + len(rid) * 100, 2033 + len(rid) * 100)))
    d.at(100_000_000)
    d.schedule("B_outside_window")
    require("B" not in d.step_candidates[d.step], "B was not outside the candidate window")
    d.cancel("head0")
    d.cancel("head1")
    d.at(500_000_000)
    d.add(req(config, "H", [*range(100, 132), 201]))
    d.schedule("B_enters_old_age")
    require(
        [r.request_id for r in d.scheduler.waiting] == ["B", "H"],
        "window entrant did not keep original age",
    )
    d.event("window_age", state=d.state(full=True))
    d.drain("window_age_end")


def enqueue_identity_case(d, config, variant):
    from vllm.v1.request import RequestStatus

    if variant.startswith("initial_"):
        options = {
            "initial_session_id": {"session_id": "native-session"},
            "initial_resumable": {"resumable": True},
            "initial_abort_immediately": {"abort_immediately": True},
        }
        d.clock.error = "nonordinary initial admission must not read guard clock"
        request = req(config, "nonordinary", range(1000, 1033), **options[variant])
        d.add(request)
        require(
            d.clock.reads == 0 and not d.scheduler._wait_guard.records,
            "nonordinary initial admission acquired guard metadata",
        )
        d.event(
            "nonordinary_admitted",
            variant=variant,
            object=d.registry.token(request),
            reads=d.clock.reads,
            state=d.state(full=True),
        )
        cancel_all(d)
        return
    first = req(
        config, "same", range(1000, 1033), max_tokens=1, resumable=variant == "session_update"
    )
    d.add(first)
    baseline = guard_snapshot(d.scheduler, d.registry)
    d.at(100_000_000)
    if variant in {"same_object_duplicate", "different_object_duplicate"}:
        duplicate = (
            first if variant == "same_object_duplicate" else req(config, "same", range(2000, 2033))
        )
        expect_error(d, lambda: d.add(duplicate), variant, expected=AssertionError)
        require(
            guard_snapshot(d.scheduler, d.registry) == baseline,
            "duplicate add refreshed guard metadata",
        )
    else:
        d.schedule("session_first_input")
        d.consume()
        require(
            first.status is RequestStatus.WAITING_FOR_STREAMING_REQ,
            "native session did not wait for next input",
        )
        d.add(req(config, "same", range(2000, 2033), max_tokens=1, resumable=True))
        require(
            d.scheduler.requests["same"] is first and not d.scheduler._wait_guard.records,
            "session update created guard generation",
        )
        d.event("session_updated", state=d.state(full=True))
    cancel_all(d)


def metadata_terminal_case(d, config, variant):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    requests = [req(config, f"T{i}", range(1000 + 100 * i, 1033 + 100 * i)) for i in range(3)]
    for request in requests:
        d.add(request)
    ids = [r.request_id for r in requests]
    supplied = (
        ids[0]
        if variant in {"str", "free_error"}
        else ids
        if variant == "list"
        else None
        if variant == "none"
        else ["unknown", ids[0], ids[0], ids[1]]
    )
    if variant == "generator":

        def once():
            for rid in (ids[0], ids[1], ids[1], "unknown"):
                d.event("generator_yield", request_id=rid)
                yield rid

        supplied = once()
    d.event(
        "finish_call",
        variant=variant,
        supplied_kind=type(supplied).__name__,
        supplied_input=list(supplied)
        if type(supplied) is list
        else supplied
        if type(supplied) is str
        else None,
        state=d.state(full=True),
    )
    if variant == "free_error":
        original = Scheduler._free_request

        def fail_free(instance, request, *args, **kwargs):
            raise RuntimeError("injected native terminal cleanup failure")

        Scheduler._free_request = fail_free
        try:
            expect_error(
                d,
                lambda: d.scheduler.finish_requests(ids[0], RequestStatus.FINISHED_ABORTED),
                "free_error",
            )
        finally:
            Scheduler._free_request = original
        require(
            requests[0] not in d.scheduler._wait_guard.records,
            "terminal finally did not clear guard record",
        )
        verify_fault_retries(d, config, variant)
        # This is the same still-bound terminal native object, never an old generation callback.
        d.event("terminal_recovery", object=d.registry.token(requests[0]), state=d.state(full=True))
        d.scheduler._free_request(requests[0])
    else:
        finished = d.scheduler.finish_requests(supplied, RequestStatus.FINISHED_ABORTED)
        d.event(
            "finish_return",
            variant=variant,
            objects=[d.registry.token(r) for r in finished],
            state=d.state(full=True),
        )
    cancel_all(d)


def shutdown_case(d, config, variant):
    d.add(req(config, "R", range(1000, 1033)))
    if variant == "running":
        d.schedule("shutdown_pending")
    d.add(req(config, "W", range(2000, 2033)))
    before = d.state(full=True)
    if variant == "raises":
        publisher = d.scheduler.kv_event_publisher
        original = publisher.shutdown

        def fail_shutdown():
            original()
            d.event(
                "fault_injection",
                site="native_publisher_shutdown_after_release",
                error="injected native shutdown failure",
            )
            raise RuntimeError("injected native shutdown failure")

        publisher.shutdown = fail_shutdown
        try:
            expect_error(d, d.scheduler.shutdown, "shutdown_raises")
        finally:
            publisher.shutdown = original
    else:
        d.scheduler.shutdown()
    d.shutdown_called = True
    after = d.state(full=True)
    require(
        not d.scheduler._wait_guard.records and d.scheduler._wait_guard.closed,
        "shutdown did not close own guard state",
    )
    require(bool(d.scheduler.requests), "native shutdown was incorrectly treated as request drain")
    d.event(
        "shutdown_observation",
        variant=variant,
        before=before,
        after=after,
        pending_outputs=len(d.pending),
    )
    expect_error(d, lambda: d.schedule("after_shutdown"), "closed_schedule")
    expect_error(d, lambda: d.add(req(config, "after_closed", [42])), "closed_add")
    cancel_all(d)


def drain_reuse_notifications(d, phase):
    """Fence a new ID generation behind native output, resource and notice drain.

    No live request is scheduled here: ordinary drain must already have released
    all owning state. Remaining terminal/preemption IDs travel through a genuine
    empty SchedulerOutput and its FIFO consumer, never a fixture-side clear().
    """
    for attempt in range(d.config["max_steps_per_phase"]):
        scheduler = d.scheduler
        state = d.state(full=True)
        require(
            not d.pending
            and not scheduler.requests
            and not scheduler.running
            and not scheduler.waiting
            and not scheduler.kv_holding_waiting
            and not scheduler.deferred_waiting,
            "ID reuse notice drain started before native requests/outputs drained",
        )
        require(
            not scheduler._inflight_prefills and not scheduler.deferred_frees,
            "ID reuse notice drain started with native deferred ownership",
        )
        require(
            all(not group for group in state["request_blocks"])
            and all(block["ref_cnt"] == 0 for block in state["blocks"] if not block["null"])
            and len(state["free_queue"])
            == len(set(state["free_queue"]))
            == scheduler.kv_cache_config.num_blocks - 1,
            "ID reuse notice drain started before KV resources were released",
        )
        guard = getattr(scheduler, "_wait_guard", None)
        require(guard is None or not guard.records, "ID reuse retained old protection records")
        finished = set(scheduler.finished_req_ids)
        preempted = set(scheduler.reset_preempted_req_ids)
        if not finished and not preempted:
            d.event(
                "reuse_notifications_drained",
                phase=phase,
                attempts=attempt,
                pending_outputs=0,
                state=state,
            )
            return
        output = d.schedule(f"{phase}_terminal_notifications")
        require(
            output.total_num_scheduled_tokens == 0 and not output.num_scheduled_tokens,
            "ID reuse notice drain unexpectedly dispatched request tokens",
        )
        require(
            set(output.finished_req_ids) == finished
            and set(output.preempted_req_ids or ()) == preempted,
            "ID reuse notice drain did not forward the native terminal/preemption IDs",
        )
        d.consume()
    raise RuntimeError(f"{phase}: finite ID reuse notification drain limit exceeded")


def id_reuse_case(d, config, variant):
    old = req(config, "reuse", range(1000, 1033))
    d.add(old)
    d.drain("generation0")
    drain_reuse_notifications(d, "generation0")
    d.event("reuse_boundary", pending_outputs=len(d.pending), state=d.state(full=True))
    d.at(1_000_000_000)
    fresh = req(config, "reuse", range(2000, 2033), arrival=-20)
    d.add(fresh)
    guard = getattr(d.scheduler, "_wait_guard", None)
    if guard is not None and variant == "stale_object_cleanup":
        before = guard_snapshot(d.scheduler, d.registry)
        removed = guard.discard(old, "identity_probe")
        require(
            not removed and guard_snapshot(d.scheduler, d.registry) == before,
            "old object cleanup deleted a new binding",
        )
        d.event(
            "identity_probe",
            old_object=d.registry.token(old),
            new_object=d.registry.token(fresh),
            removed=removed,
            before=before,
            after=guard_snapshot(d.scheduler, d.registry),
        )
    d.drain("generation1")
    d.add(req(config, "cancel_reuse", range(3000, 3033)))
    d.schedule("cancel_generation0")
    d.cancel("cancel_reuse")
    d.drain("cancel_generation0")
    drain_reuse_notifications(d, "cancel_generation0")
    d.event("reuse_boundary", pending_outputs=len(d.pending), state=d.state(full=True))
    d.at(2_000_000_000)
    d.add(req(config, "cancel_reuse", range(4000, 4033), arrival=-30))
    d.drain("cancel_generation1")


def repeat_cycles_case(d, config):
    for cycle in range(8):
        d.at(cycle * 1_000_000_000)
        d.add(req(config, "cycled", range(1000 + 100 * cycle, 1033 + 100 * cycle), max_tokens=1))
        if cycle % 2:
            d.cancel("cycled")
        else:
            d.schedule(f"cycle_{cycle}")
            d.consume()
        d.drain(f"cycle_{cycle}_drain")
        drain_reuse_notifications(d, f"cycle_{cycle}")
        require(not d.scheduler._wait_guard.records, "cycle retained protection metadata")
        d.event(
            "cycle_drained", cycle=cycle, pending_outputs=len(d.pending), state=d.state(full=True)
        )


def window_case(d, config, label):
    from vllm.v1.request import RequestStatus

    scheduler = d.scheduler
    window = scheduler.prefix_config.window
    if label in {"missing_singleton", "missing_window_one"}:
        request = req(config, "missing", range(1000, 1033))
        d.add(request)
        before = guard_snapshot(scheduler, d.registry)
        scheduler._wait_guard.records.pop(request)
        d.event(
            "fixture_remove_record",
            object=d.registry.token(request),
            before=before,
            after=guard_snapshot(scheduler, d.registry),
        )
        expect_error(d, lambda: d.schedule(label), label)
        require(
            scheduler._wait_guard.fault_code == "missing_record", "missing age did not fail closed"
        )
        verify_fault_retries(d, config, label)
        cancel_all(d)
        return
    historical._warm(d, config)
    hot = list(range(100, 132)) + [201]
    specs = [("cold", list(range(700, 733))), ("hot", hot), ("tail", hot)]
    if label == "stable_ties":
        specs = [("first", hot), ("second", hot), ("tail", hot)]
    elif label == "all_cold":
        specs = [("first", list(range(700, 733))), ("second", list(range(800, 833)))]
    reqs = [
        req(
            config,
            rid,
            tokens,
            arrival=float([2, 0, 1][i]) if label == "stable_ties" else float(i),
        )
        for i, (rid, tokens) in enumerate(specs)
    ]
    for r in reqs:
        d.add(r)
    if label in {"head_barrier", "middle_barrier"}:
        barrier = reqs[0 if label == "head_barrier" else 1]
        barrier.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        scheduler.deferred_waiting.add(barrier)
    fields = {
        "preempt_history": ("num_preemptions", 1),
        "partial_progress": ("num_computed_tokens", 1),
        "placeholder": ("num_output_placeholders", 1),
        "in_flight": ("num_in_flight_tokens", 1),
        "stale": ("num_stale_output_tokens", 1),
        "resumable": ("resumable", True),
        "unsupported": ("session_id", "unsupported-session"),
    }
    if label in fields:
        name, setting = fields[label]
        setattr(reqs[0], name, setting)
    if label == "other_queues":
        scheduler.running.append(req(config, "other_running", [42]))
        scheduler.kv_holding_waiting.add_request(req(config, "other_holding", [43]))
        scheduler.deferred_waiting.add(req(config, "other_deferred", [44]))
    before = [r.request_id for r in scheduler.waiting]
    untouched = (
        list(scheduler.running),
        list(scheduler.kv_holding_waiting),
        set(scheduler.deferred_waiting),
    )
    restored = []
    expected_error = label in {"score_exception", "pop_exception", "prepend_exception"}
    if label == "score_exception":
        original = scheduler._score_request

        def fail_score(r):
            raise RuntimeError("injected score failure")

        restored.append((scheduler, "_score_request", original))
        scheduler._score_request = fail_score
    elif label == "pop_exception":
        original = scheduler.waiting.pop_request

        def fail_pop(pop=original):
            pop()
            raise RuntimeError("injected after-pop failure")

        restored.append((scheduler.waiting, "pop_request", original))
        scheduler.waiting.pop_request = fail_pop
    elif label == "prepend_exception":
        original = scheduler.waiting.prepend_requests

        def fail_prepend(items, queue=scheduler.waiting):
            iterator = iter(items)
            queue.appendleft(next(iterator))
            raise RuntimeError("injected partial-prepend failure")

        restored.append((scheduler.waiting, "prepend_requests", original))
        scheduler.waiting.prepend_requests = fail_prepend
    d.step += 1
    caught = None
    if label == "prepend_order":
        popped = [scheduler.waiting.pop_request() for _ in reqs]
        scheduler.waiting.prepend_requests(reversed(popped))
    else:
        try:
            if window == 1:
                # Exercise actual fast schedule, observed just before native
                # processing; other tests isolate the real queue transaction.
                d.schedule("window_one")
            else:
                scheduler._reorder_waiting()
                if label == "repeated_step":
                    d.step += 1
                    scheduler._reorder_waiting()
        except RuntimeError as error:
            if not expected_error:
                raise
            caught = str(error)
    for obj, name, original in restored:
        setattr(obj, name, original)
    after = [r.request_id for r in scheduler.waiting]
    if expected_error:
        require(caught is not None and after == before, "queue rollback failed")
    elif label not in {"window_one", "repeated_step"}:
        expected = (
            ["hot", "cold", "tail"]
            if label == "window_limit"
            else ["hot", "tail", "cold"]
            if label == "other_queues"
            else before
        )
        require(after == expected, f"window case {label}: unexpected order {after}")
    if label != "window_one":
        require(
            untouched
            == (
                list(scheduler.running),
                list(scheduler.kv_holding_waiting),
                set(scheduler.deferred_waiting),
            ),
            "window changed other queue priority",
        )
    d.event(
        "check",
        name=label,
        before=before,
        after=after,
        error=caught,
        window=window,
        counts=d.counts().copy(),
        candidates=d.step_candidates.get(d.step, []),
        scores=d.step_scores.get(d.step, []),
        state=d.state(full=True),
        branch_fixture=True,
    )
    # Branch fixtures may alter progress/status. They are cancelled through
    # the native terminal API, never used as substitutes for lifecycle cases.
    if label == "other_queues":
        scheduler.running.clear()
        scheduler.kv_holding_waiting.clear()
        scheduler.deferred_waiting.clear()
    for r in reqs:
        if r.request_id in scheduler.requests:
            d.cancel(r.request_id)
    d.drain(label)


def probe_case(d, config, variant):
    from vllm_prefix_scheduler.probe import reusable_prefix_tokens

    historical._warm(d, config)
    common = list(range(100, 132))
    tests = [
        ("cold", list(range(500, 533)), 0, False),
        ("partial", common[:16] + list(range(600, 617)), 16, False),
        ("full", common + [201], 32, False),
        ("exact16", common[:16], 0, False),
        ("exact32", common, 16, False),
        ("one", [100], 0, False),
        ("skip_read", common + [201], 0, True),
    ]
    for label, tokens, expected, skip in tests:
        if label != variant:
            continue
        r = req(config, label, tokens, skip_read=skip)
        d.requests[label] = r
        before = d.state(full=True)
        d.event("probe_before", probe_id=label, state=before, expected=expected)
        score = reusable_prefix_tokens(d.scheduler.kv_cache_manager, r)
        after = d.state(full=True)
        d.event("probe_after", probe_id=label, state=after, score=score, expected=expected)
        require(before == after and score == expected, f"probe {label} changed state or score")
    # Prove normal native reporting AFTER the probes and BEFORE positive controls.
    # Warmup has already drained its events. No observer calls a draining API.
    normal = d.requests.get("full") or req(config, "full", common + [201])
    d.requests = {"A": d.requests["A"]}
    published_before, updates_before = len(d.published), len(d.output_stats)
    begin = d.state(full=True)
    require(begin["full"]["events"] == [], "old events would contaminate post-probe proof")
    d.event(
        "post_probe_begin",
        state=begin,
        published_count=published_before,
        output_count=updates_before,
    )
    d.add(normal)
    d.drain("post_probe_native")
    require(
        len(d.published) > published_before and len(d.output_stats) > updates_before,
        "native post-probe event/stat output missing",
    )
    d.event(
        "post_probe_end",
        state=d.state(full=True),
        published_delta=len(d.published) - published_before,
        output_delta=len(d.output_stats) - updates_before,
    )

    # Native-effect positive controls use a separate detached Request. Their
    # effects cannot be mistaken for the normal reporting proved above.
    manager = d.scheduler.kv_cache_manager
    pool = manager.block_pool
    r = req(config, "sensitivity_control", common + [201])
    d.requests[r.request_id] = r
    before = d.state(full=True)
    blocks, hit, _ = manager.get_computed_blocks(r)
    after = d.state(full=True)
    require(
        before != after and hit == 32 and before["full"]["events"] != after["full"]["events"],
        "full-event sensitivity",
    )
    d.event("check", name="sensitivity_event", before=before, after=after)
    before = d.state(full=True)
    pool.touch([b for group in blocks.blocks for b in group])
    after = d.state(full=True)
    require(before != after and before["free_queue"] != after["free_queue"], "touch sensitivity")
    d.event("check", name="sensitivity_touch", before=before, after=after)
    pool.free_blocks([b for group in blocks.blocks for b in group])
    before = d.state(full=True)
    manager.record_prefix_cache_stats(r, hit)
    after = d.state(full=True)
    require(before["full"]["stats"] != after["full"]["stats"], "stats sensitivity")
    d.event("check", name="sensitivity_stats", before=before, after=after)
    # Consume controls through another ordinary native lifetime; never clear
    # statistics or event buffers just to make the final state appear drained.
    d.requests.pop(r.request_id)
    d.add(r)
    d.drain("sensitivity_controls_native_drain")


def global_rejection_case(config, mode, model_path, emit, variant):
    import torch
    from vllm.config import (
        ECTransferConfig,
        KVTransferConfig,
        LoRAConfig,
        MultiModalConfig,
        SpeculativeConfig,
    )
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    tests = {
        "speculative": lambda vc, kv, args: setattr(
            vc,
            "speculative_config",
            SpeculativeConfig(
                method="ngram",
                num_speculative_tokens=1,
                target_model_config=vc.model_config,
                target_parallel_config=vc.parallel_config,
            ),
        ),
        "multimodal": lambda vc, kv, args: setattr(
            vc.model_config, "multimodal_config", MultiModalConfig()
        ),
        "kv_connector": lambda vc, kv, args: setattr(
            vc,
            "kv_transfer_config",
            KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both"),
        ),
        "ec_connector": lambda vc, kv, args: setattr(
            vc,
            "ec_transfer_config",
            ECTransferConfig(ec_connector="ECExampleConnector", ec_role="ec_both"),
        ),
        "encoder_decoder": lambda vc, kv, args: historical._mutate_model_flag(
            vc, "is_encoder_decoder", True, "is_encoder_decoder"
        ),
        "diffusion": lambda vc, kv, args: historical._mutate_model_flag(
            vc, "canvas_length", 1, "is_diffusion"
        ),
        "nonfull_spec": lambda vc, kv, args: setattr(
            kv.kv_cache_groups[0],
            "kv_cache_spec",
            SlidingWindowSpec(
                block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=16
            ),
        ),
        "external_lb": lambda vc, kv, args: setattr(
            vc.parallel_config, "data_parallel_external_lb", True
        ),
        "hybrid_lb": lambda vc, kv, args: setattr(
            vc.parallel_config, "data_parallel_hybrid_lb", True
        ),
        "cuda_only": lambda vc, kv, args: setattr(vc.device_config, "device_type", "cpu"),
        "lora": lambda vc, kv, args: setattr(vc, "lora_config", LoRAConfig()),
        "kv_offload": lambda vc, kv, args: setattr(vc.cache_config, "kv_offloading_size", 1.0),
        "cpu_offload": lambda vc, kv, args: setattr(vc.offload_config.uva, "cpu_offload_gb", 1.0),
        "prefetch_offload": lambda vc, kv, args: setattr(
            vc.offload_config.prefetch, "offload_group_size", 1
        ),
        "pooling": lambda vc, kv, args: setattr(vc.model_config, "runner_type", "pooling"),
        "expert_parallel": lambda vc, kv, args: setattr(
            vc.parallel_config, "enable_expert_parallel", True
        ),
        "dbo": lambda vc, kv, args: setattr(vc.parallel_config, "enable_dbo", True),
        "sliding_window": lambda vc, kv, args: object.__setattr__(
            kv.kv_cache_groups[0].kv_cache_spec, "sliding_window", 16
        ),
        "attention_chunk": lambda vc, kv, args: object.__setattr__(
            kv.kv_cache_groups[0].kv_cache_spec, "attention_chunk_size", 16
        ),
        "apc_off": lambda vc, kv, args: setattr(vc.cache_config, "enable_prefix_caching", False),
        "async_off": lambda vc, kv, args: setattr(vc.scheduler_config, "async_scheduling", False),
        "priority": lambda vc, kv, args: setattr(vc.scheduler_config, "policy", "priority"),
        "tensor_parallel": lambda vc, kv, args: setattr(
            vc.parallel_config, "tensor_parallel_size", 2
        ),
        "pipeline_parallel": lambda vc, kv, args: setattr(
            vc.parallel_config, "pipeline_parallel_size", 2
        ),
        "data_parallel": lambda vc, kv, args: setattr(vc.parallel_config, "data_parallel_size", 2),
        "decode_parallel": lambda vc, kv, args: setattr(
            vc.parallel_config, "decode_context_parallel_size", 2
        ),
        "prefill_parallel": lambda vc, kv, args: setattr(
            vc.parallel_config, "prefill_context_parallel_size", 2
        ),
        "aux_output": lambda vc, kv, args: setattr(
            vc.aux_output_config, "enable_return_routed_experts", True
        ),
        "hash_block": lambda vc, kv, args: args.update(hash_block_size=8),
        "multiple_groups": lambda vc, kv, args: kv.kv_cache_groups.append(kv.kv_cache_groups[0]),
        "noncausal": lambda vc, kv, args: object.__setattr__(
            kv.kv_cache_groups[0].kv_cache_spec, "non_causal", True
        ),
    }
    # Every mutation is checked at its effective read location before resolving
    # or constructing the production class. Also bind the specific rejection
    # reason, so an unrelated setup error cannot count as a successful negative.
    expectations = {
        "speculative": (
            lambda vc, kv, args: vc.speculative_config is not None,
            True,
            "speculative_config",
        ),
        "multimodal": (
            lambda vc, kv, args: vc.model_config.multimodal_config is not None,
            True,
            "multimodal",
        ),
        "kv_connector": (
            lambda vc, kv, args: vc.kv_transfer_config is not None,
            True,
            "kv_transfer_config",
        ),
        "ec_connector": (
            lambda vc, kv, args: vc.ec_transfer_config is not None,
            True,
            "ec_transfer_config",
        ),
        "encoder_decoder": (
            lambda vc, kv, args: vc.model_config.is_encoder_decoder,
            True,
            "is_encoder_decoder",
        ),
        "diffusion": (lambda vc, kv, args: vc.model_config.is_diffusion, True, "is_diffusion"),
        "nonfull_spec": (
            lambda vc, kv, args: type(kv.kv_cache_groups[0].kv_cache_spec).__name__,
            "SlidingWindowSpec",
            "exact FullAttentionSpec",
        ),
        "external_lb": (
            lambda vc, kv, args: vc.parallel_config.data_parallel_external_lb,
            True,
            "data_parallel_external_lb",
        ),
        "hybrid_lb": (
            lambda vc, kv, args: vc.parallel_config.data_parallel_hybrid_lb,
            True,
            "data_parallel_hybrid_lb",
        ),
        "cuda_only": (lambda vc, kv, args: vc.device_config.device_type, "cpu", "CUDA"),
        "lora": (lambda vc, kv, args: vc.lora_config is not None, True, "lora_config"),
        "kv_offload": (
            lambda vc, kv, args: vc.cache_config.kv_offloading_size,
            1.0,
            "KV offloading",
        ),
        "cpu_offload": (
            lambda vc, kv, args: vc.offload_config.uva.cpu_offload_gb,
            1.0,
            "CPU offload",
        ),
        "prefetch_offload": (
            lambda vc, kv, args: vc.offload_config.prefetch.offload_group_size,
            1,
            "prefetch offload",
        ),
        "pooling": (
            lambda vc, kv, args: vc.model_config.runner_type,
            "pooling",
            "ordinary text generation",
        ),
        "expert_parallel": (
            lambda vc, kv, args: vc.parallel_config.enable_expert_parallel,
            True,
            "enable_expert_parallel",
        ),
        "dbo": (lambda vc, kv, args: vc.parallel_config.enable_dbo, True, "enable_dbo"),
        "sliding_window": (
            lambda vc, kv, args: kv.kv_cache_groups[0].kv_cache_spec.sliding_window,
            16,
            "causal full attention",
        ),
        "attention_chunk": (
            lambda vc, kv, args: kv.kv_cache_groups[0].kv_cache_spec.attention_chunk_size,
            16,
            "causal full attention",
        ),
        "apc_off": (
            lambda vc, kv, args: vc.cache_config.enable_prefix_caching,
            False,
            "APC enabled",
        ),
        "async_off": (
            lambda vc, kv, args: vc.scheduler_config.async_scheduling,
            False,
            "effective async_scheduling=True",
        ),
        "priority": (lambda vc, kv, args: vc.scheduler_config.policy, "priority", "FCFS"),
        "tensor_parallel": (
            lambda vc, kv, args: vc.parallel_config.tensor_parallel_size,
            2,
            "tensor_parallel_size",
        ),
        "pipeline_parallel": (
            lambda vc, kv, args: vc.parallel_config.pipeline_parallel_size,
            2,
            "pipeline_parallel_size",
        ),
        "data_parallel": (
            lambda vc, kv, args: vc.parallel_config.data_parallel_size,
            2,
            "data_parallel_size",
        ),
        "decode_parallel": (
            lambda vc, kv, args: vc.parallel_config.decode_context_parallel_size,
            2,
            "decode_context_parallel_size",
        ),
        "prefill_parallel": (
            lambda vc, kv, args: vc.parallel_config.prefill_context_parallel_size,
            2,
            "prefill_context_parallel_size",
        ),
        "aux_output": (lambda vc, kv, args: vc.aux_output_config.enabled, True, "auxiliary output"),
        "hash_block": (
            lambda vc, kv, args: args["hash_block_size"],
            8,
            "identical scheduler, hash and physical",
        ),
        "multiple_groups": (lambda vc, kv, args: len(kv.kv_cache_groups), 2, "exactly one"),
        "noncausal": (
            lambda vc, kv, args: kv.kv_cache_groups[0].kv_cache_spec.non_causal,
            True,
            "causal full attention",
        ),
    }
    require(set(expectations) == set(tests), "negative fixture observations incomplete")
    rejected = []
    for label, mutate in tests.items():
        if label != variant:
            continue
        read_effective, expected, fragment = expectations[label]

        def checked_mutation(
            vc,
            kv,
            args,
            mutate=mutate,
            read_effective=read_effective,
            expected=expected,
            label=label,
            fragment=fragment,
        ):
            mutate(vc, kv, args)
            effective = read_effective(vc, kv, args)
            require(
                type(effective) is type(expected) and effective == expected,
                f"negative fixture {label} did not take effect: {effective}",
            )
            emit(
                {
                    "event": "negative_config",
                    "step": 0,
                    "name": label,
                    "effective": effective,
                    "expected": expected,
                    "error_fragment": fragment,
                }
            )

        caught = None
        try:
            scheduler = historical.create_scheduler(
                config, mode, model_path, mutate=checked_mutation
            )
        except (ValueError, RuntimeError, NotImplementedError) as error:
            caught = str(error)
        else:
            scheduler.shutdown()
        require(
            caught is not None and "prefix_scheduler:" in caught and fragment in caught,
            f"incompatible config {label} did not fail in the extension: {caught}",
        )
        rejected.append(label)
        emit({"event": "check", "step": 0, "name": label, "rejected": True, "error": caught})
    for window in [1] if variant == "window1_apc_off" else []:
        label = "window1_apc_off"
        fragment = "APC enabled"

        def checked_special(vc, kv, args, mode=mode, window=window, label=label, fragment=fragment):
            effective = {
                "mode": vc.additional_config["prefix_scheduler"]["mode"],
                "window": vc.additional_config["prefix_scheduler"]["window"],
                "apc": vc.cache_config.enable_prefix_caching,
            }
            expected = {"mode": mode, "window": window, "apc": False}
            require(effective == expected, "negative fast-path fixture did not take effect")
            emit(
                {
                    "event": "negative_config",
                    "step": 0,
                    "name": label,
                    "effective": effective,
                    "expected": expected,
                    "error_fragment": fragment,
                }
            )

        caught = None
        try:
            scheduler = historical.create_scheduler(
                config, mode, model_path, apc=False, window=window, mutate=checked_special
            )
        except (ValueError, RuntimeError, NotImplementedError) as error:
            caught = str(error)
        else:
            scheduler.shutdown()
        require(
            caught is not None and fragment in caught,
            "fast-path configuration bypassed target validation",
        )
        label = "window1_apc_off"
        rejected.append(label)
        emit({"event": "check", "step": 0, "name": label, "rejected": True, "error": caught})
    return {"status": "passed", "rejected": rejected}


def postcheck_case(config, mode, model_path, emit, variant):
    from collections import deque

    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

    original_validate = PrefixAsyncScheduler._validate_native_objects
    original_shutdown = Scheduler.shutdown
    observed = []

    def emit_shutdown(instance):
        emit({"event": "native_shutdown_call", "step": 0, "construction_cleanup": True})
        try:
            result = original_shutdown(instance)
        except BaseException as error:
            emit(
                {
                    "event": "native_shutdown_raise",
                    "step": 0,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        emit({"event": "native_shutdown_return", "step": 0, "construction_cleanup": True})
        return result

    def injected_validate(instance):
        emit({"event": "native_objects", "step": 0, **historical.source_record(instance)})
        coordinator = instance.kv_cache_manager.coordinator
        if variant == "queue":
            owner, field, replacement = instance, "waiting", deque()
        elif variant == "coordinator":
            owner, field, replacement = instance.kv_cache_manager, "coordinator", object()
        elif variant == "full_attention":
            owner, field, replacement = coordinator, "single_type_managers", [object()]
        elif variant == "block_size":
            owner, field, replacement = coordinator, "block_size", coordinator.block_size * 2
        else:
            owner, field, replacement = instance, "use_v2_model_runner", False
        old = getattr(owner, field)
        setattr(owner, field, replacement)
        observed.append(variant)
        effective = (
            replacement
            if type(replacement) in (int, bool)
            else type(replacement[0]).__name__
            if isinstance(replacement, list)
            else type(replacement).__name__
        )
        emit(
            {
                "event": "postcheck_input",
                "step": 0,
                "variant": variant,
                "field": field,
                "effective": effective,
            }
        )
        try:
            return original_validate(instance)
        finally:
            setattr(owner, field, old)

    PrefixAsyncScheduler._validate_native_objects = injected_validate
    Scheduler.shutdown = emit_shutdown
    caught = None
    try:
        scheduler = historical.create_scheduler(config, mode, model_path)
    except ValueError as error:
        caught = str(error)
    else:
        scheduler.shutdown()
    finally:
        PrefixAsyncScheduler._validate_native_objects = original_validate
        Scheduler.shutdown = original_shutdown
    fragments = {
        "queue": "FCFSRequestQueue",
        "coordinator": "UnitaryKVCacheCoordinator",
        "full_attention": "FullAttentionManager",
        "block_size": "native cache block sizes",
        "v2": "native Model Runner V2",
    }
    require(
        observed == [variant] and caught is not None and fragments[variant] in caught,
        "native postcheck failed for wrong reason or was bypassed",
    )
    emit({"event": "postcheck_rejected", "step": 0, "variant": variant, "error": caught})
    return {"status": "passed", "rejected": variant}


def fixture_parameters(config, metadata):
    """Explicit per-leaf geometry; included verbatim in the raw case trace."""
    scenario, variant = metadata["scenario"], metadata["variant"]
    local = copy.deepcopy(config)
    parameters = {
        "capacity": 5
        if scenario
        in {"bc_pressure", "preemption", "score_eviction", "allocation_retry", "hot_flow"}
        else 10
    }
    if scenario == "chunk":
        parameters.update(budget=16, active=1)
    if scenario in {"hot_flow", "clock_thresholds", "protected_order", "window_age"}:
        parameters["active"] = 1
    if scenario == "hot_flow":
        local["max_tokens"] = 1
    if scenario == "clock_thresholds":
        local["max_wait_ms"] = config["clock_plans"][variant]["max_wait_ms"]
    if scenario == "window":
        parameters["window"] = (
            1
            if variant in {"window_one", "missing_window_one"}
            else 2
            if variant == "window_limit"
            else 16
        )
    if scenario == "window_age":
        parameters["window"] = 2
    parameters["apc"] = scenario != "apc_off"
    return local, parameters


def run_case(config, model_path, metadata, emit):
    scenario, mode, variant = metadata["scenario"], metadata["mode"], metadata["variant"]
    if scenario == "global_rejection":
        return global_rejection_case(config, mode, model_path, emit, variant)
    if scenario == "postcheck_failure":
        return postcheck_case(config, mode, model_path, emit, variant)
    local, parameters = fixture_parameters(config, metadata)
    initial = config["clock_plans"][variant]["enqueue_ns"] if scenario == "clock_thresholds" else 0
    clock = ControlledClock(initial)
    if scenario == "clock_absence":
        clock.error = "off/prefix must never consult the private protection clock"
    scheduler = create_scheduler(local, mode, model_path, clock, **parameters)
    d = Driver(local, scheduler, emit, clock)
    d.event(
        "fixture_parameters",
        parameters=parameters,
        max_wait_ms=local["max_wait_ms"],
        max_tokens=local["max_tokens"],
    )
    try:
        from m2b_fault_cases import FAULT_VARIANTS, run_fault_case

        if variant in FAULT_VARIANTS:
            run_fault_case(d, local, variant)
        elif scenario in {"bc_pressure", "bc_ample"}:
            bc_case(d, local, variant)
        elif scenario == "apc_off":
            historical.bc_case(d, local, False)
        elif scenario == "probe":
            probe_case(d, local, variant)
        elif scenario == "window":
            window_case(d, local, variant)
        elif scenario == "chunk":
            historical.chunk_case(d, local)
        elif scenario == "preemption":
            historical.preemption_case(d, local)
        elif scenario == "cancellation":
            historical.cancellation_case(d, local)
        elif scenario == "id_reuse":
            id_reuse_case(d, local, variant)
        elif scenario == "score_eviction":
            historical.score_eviction_case(d, local)
        elif scenario == "clock_thresholds":
            threshold_case(d, local, variant)
        elif scenario == "clock_absence":
            historical.bc_case(d, local, True)
            historical.cancellation_case(d, local)
            require(
                clock.reads == 0 and not hasattr(scheduler, "_wait_guard"),
                "off/prefix used protection clock or state",
            )
            d.event(
                "clock_absence", reads=clock.reads, guard_present=hasattr(scheduler, "_wait_guard")
            )
        elif scenario == "enqueue_identity":
            enqueue_identity_case(d, local, variant)
        elif scenario == "protected_order":
            protected_order_case(d, local, variant)
        elif scenario == "window_age":
            window_age_case(d, local)
        elif scenario == "allocation_retry":
            allocation_retry_case(d, local)
        elif scenario == "hot_flow":
            hot_flow_case(d, local)
        elif scenario == "metadata_terminal":
            metadata_terminal_case(d, local, variant)
        elif scenario == "shutdown":
            shutdown_case(d, local, variant)
        elif scenario == "repeat_cycles":
            repeat_cycles_case(d, local)
        else:
            raise RuntimeError(f"Unimplemented frozen leaf: {metadata}")
        result = d.finish()
        result["clock_reads"] = clock.reads
        return result
    finally:
        d.close()


def run_matrix(config, model_path, output_path, suite="full"):
    require_assertions()
    expected = expected_cases(config, suite=suite)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    results = []
    with (output_path / "trace.jsonl").open("x") as stream:
        for case, metadata in expected.items():

            def emit(row, case=case, metadata=metadata):
                stream.write(
                    json.dumps({"case": case, **metadata, **row}, sort_keys=True, allow_nan=False)
                    + "\n"
                )
                stream.flush()

            emit({"event": "case_begin", "step": 0, "seed": config["seed"]})
            try:
                result = run_case(config, model_path, metadata, emit)
            except BaseException as error:
                emit(
                    {
                        "event": "case_failure",
                        "step": 0,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                raise
            result = {"case": case, **metadata, **result}
            results.append(result)
            emit({"event": "case_end", "step": 0, "result": result})
            (output_path / "summary.json").write_text(
                json.dumps({"verified": False, "results": results}, indent=2) + "\n"
            )
    return results
