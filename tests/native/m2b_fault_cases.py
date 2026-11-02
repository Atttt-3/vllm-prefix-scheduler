"""Explicit native-boundary fault leaves, separate from natural lifecycle cases.

The fixed Scheduler still performs admission, allocation, preemption and freeing.
Injected failures and any fixture-only orphan teardown are recorded as such;
none of these branches is evidence for natural preemption or old-output ID reuse.
"""

import gc
import weakref
from contextlib import contextmanager

from vllm_prefix_scheduler.m2b_contract import require

FAULT_VARIANTS = frozenset(
    {
        "add_failure_prebind",
        "add_failure_postbind",
        "generator_raise",
        "batch_free_error",
        "schedule_before_update_error",
        "schedule_after_update_error",
        "preempt_free_error",
        "preempt_after_transition_error",
        "fault_weakref_release",
    }
)


class InjectedNativeFailure(RuntimeError):
    """An explicitly recorded fixture boundary failure, never a native result."""


@contextmanager
def _patched(owner, name, replacement):
    """Restore descriptor ownership as well as the previous callable."""
    owned = name in vars(owner)
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        if owned:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def _checkpoint(d, variant, **details):
    d.event(
        "fault_checkpoint",
        variant=variant,
        pending_outputs=len(d.pending),
        state=d.state(full=True),
        **details,
    )


def _check_latched(d, config, variant, expected_code):
    from m2b_harness import expect_error, req

    from vllm_prefix_scheduler.wait_guard import WaitGuardError

    guard = d.scheduler._wait_guard
    require(guard.fault_code == expected_code, f"{variant}: wrong first fault code")
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
    require(d.clock.reads == reads, f"{variant}: latched retries read protection clock")


def _cancel_bound(d, variant):
    from m2b_harness import cancel_all

    cancel_all(d)
    require(not d.scheduler._wait_guard.records, f"{variant}: bound cleanup kept guard records")
    d.event(
        "fault_recovery",
        variant=variant,
        action="native_finish_all_and_consume_existing_outputs",
        pending_outputs=len(d.pending),
        state=d.state(full=True),
    )


def _add_failure(d, config, variant):
    from m2a_snapshot import compact_request
    from m2b_harness import expect_error, req
    from vllm.v1.core.sched.scheduler import Scheduler

    old = req(config, "old_waiter", range(1000, 1033))
    d.add(old)
    guard = d.scheduler._wait_guard
    before = dict(guard.records)
    sequence = guard.next_sequence
    reads = d.clock.reads
    failed = req(config, "failed_add", range(2000, 2033))
    d.at(123)

    if variant == "add_failure_prebind":
        original = d.scheduler._enqueue_waiting_request

        def enqueue_then_fail(request):
            original(request)
            d.event(
                "fault_injection",
                variant=variant,
                site="native_enqueue_complete_before_requests_binding",
                object=d.registry.token(request),
                request_id=request.request_id,
                native_bound=request.request_id in d.scheduler.requests,
                request_state=compact_request(request),
                state=d.state(full=True),
            )
            raise InjectedNativeFailure(variant)

        with _patched(d.scheduler, "_enqueue_waiting_request", enqueue_then_fail):
            expect_error(d, lambda: d.add(failed), variant, expected=InjectedNativeFailure)
        require("failed_add" not in d.scheduler.requests, "prebind fault unexpectedly bound ID")
        require(any(r is failed for r in d.scheduler.waiting), "prebind fault lost native orphan")
    else:
        original = Scheduler.add_request

        def add_then_fail(instance, request):
            result = original(instance, request)
            if instance is d.scheduler and request is failed:
                d.event(
                    "fault_injection",
                    variant=variant,
                    site="native_add_returned_with_actual_binding",
                    object=d.registry.token(request),
                    request_id=request.request_id,
                    native_bound=instance.requests.get(request.request_id) is request,
                    request_state=compact_request(request),
                    state=d.state(full=True),
                )
                raise InjectedNativeFailure(variant)
            return result

        with _patched(Scheduler, "add_request", add_then_fail):
            expect_error(d, lambda: d.add(failed), variant, expected=InjectedNativeFailure)
        require(d.scheduler.requests.get("failed_add") is failed, "postbind fault lost binding")

    require(guard.records == before, "failed native add changed prior protection records")
    require(guard.next_sequence == sequence, "failed native add allocated protection sequence")
    require(d.clock.reads == reads, "failed native add read protection clock")
    _checkpoint(
        d, variant, failed_object=d.registry.token(failed), failed_state=compact_request(failed)
    )
    _check_latched(d, config, variant, "native_add")
    _cancel_bound(d, variant)
    if variant == "add_failure_prebind":
        # This failed admission is not in the native ID map, so finish_requests
        # cannot see it. Do not fabricate a binding or terminal transition.
        d.scheduler.waiting.remove_requests([failed])
        d.scheduler.kv_holding_waiting.remove_requests([failed])
        d.scheduler.deferred_waiting.discard(failed)
        d.event(
            "fault_recovery",
            variant=variant,
            action="fixture_detach_unbound_failed_admission",
            object=d.registry.token(failed),
            request_id=failed.request_id,
            native_bound=False,
            request_state=compact_request(failed),
            native_terminal=False,
            state=d.state(full=True),
        )


def _generator_raise(d, config, variant):
    from m2b_harness import expect_error, req
    from vllm.v1.request import RequestStatus

    requests = [req(config, f"G{i}", range(1000 + i * 100, 1033 + i * 100)) for i in range(3)]
    for request in requests:
        d.add(request)
    guard_before = dict(d.scheduler._wait_guard.records)
    waiting_before = list(d.scheduler.waiting)
    native_before = dict(d.scheduler.requests)

    def broken_ids():
        for request in requests[:2]:
            d.event("generator_yield", request_id=request.request_id)
            yield request.request_id
        d.event("fault_injection", variant=variant, site="finish_generator_materialization")
        raise InjectedNativeFailure(variant)

    supplied = broken_ids()
    d.event(
        "finish_call",
        variant=variant,
        supplied_kind="generator",
        supplied_input=None,
        state=d.state(full=True),
    )
    expect_error(
        d,
        lambda: d.scheduler.finish_requests(supplied, RequestStatus.FINISHED_ABORTED),
        variant,
        expected=InjectedNativeFailure,
    )
    require(d.scheduler.requests == native_before, "generator failure changed native binding")
    require(list(d.scheduler.waiting) == waiting_before, "generator failure changed waiting queue")
    require(d.scheduler._wait_guard.records == guard_before, "generator failure changed age")
    require(
        d.scheduler._wait_guard.fault_code is None, "generator failure spuriously latched guard"
    )
    require(
        all(r.status is RequestStatus.WAITING for r in requests), "generator partially terminated"
    )
    _checkpoint(d, variant, yielded_count=2)
    _cancel_bound(d, variant)


def _batch_free_failure(d, config, variant):
    from m2b_harness import expect_error, req
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    requests = [req(config, f"F{i}", range(1000 + i * 100, 1033 + i * 100)) for i in range(3)]
    for request in requests:
        d.add(request)
    original = Scheduler._free_request
    calls = []
    failed = []

    def second_free_fails(instance, request, *args, **kwargs):
        if instance is d.scheduler:
            calls.append(request.request_id)
            if len(calls) == 2:
                failed.append(request)
                d.event(
                    "fault_injection",
                    variant=variant,
                    site="second_native_free_before_release",
                    object=d.registry.token(request),
                    request_id=request.request_id,
                    free_call_order=list(calls),
                    state=d.state(full=True),
                )
                raise InjectedNativeFailure(variant)
        return original(instance, request, *args, **kwargs)

    d.event(
        "finish_call",
        variant=variant,
        supplied_kind="list",
        supplied_input=[r.request_id for r in requests],
        state=d.state(full=True),
    )
    with _patched(Scheduler, "_free_request", second_free_fails):
        expect_error(
            d,
            lambda: d.scheduler.finish_requests(
                [r.request_id for r in requests], RequestStatus.FINISHED_ABORTED
            ),
            variant,
            expected=InjectedNativeFailure,
        )
    require(len(calls) == 2 and len(failed) == 1, "batch failure did not stop at second free")
    remaining = [r for r in requests if not r.is_finished()]
    require(len(remaining) == 1, "batch failure did not retain one not-yet-terminal object")
    require(not d.scheduler.waiting, "native batch removal did not precede frees")
    require(
        set(d.scheduler._wait_guard.records) == set(remaining),
        "batch cleanup removed wrong identities",
    )
    _checkpoint(d, variant, free_call_order=calls, failed_object=d.registry.token(failed[0]))
    _check_latched(d, config, variant, "native_free")
    terminal = failed.pop()
    require(
        d.scheduler.requests.get(terminal.request_id) is terminal,
        "failed terminal object changed generation",
    )
    d.event(
        "fault_recovery",
        variant=variant,
        action="retry_free_same_current_terminal_object",
        object=d.registry.token(terminal),
        state=d.state(full=True),
    )
    d.scheduler._free_request(terminal)
    _cancel_bound(d, variant)


@contextmanager
def _update_context(d, variant, before=None, after=None):
    from vllm.v1.core.sched import scheduler as native_scheduler

    original = native_scheduler.record_function_or_nullcontext

    def factory(*args, **kwargs):
        name = args[0] if args else kwargs.get("name")
        if name != "schedule: update_after_schedule":
            return original(*args, **kwargs)

        @contextmanager
        def boundary():
            with original(*args, **kwargs) as value:
                if before is not None:
                    before()
                yield value
            if after is not None:
                after()

        return boundary()

    with _patched(native_scheduler, "record_function_or_nullcontext", factory):
        yield


def _schedule_failure(d, config, variant):
    from m2b_harness import expect_error, req

    request = req(config, "S", range(1000, 1033))
    d.add(request)
    record = d.scheduler._wait_guard.require_record(request)

    def fail():
        d.event(
            "fault_injection",
            variant=variant,
            site="profiling_context_before_update"
            if variant == "schedule_before_update_error"
            else "profiling_context_exit_after_update",
            object=d.registry.token(request),
            state=d.state(full=True),
        )
        raise InjectedNativeFailure(variant)

    before = fail if variant == "schedule_before_update_error" else None
    after = fail if variant == "schedule_after_update_error" else None
    with _update_context(d, variant, before=before, after=after):
        expect_error(d, lambda: d.schedule(variant), variant, expected=InjectedNativeFailure)
    require(
        d.scheduler._wait_guard.require_record(request) is record,
        "failed native schedule cleared first-wait record",
    )
    require(
        not d.pending and not d.dispatches, "unreturned native output was counted as dispatched"
    )
    if before is not None:
        require(
            request.num_in_flight_tokens == 0, "before-update fault already advanced native tokens"
        )
    else:
        require(
            request.num_in_flight_tokens > 0,
            "after-update fault did not exercise partial native counters",
        )
    _checkpoint(d, variant, staged_unreturned_intervals=list(d.pending_intervals))
    _check_latched(d, config, variant, "native_schedule")
    d.pending_intervals.clear()  # Observer-only tentative intervals, never native output.
    _cancel_bound(d, variant)


def _preempt_failure(d, config, variant):
    from m2b_harness import expect_error, req
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    request = req(config, "P", range(1000, 1033))
    d.add(request)
    record = d.scheduler._wait_guard.require_record(request)

    def reset_before_first_update():
        d.event(
            "fault_injection",
            variant=variant,
            site="explicit_native_reset_after_admission_before_final_update",
            object=d.registry.token(request),
            natural_preemption=False,
            state=d.state(full=True),
        )
        d.scheduler.reset_prefix_cache(reset_running_requests=True)
        raise AssertionError("injected preemption unexpectedly returned")

    if variant == "preempt_free_error":
        owner, name = d.scheduler, "_free_request_blocks"

        def failure(target):
            require(target is request, "preempt fault targeted wrong actual object")
            d.event(
                "fault_injection",
                variant=variant,
                site="preempt_before_native_block_free",
                object=d.registry.token(target),
                state=d.state(full=True),
            )
            raise InjectedNativeFailure(variant)

    else:
        owner, name = Scheduler, "_preempt_request"
        original = Scheduler._preempt_request

        def failure(instance, target, *args, **kwargs):
            result = original(instance, target, *args, **kwargs)
            if instance is d.scheduler:
                d.event(
                    "fault_injection",
                    variant=variant,
                    site="native_preempt_transition_completed_before_return",
                    object=d.registry.token(target),
                    state=d.state(full=True),
                )
                raise InjectedNativeFailure(variant)
            return result

    with (
        _patched(owner, name, failure),
        _update_context(d, variant, before=reset_before_first_update),
    ):
        expect_error(d, lambda: d.schedule(variant), variant, expected=InjectedNativeFailure)
    require(not d.pending and not d.dispatches, "preempt failure counted an unreturned output")
    if variant == "preempt_free_error":
        require(
            request.status is RequestStatus.RUNNING and request.num_preemptions == 0,
            "preempt failure unexpectedly transitioned",
        )
        require(
            d.scheduler._wait_guard.require_record(request) is record,
            "failed preemption removed original age",
        )
    else:
        require(
            request.status is RequestStatus.PREEMPTED and request.num_preemptions == 1,
            "post-transition failure did not perform native preempt",
        )
        require(
            request not in d.scheduler._wait_guard.records,
            "preempted object retained fresh protection",
        )
    _checkpoint(d, variant, natural_preemption=False)
    _check_latched(d, config, variant, "native_preempt")
    _cancel_bound(d, variant)


def _weakref_release(d, config, variant):
    from m2b_harness import cancel_all, expect_error, req

    from vllm_prefix_scheduler.wait_guard import WaitGuardError

    def exercise_actual_request():
        request = req(config, "weak_request", range(1000, 1033))
        reference = weakref.ref(request)
        token = d.registry.token(request)
        d.clock.error = "private clock failure for actual Request lifetime probe"
        expect_error(d, lambda: d.add(request), variant, expected=WaitGuardError)
        d.clock.error = None
        require(
            d.scheduler.requests.get(request.request_id) is request,
            "clock fault did not follow actual native admission",
        )
        require(not d.scheduler._wait_guard.records, "bad enqueue clock registered actual Request")
        d.event(
            "weakref_probe",
            object=token,
            label="before_release",
            alive=reference() is not None,
            owned_by_guard=request in d.scheduler._wait_guard.records,
            native_bound=True,
            observer_retained=True,
        )
        cancel_all(d)
        require(request.is_finished(), "actual weakref target was not natively terminated")
        require(
            request.request_id not in d.scheduler.requests, "native map retained weakref target"
        )
        for key in [key for key, item in d.requests.items() if item is request]:
            del d.requests[key]
        for key in [key for key, item in d.bound_objects.items() if item is request]:
            del d.bound_objects[key]
        released = []
        for index, item in enumerate(d.registry.objects):
            if item is request:
                # Keep its token position reserved, rather than renumbering history.
                d.registry.objects[index] = None
                released.append(f"object:{index}")
        require(released == [token], "observer did not relinquish exactly the actual target")
        d.event(
            "fault_recovery",
            variant=variant,
            action="release_observer_history_after_native_termination",
            object=token,
            retained_local_owner=True,
            state=d.state(full=True),
        )
        return reference, token

    reference, token = exercise_actual_request()
    gc.collect()
    require(
        reference() is None,
        "faulted guard retained actual Request after native and observer cleanup",
    )
    require(
        d.scheduler._wait_guard.fault_code == "clock_provider", "weakref fixture lost fault state"
    )
    d.event(
        "weakref_probe",
        object=token,
        label="after_release",
        alive=reference() is not None,
        owned_by_guard=False,
        native_bound=False,
        observer_retained=False,
    )
    _checkpoint(d, variant)


def run_fault_case(d, config, variant):
    require(variant in FAULT_VARIANTS, "unknown native fault leaf")
    if variant.startswith("add_failure_"):
        _add_failure(d, config, variant)
    elif variant == "generator_raise":
        _generator_raise(d, config, variant)
    elif variant == "batch_free_error":
        _batch_free_failure(d, config, variant)
    elif variant.startswith("schedule_"):
        _schedule_failure(d, config, variant)
    elif variant.startswith("preempt_"):
        _preempt_failure(d, config, variant)
    else:
        _weakref_release(d, config, variant)
