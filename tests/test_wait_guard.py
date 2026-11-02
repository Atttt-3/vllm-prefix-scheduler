"""Metadata/numeric layer only; native scheduler acceptance is separate."""

import gc
import sys
import weakref
from fractions import Fraction

import pytest

from vllm_prefix_scheduler.wait_guard import WaitGuard, WaitGuardError, wait_threshold_ns


class RequestIdentity:
    def __init__(self, request_id="same"):
        self.request_id = request_id


class Clock:
    def __init__(self, now=0):
        self.now = now
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.now


@pytest.mark.parametrize(
    "milliseconds",
    [0, -0.0, 500, 2.5, 0.1, float.fromhex("0x0.0000000000001p-1022"), sys.float_info.max, 10**400],
)
def test_exact_boundary_against_independent_fraction_oracle(milliseconds):
    threshold = wait_threshold_ns(milliseconds)
    expected = Fraction(milliseconds)
    assert type(threshold) is int
    assert Fraction(threshold, 1_000_000) >= expected
    if threshold:
        assert Fraction(threshold - 1, 1_000_000) < expected
    clock = Clock(-(10**500))
    guard = WaitGuard(milliseconds, clock)
    request = RequestIdentity()
    record = guard.register(request)
    if threshold:
        clock.now = record.enqueued_ns + threshold - 1
        assert not guard.is_protected(request, guard.read_clock("decision"))
    clock.now = record.enqueued_ns + threshold
    assert guard.is_protected(request, guard.read_clock("decision"))
    clock.now += 1
    assert guard.is_protected(request, guard.read_clock("decision"))


@pytest.mark.parametrize("value", [True, False, -1, -0.1, float("inf"), float("nan"), "500", None])
def test_invalid_thresholds(value):
    with pytest.raises(ValueError):
        wait_threshold_ns(value)


def test_equal_ticks_have_independent_sequences_and_duplicate_does_not_read_clock():
    clock = Clock(-7)
    guard = WaitGuard(500, clock)
    first, second = RequestIdentity(), RequestIdentity()
    a, b = guard.register(first), guard.register(second)
    assert (a.enqueued_ns, b.enqueued_ns) == (-7, -7)
    assert (a.sequence, b.sequence) == (0, 1)
    clock.now = 100
    assert guard.register(first) is a
    assert clock.calls == 2
    assert guard.next_sequence == 2


def test_object_cleanup_cannot_remove_same_id_new_binding():
    clock = Clock()
    guard = WaitGuard(500, clock)
    old, new = RequestIdentity(), RequestIdentity()
    guard.register(old)
    assert guard.discard(old, "terminal")
    clock.now = 10
    record = guard.register(new)
    assert not guard.discard(old, "terminal")
    assert guard.require_record(new) is record
    assert (record.enqueued_ns, record.sequence) == (10, 1)
    assert len(guard.records) == 1


class IntSubclass(int):
    pass


@pytest.mark.parametrize("value", [True, False, 0.0, None, "0", IntSubclass(0)])
def test_bad_clock_type_latches_without_record_or_sequence(value):
    clock = Clock(value)
    guard = WaitGuard(500, clock)
    with pytest.raises(WaitGuardError, match="clock_type"):
        guard.register(RequestIdentity())
    assert guard.records == {}
    assert guard.next_sequence == 0
    clock.now = 10
    with pytest.raises(WaitGuardError, match="faulted"):
        guard.register(RequestIdentity())
    assert clock.calls == 1


def test_backwards_clock_never_refreshes_or_clamps():
    clock = Clock(100)
    guard = WaitGuard(500, clock)
    request = RequestIdentity()
    record = guard.register(request)
    clock.now = 99
    with pytest.raises(WaitGuardError, match="clock_backwards"):
        guard.read_clock("decision")
    assert guard.require_record(request) is record
    assert guard.last_ns == 100
    assert guard.discard(request, "terminal")
    guard.close()
    assert guard.records == {}
    assert guard.fault_code == "clock_backwards"


def test_negative_elapsed_is_rejected_even_for_zero_threshold():
    guard = WaitGuard(0, lambda: 10)
    request = RequestIdentity()
    guard.register(request)
    with pytest.raises(WaitGuardError, match="negative_elapsed"):
        guard.is_protected(request, 9)


def test_missing_record_latches_without_late_registration():
    clock = Clock()
    guard = WaitGuard(500, clock)
    with pytest.raises(WaitGuardError, match="missing_record"):
        guard.require_record(RequestIdentity())
    assert guard.next_sequence == clock.calls == 0


def test_fault_state_does_not_retain_exception_traceback_or_request():
    def provider():
        raise RuntimeError("provider failure")

    def failed_registration():
        guard = WaitGuard(500, provider)
        request = RequestIdentity()
        reference = weakref.ref(request)
        try:
            guard.register(request)
        except WaitGuardError as error:
            assert isinstance(error.__cause__, RuntimeError)
        return guard, reference

    guard, reference = failed_registration()
    gc.collect()
    assert reference() is None
    assert guard.fault_code == "clock_provider"
    assert not any(isinstance(value, BaseException) for value in vars(guard).values())
    guard.close()
    assert guard.records == {}


def test_discard_and_close_release_owned_references_and_do_not_read_clock():
    clock = Clock()
    guard = WaitGuard(500, clock)
    old, live = RequestIdentity(), RequestIdentity()
    old_ref, live_ref = weakref.ref(old), weakref.ref(live)
    guard.register(old)
    guard.register(live)
    guard.discard(old, "first_dispatch")
    del old
    assert old_ref() is None
    guard.close()
    del live
    gc.collect()
    assert live_ref() is None
    guard.close()
    assert clock.calls == 2
    with pytest.raises(WaitGuardError, match="closed"):
        guard.register(RequestIdentity())
    with pytest.raises(WaitGuardError, match="closed"):
        guard.read_clock("decision")
    assert clock.calls == 2
