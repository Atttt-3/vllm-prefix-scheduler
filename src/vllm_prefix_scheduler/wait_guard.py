"""Private object-bound wait accounting, independent of vLLM and wall clocks.

Only guarded schedulers own this state. Request keys must have identity equality
and hashing (as the pinned native Request does). No completed-object history or
exception objects are retained here.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


class WaitGuardError(RuntimeError):
    """The guarded instance cannot safely accept or schedule more requests."""


@dataclass(frozen=True, slots=True)
class WaitRecord:
    enqueued_ns: int
    sequence: int


def wait_threshold_ns(max_wait_ms: int | float) -> int:
    """Ceil the exact accepted number of milliseconds to integral nanoseconds.

    In particular, binary float 0.1 is slightly above decimal 0.1 and therefore
    requires 100001 ns. Arbitrarily large Python integers never become floats.
    """
    if (
        type(max_wait_ms) not in (int, float)
        or max_wait_ms < 0
        or (type(max_wait_ms) is float and not math.isfinite(max_wait_ms))
    ):
        raise ValueError("max_wait_ms must be a finite nonnegative int or float (not bool)")
    numerator, denominator = (
        max_wait_ms.as_integer_ratio() if type(max_wait_ms) is float else (max_wait_ms, 1)
    )
    return (numerator * 1_000_000 + denominator - 1) // denominator


class WaitGuard:
    def __init__(self, max_wait_ms: int | float, clock: Callable[[], int]):
        self.threshold_ns = wait_threshold_ns(max_wait_ms)
        self._clock = clock
        self.records: dict[object, WaitRecord] = {}
        self.next_sequence = 0
        self.last_ns: int | None = None
        self.fault_code: str | None = None
        self.closed = False

    def ensure_active(self) -> None:
        if self.closed:
            raise WaitGuardError("wait_guard is closed; construct a new scheduler")
        if self.fault_code is not None:
            raise WaitGuardError(f"wait_guard is faulted: {self.fault_code}")

    def fail(self, code: str) -> None:
        # Never retain an exception/traceback: either can retain old Requests.
        if type(code) is not str:
            raise TypeError("wait_guard fault code must be a string")
        if self.fault_code is None:
            self.fault_code = code

    def _reject(self, code: str) -> None:
        self.fail(code)
        raise WaitGuardError(f"wait_guard: {code}")

    def read_clock(self, reason: str) -> int:
        """Read once; reason identifies enqueue/decision to private test observers."""
        self.ensure_active()
        try:
            now = self._clock()
        except BaseException as error:
            self.fail("clock_provider")
            raise WaitGuardError("wait_guard: clock_provider") from error
        if type(now) is not int:
            self._reject("clock_type")
        if self.last_ns is not None and now < self.last_ns:
            self._reject("clock_backwards")
        self.last_ns = now
        return now

    def register(self, request: object) -> WaitRecord:
        self.ensure_active()
        existing = self.records.get(request)
        if existing is not None:
            return existing
        now = self.read_clock("enqueue")
        record = WaitRecord(now, self.next_sequence)
        self.records[request] = record
        self.next_sequence += 1
        return record

    def require_record(self, request: object) -> WaitRecord:
        record = self.records.get(request)
        if record is None:
            self._reject("missing_record")
        return record

    def is_protected(self, request: object, now: int) -> bool:
        record = self.require_record(request)
        elapsed = now - record.enqueued_ns
        if elapsed < 0:
            self._reject("negative_elapsed")
        return elapsed >= self.threshold_ns

    def discard(self, request: object, reason: str) -> bool:
        """Idempotent object-only cleanup; no clock or active-state requirement."""
        return self.records.pop(request, None) is not None

    def close(self) -> None:
        self.records.clear()
        self.closed = True
