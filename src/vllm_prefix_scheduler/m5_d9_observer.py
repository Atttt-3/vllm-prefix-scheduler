"""Single-arm, bounded two-window observation of the pinned native FIFO."""

import json
import time

from vllm_prefix_scheduler.m5_d6_observer import (
    D6_BATCH_LIMIT,
    D6_EVENT_LIMIT,
    ROW_KINDS,
    _bind_native_core,
    _Trace,
)
from vllm_prefix_scheduler.m5_d7_observer import _DelayedTrace
from vllm_prefix_scheduler.m5_observer import M5WorkerExtension

D9_WINDOWS_NS = ((0, 3_000_000_000), (6_000_000_000, 8_500_000_000))
D9_TOKEN_LIMIT = 60 * 16


def write_d9_trace(path, trace):
    """Encode bounded D9 rows once, then write them after the measured drain."""
    payload = json.dumps(trace, separators=(",", ":"), allow_nan=False)
    with path.open("x") as stream:
        stream.write(payload + "\n")


class D9WorkerExtension(M5WorkerExtension):
    def m5_d9_arm(self, t0_ns, windows_ns):
        if hasattr(self, "_m5_d9"):
            raise RuntimeError("D9 worker already armed")
        core, future_type, output_type = _bind_native_core(self, "D9")
        self._m5_d9 = _DualTrace(
            core, self.model_runner, future_type, output_type, t0_ns, windows_ns
        )
        return {"armed_ns": self._m5_d9.armed_ns, "t0_ns": t0_ns, "windows_ns": windows_ns}

    def m5_d9_export(self):
        return self._m5_d9.export()


class _DualTrace(_DelayedTrace):
    def __init__(self, core, runner, future_type, output_type, t0_ns, windows_ns):
        if type(t0_ns) is not int:
            raise ValueError("D9 origin must be a monotonic nanosecond timestamp")
        expected = tuple((t0_ns + start, t0_ns + end) for start, end in D9_WINDOWS_NS)
        if tuple(map(tuple, windows_ns)) != expected or any(
            type(stamp) is not int for pair in windows_ns for stamp in pair
        ):
            raise ValueError("D9 windows differ from the frozen intervals")
        self.t0_ns = t0_ns
        self.windows = expected
        self.window_index = 0
        self.boundaries = [None, None]
        self.carries = [[], []]
        self.boundary = None
        self.carry_in = self.carries[0]
        self.late_row_offset = None
        self.early_batch_count = 0
        self.generations = {}
        self.ordinals = {}
        self.packet_count = 0
        self.untracked = [None] * D9_TOKEN_LIMIT
        self.untracked_count = 0
        start, end = expected[0]
        _Trace.__init__(
            self,
            core,
            runner,
            future_type,
            output_type,
            start,
            end,
            window_ns=end - start,
            arm_deadline_ns=t0_ns,
        )

    def at_event_limit(self):
        return self.used + self.untracked_count + sum(map(len, self.carries)) >= D6_EVENT_LIMIT

    def append(self, kind, row):
        if self.at_event_limit() or (
            kind == "batches"
            and self.early_batch_count + self.counts["batches"] + sum(map(len, self.carries))
            >= D6_BATCH_LIMIT
        ):
            self.overflow = True
            return False
        return _Trace.append(self, kind, row)

    def add_request(self, request, *args, **kwargs):
        start = time.monotonic_ns()
        result = self.add_original(request, *args, **kwargs)
        end = time.monotonic_ns()
        if start >= self.t0_ns:
            self.observe("enqueue", self.record_enqueue, request, start, end)
        return result

    def advance_window(self, now):
        if self.window_index == 0 and now >= self.end_ns and not self.pending:
            self.late_row_offset = self.used
            self.early_batch_count = self.counts["batches"]
            self.counts["steps"] = self.counts["batches"] = 0
            self.window_index = 1
            self.start_ns, self.end_ns = self.windows[1]
            self.boundary = None
            self.carry_in = self.carries[1]

    def process_input(self, *args, **kwargs):
        self.advance_window(time.monotonic_ns())
        return _Trace.process_input(self, *args, **kwargs)

    def process_loop(self, *args, **kwargs):
        now = time.monotonic_ns()
        if self.window_index == 0 and now >= self.windows[1][0] and self.pending:
            self.error("boundary", "D9 early FIFO did not close before the late window")
        self.advance_window(now)
        if self.boundary is None and self.inside(now):
            self.boundary = {"at_ns": now, "state": None}
            self.boundaries[self.window_index] = self.boundary
            self.observe("carry_in", self.bind_carry_in)
            if self.window_index == 0 and self.carry_in:
                self.error("carry_in", "D9 early window did not begin with an empty native FIFO")
        if not self.inside(now) and not self.pending:
            return self.loop_original(*args, **kwargs)
        return _Trace.process_loop(self, *args, **kwargs)

    def step(self, *args, **kwargs):
        now = time.monotonic_ns()
        if self.boundary is None or (not self.inside(now) and not self.pending):
            return self.step_original(*args, **kwargs)
        return self.record_step(now, *args, **kwargs)

    def schedule(self, *args, **kwargs):
        if self.current_step is None:
            return self.schedule_original(*args, **kwargs)
        return _Trace.schedule(self, *args, **kwargs)

    def consume(self, output, result, *args, **kwargs):
        if id(output) in self.pending:
            return _Trace.consume(self, output, result, *args, **kwargs)
        packets = self.consume_original(output, result, *args, **kwargs)
        if packets:
            self.observe("untracked_packet", self.record_untracked, packets)
        return packets

    def next_ordinal(self, identity, generation, tokens):
        if self.packet_count >= D9_TOKEN_LIMIT:
            self.overflow = True
            raise RuntimeError("D9 real packet count exceeds the frozen denominator")
        ordinal = super().next_ordinal(identity, generation, tokens)
        self.packet_count += 1
        return ordinal

    def record_untracked(self, packets):
        group = "tail_tokens" if self.window_index == 1 and self.boundary else "gap_tokens"
        for packet in packets.values():
            for item in packet.outputs:
                if not item.new_token_ids:
                    continue
                if self.at_event_limit() or self.untracked_count >= D9_TOKEN_LIMIT:
                    self.overflow = True
                    return
                identity = item.request_id
                generation = self.generations[identity]
                ordinal = self.next_ordinal(identity, generation, item.new_token_ids)
                self.untracked[self.untracked_count] = (
                    group,
                    {
                        "request_id": identity,
                        "generation": generation,
                        "ordinal": ordinal,
                        "token_id": item.new_token_ids[0],
                        "packet_ns": round(packet.timestamp * 1_000_000_000),
                    },
                )
                self.untracked_count += 1

    def export(self):
        exported = time.monotonic_ns()
        self.unpatch()
        drained = {
            "pending": len(self.pending),
            "native_fifo": len(self.core.batch_queue),
            "sampling": int(self.to_sample is not None),
            "has_requests": self.core.scheduler.has_requests(),
        }
        if exported < self.windows[-1][1] or any(drained.values()):
            self.error("export", "D9 capture did not reach a drained post-window boundary")
        windows = [{kind: [] for kind in ROW_KINDS} for _ in self.windows]
        for index, (kind, row) in enumerate(self.storage[: self.used]):
            if kind == "enqueues":
                for (_, end), window in zip(self.windows, windows, strict=True):
                    if row["start_ns"] < end:
                        window[kind].append(row)
            else:
                late = self.late_row_offset is not None and index >= self.late_row_offset
                window = windows[1] if late else windows[0]
                window[kind].append(row)
        packets = {"gap_tokens": [], "tail_tokens": []}
        for group, row in self.untracked[: self.untracked_count]:
            packets[group].append(row)
        early_packets = [
            {
                **{key: item[key] for key in ("request_id", "generation", "ordinal", "packet_ns")},
                "token_id": item["token_ids"][0],
            }
            for batch in windows[0]["batches"]
            for item in batch["output_counts"]
            if "ordinal" in item
        ]
        prefix = [*early_packets, *packets["gap_tokens"]]
        for window, bounds, boundary, carry, prefix_rows in zip(
            windows, self.windows, self.boundaries, self.carries, ([], prefix), strict=True
        ):
            if boundary is None or boundary["state"] is None:
                self.error("boundary", "D9 window has no native boundary")
            if not window["enqueues"] or not any(row["scheduled"] for row in window["batches"]):
                self.error("export", "D9 window lacks native enqueues or positive batches")
            if [row["step_id"] for row in window["steps"]] != [
                row["step_id"] for row in window["loops"]
            ]:
                self.error("export", "D9 engine ticks do not bind native loop calls")
            window.update(
                t0_ns=self.t0_ns,
                start_ns=bounds[0],
                end_ns=bounds[1],
                boundary=boundary,
                carry_in=carry,
                preroll_tokens=prefix_rows,
            )
        shared = {
            "schema_version": 1,
            "status": "incomplete" if self.errors or self.overflow else "complete",
            "overflow": self.overflow,
            "errors": self.errors,
            "armed_ns": self.armed_ns,
            "export_ns": exported,
        }
        return {
            **shared,
            "t0_ns": self.t0_ns,
            "drained": drained,
            "windows": [{**shared, **window} for window in windows],
            **packets,
        }
