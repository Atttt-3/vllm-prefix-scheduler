"""Delayed D7 observer with an actual native packet prefix and FIFO carry-in."""

import time

from vllm_prefix_scheduler.m5_d6_observer import (
    D6_BATCH_LIMIT,
    D6_EVENT_LIMIT,
    D6_REQUEST_LIMIT,
    _bind_native_core,
    _Trace,
)
from vllm_prefix_scheduler.m5_observer import M5WorkerExtension

D7_START_OFFSET_NS = 3_500_000_000
D7_END_OFFSET_NS = 7_500_000_000
D7_PREROLL_TOKEN_LIMIT = 64 * 16


class D7WorkerExtension(M5WorkerExtension):
    def m5_d7_arm(self, t0_ns, start_ns, end_ns):
        if hasattr(self, "_m5_d7"):
            raise RuntimeError("D7 worker already armed")
        core, future_type, output_type = _bind_native_core(self, "D7")
        self._m5_d7 = _DelayedTrace(
            core, self.model_runner, future_type, output_type, t0_ns, start_ns, end_ns
        )
        return {
            "armed_ns": self._m5_d7.armed_ns,
            "t0_ns": t0_ns,
            "start_ns": start_ns,
            "end_ns": end_ns,
        }

    def m5_d7_export(self):
        return self._m5_d7.export()


class _DelayedTrace(_Trace):
    def __init__(self, core, runner, future_type, output_type, t0_ns, start_ns, end_ns):
        if (
            any(type(stamp) is not int for stamp in (t0_ns, start_ns, end_ns))
            or start_ns - t0_ns != D7_START_OFFSET_NS
            or end_ns - t0_ns != D7_END_OFFSET_NS
        ):
            raise ValueError("D7 window differs from the frozen delayed interval")
        self.t0_ns = t0_ns
        self.boundary = None
        self.carry_in = []
        self.preroll = [None] * D7_PREROLL_TOKEN_LIMIT
        self.preroll_count = 0
        self.generations = {}
        self.ordinals = {}
        super().__init__(
            core,
            runner,
            future_type,
            output_type,
            start_ns,
            end_ns,
            window_ns=D7_END_OFFSET_NS - D7_START_OFFSET_NS,
            arm_deadline_ns=t0_ns,
        )

    def append(self, kind, row):
        if self.at_event_limit() or (
            kind == "batches" and self.counts["batches"] + len(self.carry_in) >= D6_BATCH_LIMIT
        ):
            self.overflow = True
            return False
        return super().append(kind, row)

    def at_event_limit(self):
        return self.used + self.preroll_count + len(self.carry_in) >= D6_EVENT_LIMIT

    def add_request(self, request, *args, **kwargs):
        start = time.monotonic_ns()
        result = self.add_original(request, *args, **kwargs)
        end = time.monotonic_ns()
        if self.t0_ns <= start < self.end_ns:
            self.observe("enqueue", self.record_enqueue, request, start, end)
        return result

    def record_enqueue(self, request, start, end):
        identity = request.request_id
        if identity in self.generations:
            raise RuntimeError("D7 repeated native request identity")
        if len(self.generations) >= D6_REQUEST_LIMIT:
            self.overflow = True
            return
        super().record_enqueue(request, start, end)
        self.generations[identity] = self.registry.binding(request)["generation"]

    def process_loop(self, *args, **kwargs):
        if self.boundary is None:
            now = time.monotonic_ns()
            if now < self.start_ns:
                return self.loop_original(*args, **kwargs)
            if now >= self.end_ns:
                self.unpatch()
                return self.loop_original(*args, **kwargs)
            self.boundary = {"at_ns": now, "state": None}
            self.observe("carry_in", self.bind_carry_in)
        return super().process_loop(*args, **kwargs)

    def bind_carry_in(self):
        # EngineCore returns each native step with at most one FIFO entry.
        # Do not call result/done, or infer launch times of this existing batch.
        if len(self.core.batch_queue) > 1 or self.pending or self.to_sample:
            raise RuntimeError("D7 carry-in is not a native FIFO boundary")
        for future, output, _ in self.core.batch_queue:
            if self.at_event_limit():
                self.overflow = True
                return
            if output.pending_structured_output_tokens:
                raise RuntimeError("D7 carry-in has deferred native sampling")
            rows = [
                {"request_id": identity, "generation": self.generations[identity], "tokens": count}
                for identity, count in output.num_scheduled_tokens.items()
            ]
            if sum(row["tokens"] for row in rows) != output.total_num_scheduled_tokens:
                raise RuntimeError("D7 carry-in scheduled count differs")
            row = {
                "batch_id": 0,
                "scheduled": rows,
                "future_result_start_ns": None,
                "future_result_end_ns": None,
                "materialize_start_ns": None,
                "materialize_end_ns": None,
                "consume_start_ns": None,
                "consume_end_ns": None,
                "consume_before": None,
                "consume_after": None,
                "materialized_counts": [],
                "output_counts": [],
            }
            record = {"output": output, "row": row, "result": None}
            self.bind_future(future, record)
            self.pending[id(output)] = record
            self.carry_in.append(row)
        self.boundary["state"] = self.inventory()

    def step(self, *args, **kwargs):
        if self.boundary is None:
            return self.step_original(*args, **kwargs)
        return super().step(*args, **kwargs)

    def schedule(self, *args, **kwargs):
        if self.boundary is None:
            return self.schedule_original(*args, **kwargs)
        return super().schedule(*args, **kwargs)

    def consume(self, output, result, *args, **kwargs):
        if self.boundary is None:
            packets = self.consume_original(output, result, *args, **kwargs)
            self.observe("preroll", self.record_preroll, packets)
            return packets
        return super().consume(output, result, *args, **kwargs)

    def next_ordinal(self, identity, generation, tokens):
        if generation != self.generations.get(identity) or len(tokens) != 1:
            raise RuntimeError("D7 native packet generation or one-token scope differs")
        count = self.ordinals.get(identity, 0) + 1
        if count > 16:
            raise RuntimeError("D7 native packet ordinal exceeds the frozen response length")
        self.ordinals[identity] = count
        return count

    def record_preroll(self, packets):
        for packet in packets.values():
            for item in packet.outputs:
                if not item.new_token_ids:
                    continue
                if self.preroll_count >= D7_PREROLL_TOKEN_LIMIT or self.at_event_limit():
                    self.overflow = True
                    return
                identity = item.request_id
                generation = self.generations[identity]
                ordinal = self.next_ordinal(identity, generation, item.new_token_ids)
                self.preroll[self.preroll_count] = {
                    "request_id": identity,
                    "generation": generation,
                    "ordinal": ordinal,
                    "token_id": item.new_token_ids[0],
                    "packet_ns": round(packet.timestamp * 1_000_000_000),
                }
                self.preroll_count += 1

    def record_consume(self, record, result, packets):
        row = record["row"]
        super().record_consume(record, result, packets, allow_missing_runner=row["batch_id"] == 0)
        for item in row["output_counts"]:
            item["ordinal"] = self.next_ordinal(
                item["request_id"], item["generation"], item["token_ids"]
            )

    def export(self):
        if self.boundary is None or self.boundary["state"] is None:
            self.error("boundary", "D7 capture has no native carry-in boundary")
        return {
            **super().export(),
            "t0_ns": self.t0_ns,
            "boundary": self.boundary,
            "carry_in": self.carry_in,
            "preroll_tokens": self.preroll[: self.preroll_count],
        }
