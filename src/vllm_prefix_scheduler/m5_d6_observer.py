"""Observe pinned UniProc EngineCore boundaries without extra materialization."""

import sys
import time

from vllm_prefix_scheduler.m4_observer import WeakIdentityRegistry
from vllm_prefix_scheduler.m5_observer import M5WorkerExtension

D6_WINDOW_NS = 3_000_000_000
D6_EVENT_LIMIT = 32_768
D6_BATCH_LIMIT = 1_024
D6_REQUEST_LIMIT = 64
ROW_KINDS = ("enqueues", "steps", "batches", "input_polls", "loops")


def _bind_native_core(worker, diagnostic):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture, UniProcExecutor
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

    # Resolve the exact owner from this one-time RPC stack. Startup objects
    # may be GC-frozen. No heap search or patching other engines is needed.
    frame = sys._getframe(1)
    while frame is not None and frame.f_code is not EngineCore.collective_rpc.__code__:
        frame = frame.f_back
    core = None if frame is None else frame.f_locals["self"]
    if (
        core is None
        or type(core.model_executor) is not UniProcExecutor
        or core.model_executor.driver_worker.worker is not worker
        or type(worker.model_runner) is not GPUModelRunner
        or type(core.scheduler) is not PrefixAsyncScheduler
        or core.step_fn.__func__ is not EngineCore.step_with_batch_queue
        or core.batch_queue is None
        or core.batch_queue.maxlen != 2
        or core.batch_queue
        or core.scheduler.has_requests()
        or worker.model_runner.execute_model_state is not None
        or not core.async_scheduling
        or core.is_mm_encoder_only
        or core.is_pooling_model
        or core.check_for_draft_tokens
    ):
        raise RuntimeError(f"{diagnostic} requires the idle pinned UniProc async EngineCore")
    return core, AsyncOutputFuture, AsyncOutput


class D6WorkerExtension(M5WorkerExtension):
    def m5_d6_arm(self, start_ns, end_ns):
        if hasattr(self, "_m5_d6"):
            raise RuntimeError("D6 worker already armed")
        core, future_type, output_type = _bind_native_core(self, "D6")
        self._m5_d6 = _Trace(core, self.model_runner, future_type, output_type, start_ns, end_ns)
        return {"armed_ns": self._m5_d6.armed_ns, "start_ns": start_ns, "end_ns": end_ns}

    def m5_d6_export(self):
        return self._m5_d6.export()


class _Trace:
    def __init__(
        self,
        core,
        runner,
        future_type,
        output_type,
        start_ns,
        end_ns,
        *,
        window_ns=D6_WINDOW_NS,
        arm_deadline_ns=None,
    ):
        self.armed_ns = time.monotonic_ns()
        if (
            type(start_ns) is not int
            or type(end_ns) is not int
            or not self.armed_ns < (start_ns if arm_deadline_ns is None else arm_deadline_ns)
            or end_ns - start_ns != window_ns
        ):
            raise ValueError("D6 window differs from the frozen future interval")
        self.start_ns, self.end_ns = start_ns, end_ns
        self.core, self.runner = core, runner
        self.future_type, self.output_type = future_type, output_type
        self.registry = WeakIdentityRegistry()
        self.storage = [None] * D6_EVENT_LIMIT
        self.counts = dict.fromkeys(ROW_KINDS, 0)
        self.used = 0
        self.pending = {}
        self.errors = []
        self.overflow = False
        self.current_step = None
        self.current_loop = None
        self.to_sample = None
        self.patches = []
        self.step_original = self.patch(core, "step_fn", self.step)
        self.input_original = self.patch(core, "_process_input_queue", self.process_input)
        self.loop_original = self.patch(core, "_process_engine_step", self.process_loop)
        self.add_original = self.patch(core.scheduler, "add_request", self.add_request)
        self.schedule_original = self.patch(core.scheduler, "schedule", self.schedule)
        self.consume_original = self.patch(core.scheduler, "update_from_output", self.consume)
        self.execute_original = self.patch(core.model_executor, "execute_model", self.execute)
        self.sample_original = self.patch(core.model_executor, "sample_tokens", self.sample)

    def patch(self, owner, name, wrapper):
        original = getattr(owner, name)
        self.patches.append((owner, name, name in vars(owner), original))
        setattr(owner, name, wrapper)
        return original

    def unpatch(self):
        while self.patches:
            owner, name, local, original = self.patches.pop()
            if local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)

    def error(self, operation, error):
        if len(self.errors) < 32:
            self.errors.append({"operation": operation, "error": str(error)})

    def observe(self, operation, callback, *args):
        try:
            callback(*args)
        except Exception as error:
            self.error(operation, error)

    def append(self, kind, row):
        limit = {"enqueues": D6_REQUEST_LIMIT, "batches": D6_BATCH_LIMIT}.get(kind, D6_EVENT_LIMIT)
        if self.used >= D6_EVENT_LIMIT or self.counts[kind] >= limit:
            self.overflow = True
            return False
        self.storage[self.used] = (kind, row)
        self.used += 1
        self.counts[kind] += 1
        return True

    def inside(self, stamp):
        return self.start_ns <= stamp < self.end_ns

    def process_input(self, *args, **kwargs):
        start = time.monotonic_ns()
        try:
            return self.input_original(*args, **kwargs)
        finally:
            end = time.monotonic_ns()
            if start < self.end_ns and self.start_ns <= end:
                self.append("input_polls", {"start_ns": start, "end_ns": end})

    def process_loop(self, *args, **kwargs):
        start = time.monotonic_ns()
        if start >= self.end_ns and not self.pending:
            return self.loop_original(*args, **kwargs)
        row = {"start_ns": start, "end_ns": None, "step_id": None, "model_executed": None}
        self.current_loop = row
        try:
            value = self.loop_original(*args, **kwargs)
            row["model_executed"] = value
            return value
        finally:
            row["end_ns"] = time.monotonic_ns()
            self.current_loop = None
            if row["step_id"] is not None:
                self.append("loops", row)

    def step(self, *args, **kwargs):
        start = time.monotonic_ns()
        if start >= self.end_ns and not self.pending:
            self.unpatch()
            return self.step_original(*args, **kwargs)
        if not self.inside(start) and not self.pending:
            return self.step_original(*args, **kwargs)
        row = {
            "step_id": self.counts["steps"] + 1,
            "start_ns": start,
            "end_ns": None,
            "queue_before": len(self.core.batch_queue),
            "queue_after": None,
            "scheduled_batch_id": None,
            "consumed_batch_id": None,
            "model_executed": None,
        }
        if not self.append("steps", row):
            return self.step_original(*args, **kwargs)
        if self.current_loop is not None:
            self.current_loop["step_id"] = row["step_id"]
        self.current_step = row
        try:
            result = self.step_original(*args, **kwargs)
            row["model_executed"] = result[1]
            return result
        finally:
            row["end_ns"] = time.monotonic_ns()
            row["queue_after"] = len(self.core.batch_queue)
            self.current_step = None

    def add_request(self, request, *args, **kwargs):
        start = time.monotonic_ns()
        result = self.add_original(request, *args, **kwargs)
        end = time.monotonic_ns()
        if self.inside(start):
            self.observe("enqueue", self.record_enqueue, request, start, end)
        return result

    def record_enqueue(self, request, start, end):
        if self.core.scheduler.requests.get(request.request_id) is not request:
            raise RuntimeError("D6 enqueue did not retain its native request object")
        self.append(
            "enqueues",
            {
                "request_id": request.request_id,
                "generation": self.registry.binding(request)["generation"],
                "start_ns": start,
                "end_ns": end,
            },
        )

    def schedule(self, *args, **kwargs):
        start = time.monotonic_ns()
        before = self.snapshot("schedule_before") if self.inside(start) else None
        start = time.monotonic_ns() if before is not None else start
        output = self.schedule_original(*args, **kwargs)
        end = time.monotonic_ns()
        if before is not None and self.inside(start):
            self.observe("schedule", self.record_schedule, output, start, end, before)
        return output

    def snapshot(self, operation):
        try:
            return self.inventory()
        except Exception as error:
            self.error(operation, error)
            return None

    def inventory(self):
        scheduler = self.core.scheduler
        queues = {
            "running": scheduler.running,
            "waiting": scheduler.waiting,
            "kv_holding_waiting": scheduler.kv_holding_waiting,
        }
        if sum(map(len, queues.values())) > D6_REQUEST_LIMIT:
            raise RuntimeError("D6 scheduler inventory exceeds its request bound")
        return {
            **{
                label: [[req.request_id, self.registry.binding(req)["generation"]] for req in queue]
                for label, queue in queues.items()
            },
            "pending_batch_ids": [
                self.pending[id(output)]["row"]["batch_id"] if id(output) in self.pending else None
                for _, output, _ in self.core.batch_queue
            ],
        }

    def record_schedule(self, output, start, end, before):
        if self.current_step is None or self.current_step["scheduled_batch_id"] is not None:
            raise RuntimeError("D6 schedule is not uniquely bound to an engine tick")
        if output.pending_structured_output_tokens:
            raise RuntimeError("D6 deferred sampling is outside the frozen workload")
        scheduler = self.core.scheduler
        rows = []
        for rid, count in output.num_scheduled_tokens.items():
            request = scheduler.requests[rid]
            stop = request.num_computed_tokens
            if not 0 < count <= stop:
                raise RuntimeError("D6 scheduler dispatch position is invalid")
            rows.append(
                {
                    "request_id": rid,
                    "generation": self.registry.binding(request)["generation"],
                    "tokens": count,
                    "start": stop - count,
                    "end": stop,
                    "prompt_length": request.num_prompt_tokens,
                    "preemptions": request.num_preemptions,
                }
            )
        if sum(row["tokens"] for row in rows) != output.total_num_scheduled_tokens:
            raise RuntimeError("D6 scheduler token inventory differs")
        row = {
            "batch_id": self.counts["batches"] + 1,
            "effective_active_cap": scheduler.max_num_active_reqs,
            "admission_stop_reason": "unknown",
            "schedule_start_ns": start,
            "schedule_end_ns": end,
            "schedule_before": before,
            "schedule_after": self.inventory(),
            "scheduled": rows,
            "execute_start_ns": None,
            "execute_end_ns": None,
            "sample_start_ns": None,
            "sample_end_ns": None,
            "future_result_start_ns": None,
            "future_result_end_ns": None,
            "materialize_start_ns": None,
            "materialize_end_ns": None,
            "consume_start_ns": None,
            "consume_end_ns": None,
            "consume_before": None,
            "consume_after": None,
            "runner": None,
            "materialized_counts": [],
            "output_counts": [],
        }
        if self.append("batches", row):
            self.pending[id(output)] = {"output": output, "row": row, "result": None}
            self.current_step["scheduled_batch_id"] = row["batch_id"]

    def execute(self, output, *args, **kwargs):
        record = self.pending.get(id(output))
        if record is None:
            return self.execute_original(output, *args, **kwargs)
        row = record["row"]
        row["execute_start_ns"] = time.monotonic_ns()
        future = self.execute_original(output, *args, **kwargs)
        row["execute_end_ns"] = time.monotonic_ns()
        if row["scheduled"]:
            self.to_sample = record
            self.observe("runner_input", self.runner_input, record)
        else:
            self.observe("future", self.bind_future, future, record)
        return future

    def runner_input(self, record):
        row = record["row"]
        batch = self.runner.execute_model_state.input_batch
        generation = {item["request_id"]: item["generation"] for item in row["scheduled"]}
        offsets = batch.query_start_loc_np
        ids = batch.req_ids
        effective, padded = int(batch.num_tokens), int(batch.num_tokens_after_padding)
        if (
            len(ids) != len(generation)
            or set(ids) != set(generation)
            or not 0 < effective <= padded
            or int(offsets[0]) != 0
            or int(offsets[len(ids)]) != effective
        ):
            raise RuntimeError("D6 CPU runner input does not bind the scheduled requests")
        row["runner"] = {
            "effective_tokens": effective,
            "padded_tokens": padded,
            "rows": [
                {
                    "request_id": rid,
                    "generation": generation[rid],
                    "query_start": int(offsets[i]),
                    "query_end": int(offsets[i + 1]),
                    "prefill_length": int(batch.prefill_len_np[i]),
                }
                for i, rid in enumerate(ids)
            ],
        }

    def sample(self, *args, **kwargs):
        record, self.to_sample = self.to_sample, None
        if record is None:
            return self.sample_original(*args, **kwargs)
        row = record["row"]
        row["sample_start_ns"] = time.monotonic_ns()
        future = self.sample_original(*args, **kwargs)
        row["sample_end_ns"] = time.monotonic_ns()
        self.observe("future", self.bind_future, future, record)
        return future

    def bind_future(self, future, record):
        row = record["row"]
        if "result" in vars(future):
            raise RuntimeError("D6 received an already wrapped native future")
        original_result = future.result
        if row["scheduled"]:
            if (
                type(future) is not self.future_type
                or type(future.async_output) is not self.output_type
                or future.single_value is not True
                or "get_output" in vars(future.async_output)
            ):
                raise RuntimeError("D6 expected the pinned native UniProc future")
            output = future.async_output
            original_get = output.get_output

            def get_output(*args, **kwargs):
                del output.get_output
                row["materialize_start_ns"] = time.monotonic_ns()
                try:
                    return original_get(*args, **kwargs)
                finally:
                    row["materialize_end_ns"] = time.monotonic_ns()

            output.get_output = get_output

        def result(*args, **kwargs):
            del future.result
            row["future_result_start_ns"] = time.monotonic_ns()
            try:
                value = original_result(*args, **kwargs)
            finally:
                row["future_result_end_ns"] = time.monotonic_ns()
            record["result"] = value
            if row["scheduled"]:
                self.observe("materialized_counts", self.materialized_counts, row, value)
            return value

        future.result = result

    @staticmethod
    def materialized_counts(row, output):
        generation = {item["request_id"]: item["generation"] for item in row["scheduled"]}
        if len(output.req_ids) != len(generation) or set(output.req_ids) != set(generation):
            raise RuntimeError("D6 materialized output request identity differs")
        row["materialized_counts"] = [
            {"request_id": rid, "generation": generation[rid], "generated_tokens": len(tokens)}
            for rid, tokens in zip(output.req_ids, output.sampled_token_ids, strict=True)
        ]

    def consume(self, output, result, *args, **kwargs):
        record = self.pending.pop(id(output), None)
        if record is None:
            return self.consume_original(output, result, *args, **kwargs)
        row = record["row"]
        row["consume_before"] = self.snapshot("consume_before")
        row["consume_start_ns"] = time.monotonic_ns()
        value = self.consume_original(output, result, *args, **kwargs)
        row["consume_end_ns"] = time.monotonic_ns()
        row["consume_after"] = self.snapshot("consume_after")
        self.observe("consume", self.record_consume, record, result, value)
        return value

    def record_consume(self, record, result, packets, *, allow_missing_runner=False):
        row = record["row"]
        if (
            self.current_step is None
            or self.current_step["consumed_batch_id"] is not None
            or row["future_result_end_ns"] is None
            or record["result"] is not result
            or row["scheduled"]
            and (
                row["materialize_end_ns"] is None
                or not allow_missing_runner
                and row.get("runner") is None
            )
        ):
            raise RuntimeError("D6 native consumption does not bind its FIFO result")
        self.current_step["consumed_batch_id"] = row["batch_id"]
        generation = {item["request_id"]: item["generation"] for item in row["scheduled"]}
        row["output_counts"] = [
            {
                "request_id": item.request_id,
                "generation": generation[item.request_id],
                "generated_tokens": len(item.new_token_ids),
                "token_ids": list(item.new_token_ids),
                "packet_ns": round(packet.timestamp * 1_000_000_000),
            }
            for packet in packets.values()
            for item in packet.outputs
            if item.new_token_ids
        ]

    def export(self):
        exported = time.monotonic_ns()
        self.unpatch()
        if (
            exported < self.end_ns
            or self.pending
            or self.core.batch_queue
            or self.to_sample
            or self.core.scheduler.has_requests()
        ):
            self.error("export", "D6 capture did not reach a drained post-window boundary")
        rows = {kind: [] for kind in ROW_KINDS}
        for kind, row in self.storage[: self.used]:
            rows[kind].append(row)
        if (
            not rows["enqueues"]
            or not rows["steps"]
            or not any(row["scheduled"] for row in rows["batches"])
        ):
            self.error("export", "D6 capture lacks measured requests or positive batches")
        if [row["step_id"] for row in rows["steps"]] != [row["step_id"] for row in rows["loops"]]:
            self.error("export", "D6 engine ticks do not bind native loop calls")
        return {
            "schema_version": 1,
            "status": "incomplete" if self.errors or self.overflow else "complete",
            "overflow": self.overflow,
            "errors": self.errors,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "armed_ns": self.armed_ns,
            "export_ns": exported,
            **rows,
        }
