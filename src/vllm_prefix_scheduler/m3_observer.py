"""M3-only passive scheduler subclasses for pinned vLLM 0.32.0.

Source anchors: vllm 5df63a7203cdb9328d824a27b682e77c7b9928be,
v1/engine/core.py:161-182,698-812; core/sched/scheduler.py:2026,2616.
The production scheduler and its policy are unchanged. Each overridden method
calls its parent once, retains native results/exceptions, and observes storage.
"""

import sys
from itertools import islice

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_prefix_scheduler import config, probe, scheduler, wait_guard
from vllm_prefix_scheduler import m3_events as events
from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler


def object_token(request):
    return f"{request.request_id}@{id(request):x}"


def guard_records(instance):
    guard = getattr(instance, "_wait_guard", None)
    if guard is None:
        return []
    return [
        {
            "request_id": request.request_id,
            "object_id": object_token(request),
            "enqueued_ns": record.enqueued_ns,
            "sequence": record.sequence,
        }
        for request, record in guard.records.items()
    ]


def scheduler_resources(instance):
    """Storage-only aggregate scan, used at case boundaries and shutdown only."""
    pool = instance.kv_cache_manager.block_pool
    queue = pool.free_block_queue
    free_ids, seen = [], set()
    block = queue.fake_free_list_head.next_free_block
    while block is not queue.fake_free_list_tail:
        if block is None or id(block) in seen:
            raise RuntimeError("invalid or cyclic native free queue")
        seen.add(id(block))
        free_ids.append(block.block_id)
        block = block.next_free_block
    singles = instance.kv_cache_manager.coordinator.single_type_managers
    guard = getattr(instance, "_wait_guard", None)
    requests = list(instance.requests.values())
    return {
        "requests": sorted(instance.requests),
        "waiting": [r.request_id for r in instance.waiting],
        "running": [r.request_id for r in instance.running],
        "holding": [r.request_id for r in instance.kv_holding_waiting],
        "deferred": sorted(r.request_id for r in instance.deferred_waiting),
        "finished_req_ids": sorted(instance.finished_req_ids),
        "reset_preempted_req_ids": sorted(instance.reset_preempted_req_ids),
        "inflight_prefills": sorted(r.request_id for r in instance._inflight_prefills),
        "deferred_frees": len(instance.deferred_frees),
        "request_blocks": sum(len(m.req_to_blocks) for m in singles),
        "non_null_ref_sum": sum(b.ref_cnt for b in pool.blocks if not b.is_null),
        "free_count": queue.num_free_blocks,
        "free_unique_count": len(set(free_ids)),
        "free_traversed_count": len(free_ids),
        "total_non_null_blocks": sum(not b.is_null for b in pool.blocks),
        "cached_block_count": sum(b._block_hash is not None for b in pool.blocks),
        "pending_batches": events.pending_batches(),
        "inflight_tokens": sum(r.num_in_flight_tokens for r in requests),
        "stale_tokens": sum(r.num_stale_output_tokens for r in requests),
        "output_placeholders": sum(r.num_output_placeholders for r in requests),
        "guard_records": guard_records(instance),
        "guard_closed": None if guard is None else guard.closed,
        "guard_fault_code": None if guard is None else guard.fault_code,
        "pause_state": instance.pause_state.name,
        "observation_errors": events.observation_errors(),
    }


def scheduler_checkpoint(instance, label, case_id=None):
    resources = scheduler_resources(instance)
    events.emit("scheduler_checkpoint", label=label, case_id=case_id, resources=resources)
    return resources


class _ObserverMixin:
    def __init__(self, *args, **kwargs):
        self._m3_decision = None
        super().__init__(*args, **kwargs)
        events.register_scheduler(self)
        events.observe("scheduler_init", self._m3_init_event)

    def _m3_init_event(self):
        cfg = self.vllm_config
        prefix = getattr(self, "prefix_config", None)
        modules = {
            name: events.module_source(module)
            for name, module in {
                "scheduler": scheduler,
                "config": config,
                "probe": probe,
                "wait_guard": wait_guard,
                "observer": sys.modules[__name__],
                "native_async_scheduler": sys.modules[AsyncScheduler.__module__],
            }.items()
        }
        events.emit(
            "scheduler_init",
            actual_type=events.qualified_type(self),
            mro=[events.qualified_type(cls) for cls in type(self).__mro__],
            modules=modules,
            effective_config={
                "async_scheduling": cfg.scheduler_config.async_scheduling,
                "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
                "enable_chunked_prefill": cfg.scheduler_config.enable_chunked_prefill,
                "max_num_seqs": cfg.scheduler_config.max_num_seqs,
                "max_num_active_seqs": cfg.scheduler_config.max_num_active_seqs,
                "max_num_active_reqs": self.max_num_active_reqs,
                "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
                "max_model_len": cfg.model_config.max_model_len,
                "use_v2_model_runner": cfg.use_v2_model_runner,
                "dtype": str(cfg.model_config.dtype),
                "executor_backend": cfg.parallel_config.distributed_executor_backend,
                "window": None if prefix is None else prefix.window,
                "max_wait_ms": None if prefix is None else prefix.max_wait_ms,
                "mode": "native" if prefix is None else prefix.mode,
                "block_size": self.block_size,
                "hash_block_size": self.hash_block_size,
                "num_gpu_blocks": self.kv_cache_manager.block_pool.num_gpu_blocks,
                "kv_spec_types": [
                    events.qualified_type(group.kv_cache_spec)
                    for group in self.kv_cache_config.kv_cache_groups
                ],
            },
        )

    def add_request(self, request):
        try:
            result = super().add_request(request)
        except BaseException as error:
            events.emit(
                "enqueue_error",
                request_id=request.request_id,
                error_type=events.qualified_type(error),
            )
            raise

        def record():
            guard = next(
                (r for r in guard_records(self) if r["object_id"] == object_token(request)), None
            )
            events.emit(
                "enqueue",
                request_id=request.request_id,
                object_id=object_token(request),
                prompt_token_ids=list(request.prompt_token_ids),
                num_computed_tokens=request.num_computed_tokens,
                status=request.status.name,
                bound=self.requests.get(request.request_id) is request,
                waiting=[r.request_id for r in self.waiting],
                guard=guard,
            )

        events.observe("enqueue", record)
        return result

    def schedule(self, throttle_prefills=False):
        before = events.observe("guard_before_schedule", lambda: guard_records(self))
        try:
            output = super().schedule(throttle_prefills=throttle_prefills)
        except BaseException as error:
            events.emit("schedule_error", error_type=events.qualified_type(error))
            raise

        def record():
            batch_id = events.register_batch(output)
            intervals = []
            for rid, count in output.num_scheduled_tokens.items():
                request = self.requests[rid]
                # Native _update_after_schedule has advanced computed by count.
                end = request.num_computed_tokens
                intervals.append(
                    {
                        "request_id": rid,
                        "object_id": object_token(request),
                        "start": end - count,
                        "end": end,
                        "prompt_length": request.num_prompt_tokens,
                    }
                )
            events.emit(
                "schedule_success",
                batch_id=batch_id,
                output_object=f"{id(output):x}",
                intervals=intervals,
                guard_before=before,
                guard_after=guard_records(self),
                **events.output_descriptor(output),
            )

        events.observe("schedule_success", record)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        batch_id = events.observe("consume_binding", lambda: events.batch_id_for(scheduler_output))
        events.observe(
            "update_from_output_start",
            lambda: events.emit(
                "update_from_output_start",
                batch_id=batch_id,
                **events.model_output_fields(model_runner_output),
            ),
        )
        try:
            result = super().update_from_output(scheduler_output, model_runner_output)
        except BaseException as error:
            events.emit(
                "update_from_output_error",
                batch_id=batch_id,
                error_type=events.qualified_type(error),
            )
            raise

        def record():
            outputs = [
                {
                    "request_id": output.request_id,
                    "new_token_ids": list(output.new_token_ids),
                    "finish_reason": None
                    if output.finish_reason is None
                    else output.finish_reason.name,
                }
                for group in result.values()
                for output in group.outputs
            ]
            events.emit(
                "update_from_output_success",
                batch_id=batch_id,
                engine_outputs=outputs,
                guard_after=guard_records(self),
                **events.model_output_fields(model_runner_output),
            )
            events.consume_batch(scheduler_output)

        events.observe("update_from_output_success", record)
        return result

    def _free_request(self, request, delay_free_blocks=False):
        before = events.observe("guard_before_free", lambda: guard_records(self))
        try:
            result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        except BaseException as error:
            events.emit(
                "free_request_error",
                request_id=request.request_id,
                error_type=events.qualified_type(error),
            )
            raise
        events.observe(
            "request_terminal",
            lambda: events.emit(
                "request_terminal",
                request_id=request.request_id,
                object_id=object_token(request),
                status=request.status.name,
                guard_before=before,
                guard_after=guard_records(self),
            ),
        )
        return result

    def shutdown(self):
        events.observe(
            "shutdown_before",
            lambda: events.emit("scheduler_shutdown_before", resources=scheduler_resources(self)),
        )
        try:
            result = super().shutdown()
        except BaseException as error:
            events.emit("scheduler_shutdown_error", error_type=events.qualified_type(error))
            raise
        events.observe(
            "shutdown_after",
            lambda: events.emit("scheduler_shutdown_after", resources=scheduler_resources(self)),
        )
        return result


class ObserverNativeScheduler(_ObserverMixin, AsyncScheduler):
    """Native AsyncScheduler plus passive observations, no prefix strategy."""


class ObserverPrefixScheduler(_ObserverMixin, PrefixAsyncScheduler):
    """Production strategy subclass; never recomputes scores or clock values."""

    def _candidate_window(self):
        result = super()._candidate_window()

        def record():
            events.emit(
                "candidate_window",
                waiting_head=[
                    r.request_id
                    for r in islice(
                        self.waiting, min(self.prefix_config.window + 1, len(self.waiting))
                    )
                ],
                window_limit=self.prefix_config.window,
                candidates=[
                    {
                        "request_id": r.request_id,
                        "object_id": object_token(r),
                        "status": r.status.name,
                        "num_computed_tokens": r.num_computed_tokens,
                        "num_in_flight_tokens": r.num_in_flight_tokens,
                        "num_output_tokens": r.num_output_tokens,
                        "num_output_placeholders": r.num_output_placeholders,
                        "num_preemptions": r.num_preemptions,
                    }
                    for r in result
                ],
            )

        events.observe("candidate_window", record)
        return result

    def _sort_window(self, candidates):
        self._m3_decision = {"scores": {}, "now_ns": None}
        try:
            result = super()._sort_window(candidates)
        except BaseException as error:
            events.emit("decision_error", error_type=events.qualified_type(error))
            raise
        else:

            def record():
                guard = getattr(self, "_wait_guard", None)
                events.emit(
                    "decision",
                    before=[r.request_id for r in candidates],
                    after=[r.request_id for r in result],
                    objects={r.request_id: object_token(r) for r in candidates},
                    scores=dict(self._m3_decision["scores"]),
                    now_ns=self._m3_decision["now_ns"],
                    threshold_ns=None if guard is None else guard.threshold_ns,
                    records={r["request_id"]: r for r in guard_records(self)},
                )

            events.observe("decision", record)
            return result
        finally:
            self._m3_decision = None

    def _score_request(self, request):
        result = super()._score_request(request)
        if self._m3_decision is not None:
            self._m3_decision["scores"][request.request_id] = result
        return result

    def _wait_clock_ns(self):
        result = super()._wait_clock_ns()
        if self._m3_decision is not None:
            self._m3_decision["now_ns"] = result
        return result
