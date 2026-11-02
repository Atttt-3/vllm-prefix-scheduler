"""M4-only passive state observations on the pinned native async execution chain.

No clocks, queue orders, model outputs, allocations or cancellation are changed.
The M3 runner/FIFO observer is reused unchanged; M4 adds weak object generations.
"""

import gc
import hashlib
import json
import os
import weakref
from collections import deque


class WeakIdentityRegistry:
    """Identity-safe weak keys; one scalar counter, no historical request ledger."""

    def __init__(self):
        self._objects = weakref.WeakKeyDictionary()
        self.next_generation = 0

    def binding(self, request):
        found = self._objects.get(request)
        if found is None:
            generation = self.next_generation
            self.next_generation += 1
            found = {"object_id": f"object:{generation}", "generation": generation}
            self._objects[request] = found
        return dict(found)

    def __len__(self):
        return len(self._objects)


def request_state(request, registry, instance=None):
    guard = getattr(instance, "_wait_guard", None)
    record = None if guard is None else guard.records.get(request)
    return {
        "request_id": request.request_id,
        **registry.binding(request),
        "status": request.status.name,
        "prompt_length": request.num_prompt_tokens,
        "kv_cache_report_mode": request.kv_cache_report_mode,
        **{
            name: getattr(request, name)
            for name in (
                "num_computed_tokens",
                "num_in_flight_tokens",
                "num_stale_output_tokens",
                "num_output_placeholders",
                "num_output_tokens",
                "num_preemptions",
                "is_prefill_chunk",
                "drop_stale_output",
                "next_decode_eligible_step",
                "last_sched_seq",
            )
        },
        "guard": None
        if record is None
        else {"enqueued_ns": record.enqueued_ns, "sequence": record.sequence},
    }


def guard_state(instance, registry):
    guard = getattr(instance, "_wait_guard", None)
    return (
        []
        if guard is None
        else [
            {
                "request_id": r.request_id,
                **registry.binding(r),
                "enqueued_ns": rec.enqueued_ns,
                "sequence": rec.sequence,
            }
            for r, rec in guard.records.items()
        ]
    )


def storage_snapshot(instance, request):
    """Serialize stored fields only. No draining, allocating or stats getter API.

    Native request/model configurations contain immutable functions and classes;
    those are represented by qualified identity, not invoked. Tensor contents are
    excluded: this snapshot covers scheduler/cache metadata, not model execution.
    """
    import enum
    import types

    seen = {}

    def visit(value):
        if value is None or type(value) in (str, int, bool, float):
            return value
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, enum.Enum):
            return {"enum": f"{type(value).__module__}.{type(value).__name__}", "name": value.name}
        if isinstance(value, (types.FunctionType, types.MethodType, type)):
            return {"callable": f"{value.__module__}.{value.__qualname__}"}
        if type(value).__module__ == "torch":
            return {"torch": str(value)}
        oid = id(value)
        if oid in seen:
            return {"ref": seen[oid]}
        seen[oid] = len(seen)
        identity = {"identity": oid, "type": f"{type(value).__module__}.{type(value).__name__}"}
        if isinstance(value, dict):
            return {**identity, "items": [[visit(k), visit(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple, deque)):
            return {**identity, "items": [visit(v) for v in value]}
        if isinstance(value, (set, frozenset)):
            return {**identity, "items": [visit(v) for v in sorted(value, key=id)]}
        if isinstance(value, weakref.ReferenceType):
            return {**identity, "target": None if value() is None else id(value())}
        fields = dict(getattr(value, "__dict__", {}))
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name not in {"__dict__", "__weakref__"}:
                    try:
                        fields[name] = object.__getattribute__(value, name)
                    except AttributeError:
                        pass
        if hasattr(type(value), "__struct_fields__"):
            fields.update(
                {
                    name: object.__getattribute__(value, name)
                    for name in type(value).__struct_fields__
                }
            )
        return {**identity, "fields": {k: visit(v) for k, v in sorted(fields.items())}}

    manager = instance.kv_cache_manager
    # A single traversal retains cross-root aliasing and compares object identity.
    roots = {
        "request": request,
        "manager": manager,
        "pool": manager.block_pool,
        "metrics": instance.kv_metrics_collector,
        "waiting": instance.waiting,
        "running": instance.running,
        "holding": instance.kv_holding_waiting,
        "deferred": instance.deferred_waiting,
    }
    return {key: visit(value) for key, value in roots.items()}


def _define_native_classes():
    from vllm_prefix_scheduler import m3_events as events
    from vllm_prefix_scheduler.m3_observer import (
        ObserverNativeScheduler,
        ObserverPrefixScheduler,
    )
    from vllm_prefix_scheduler.m3_worker import M3WorkerExtension
    from vllm_prefix_scheduler.probe import reusable_prefix_tokens

    active_case = None
    active_metadata = {}

    def emit(event, **fields):
        if "case_id" not in fields:
            fields["case_id"] = active_case
        fields = {**active_metadata, **fields}
        events.emit(
            event,
            task_id="M4-scheduler-stability",
            phase=os.environ.get("M4_PHASE", "formal"),
            **fields,
        )

    class M4Mixin:
        def __init__(self, *args, **kwargs):
            self._m4_registry = WeakIdentityRegistry()
            self._m4_pending = {}
            super().__init__(*args, **kwargs)
            original_publish = self.kv_event_publisher.publish

            def publish(batch):
                result = original_publish(batch)
                events.observe(
                    "m4_kv_events",
                    lambda: emit(
                        "m4_kv_events",
                        layer="gpu",
                        event_count=len(batch.events),
                        event_types=[type(event).__name__ for event in batch.events],
                        stored_blocks=sum(
                            len(event.block_hashes)
                            for event in batch.events
                            if type(event).__name__ == "BlockStored"
                        ),
                    ),
                )
                return result

            self.kv_event_publisher.publish = publish
            original_stats = self.kv_cache_manager.record_prefix_cache_stats

            def record_stats(request, num_hits):
                result = original_stats(request, num_hits)
                events.observe(
                    "m4_prefix_stats_activity",
                    lambda: emit(
                        "m4_prefix_stats_activity",
                        layer="gpu",
                        request_id=request.request_id,
                        hit_tokens=num_hits,
                        prompt_length=request.num_prompt_tokens,
                        full_report=request.kv_cache_report_mode == "full",
                    ),
                )
                return result

            self.kv_cache_manager.record_prefix_cache_stats = record_stats
            emit(
                "m4_native_identity",
                layer="gpu",
                actual_type=events.qualified_type(self),
                source=events.module_source(__import__(__name__, fromlist=["_"])),
            )

        def _m4_states(self):
            return [request_state(r, self._m4_registry, self) for r in self.requests.values()]

        def _m4_pending_rows(self):
            return list(self._m4_pending.values())

        def add_request(self, request):
            nonlocal active_case
            result = super().add_request(request)
            active_case = request.request_id.split("::", 1)[0]
            events.observe(
                "m4_request_bound",
                lambda: emit(
                    "m4_request_bound",
                    layer="gpu",
                    bound=self.requests.get(request.request_id) is request,
                    request={
                        **request_state(request, self._m4_registry, self),
                        "prompt_token_ids": list(request.prompt_token_ids),
                        "token_ids_sha256": hashlib.sha256(
                            json.dumps(list(request.prompt_token_ids)).encode()
                        ).hexdigest(),
                    },
                ),
            )
            return result

        def schedule(self, throttle_prefills=False):
            before = events.observe("m4_guard_before", lambda: guard_state(self, self._m4_registry))
            output = super().schedule(throttle_prefills=throttle_prefills)

            def record():
                batch_id = events.batch_id_for(output)
                intervals = []
                for rid, count in output.num_scheduled_tokens.items():
                    r = self.requests[rid]
                    intervals.append(
                        {
                            "request_id": rid,
                            **self._m4_registry.binding(r),
                            "start": r.num_computed_tokens - count,
                            "end": r.num_computed_tokens,
                            "prompt_length": r.num_prompt_tokens,
                        }
                    )
                row = {
                    "batch_id": batch_id,
                    "intervals": intervals,
                    "finished_req_ids": sorted(output.finished_req_ids),
                    "preempted_req_ids": sorted(output.preempted_req_ids or ()),
                }
                self._m4_pending[batch_id] = row
                emit(
                    "m4_schedule",
                    layer="gpu",
                    **row,
                    num_scheduled_tokens=dict(output.num_scheduled_tokens),
                    states=self._m4_states(),
                    guard_before=before,
                    guard_after=guard_state(self, self._m4_registry),
                )

            events.observe("m4_schedule", record)
            return output

        def update_from_output(self, scheduler_output, model_runner_output):
            batch_id = events.observe(
                "m4_consume_binding", lambda: events.batch_id_for(scheduler_output)
            )
            binding = self._m4_pending.get(batch_id)
            events.observe(
                "m4_consume_start",
                lambda: emit(
                    "m4_consume_start",
                    layer="gpu",
                    batch_id=batch_id,
                    intervals=None if binding is None else binding["intervals"],
                    states=self._m4_states(),
                ),
            )
            result = super().update_from_output(scheduler_output, model_runner_output)

            def record():
                if binding is None:
                    raise RuntimeError("missing M4 schedule-time object binding")
                if next(iter(self._m4_pending)) != batch_id:
                    raise RuntimeError("M4 observed native output FIFO violation")
                del self._m4_pending[batch_id]
                emit(
                    "m4_consume_success",
                    layer="gpu",
                    batch_id=batch_id,
                    intervals=binding["intervals"],
                    states=self._m4_states(),
                    outputs=[
                        {
                            "request_id": o.request_id,
                            "new_token_ids": list(o.new_token_ids),
                            "finish_reason": None
                            if o.finish_reason is None
                            else o.finish_reason.name,
                        }
                        for group in result.values()
                        for o in group.outputs
                    ],
                )

            events.observe("m4_consume_success", record)
            return result

        def _preempt_request(self, request, timestamp, drop_stale_output=False):
            before = events.observe(
                "m4_preempt_state", lambda: request_state(request, self._m4_registry, self)
            )
            events.observe(
                "m4_preempt_start",
                lambda: emit(
                    "m4_preempt_start",
                    layer="gpu",
                    request=request_state(request, self._m4_registry, self),
                    pending=self._m4_pending_rows(),
                ),
            )
            result = super()._preempt_request(request, timestamp, drop_stale_output)
            events.observe(
                "m4_preempt_success",
                lambda: emit(
                    "m4_preempt_success",
                    layer="gpu",
                    before=before,
                    request=request_state(request, self._m4_registry, self),
                    pending=self._m4_pending_rows(),
                ),
            )
            return result

        def finish_requests(self, request_ids, finished_status):
            # Read active state without consuming a potentially single-use iterable.
            before = events.observe("m4_cancel_state", self._m4_states)
            events.observe(
                "m4_cancel_start",
                lambda: emit(
                    "m4_cancel_start",
                    layer="gpu",
                    requests=before,
                    requested_ids=request_ids
                    if isinstance(request_ids, (str, list, tuple))
                    else None,
                    pending=self._m4_pending_rows(),
                ),
            )
            result = super().finish_requests(request_ids, finished_status)
            events.observe(
                "m4_cancel_success",
                lambda: emit(
                    "m4_cancel_success",
                    layer="gpu",
                    requests=[request_state(r, self._m4_registry, self) for r in result],
                    guard_after=guard_state(self, self._m4_registry),
                    pending=self._m4_pending_rows(),
                ),
            )
            return result

        def _free_request(self, request, delay_free_blocks=False):
            result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
            events.observe(
                "m4_request_terminal",
                lambda: emit(
                    "m4_request_terminal",
                    layer="gpu",
                    request=request_state(request, self._m4_registry, self),
                ),
            )
            return result

    class M4NativeScheduler(M4Mixin, ObserverNativeScheduler):
        pass

    class M4PrefixScheduler(M4Mixin, ObserverPrefixScheduler):
        pass

    class M4WorkerExtension(M3WorkerExtension):
        def m4_set_case(self, case_id, metadata):
            nonlocal active_case, active_metadata
            active_case = case_id
            active_metadata = {k: metadata[k] for k in ("scenario", "repeat", "seed", "mode")}
            emit("m4_case_context", layer="gpu", case_id=case_id)
            return {"case_id": case_id}

        def m4_snapshot(self, label="boundary", case_id=None):
            result = self.m3_snapshot(label, case_id)
            instance = events.get_scheduler()
            if instance is None:
                raise RuntimeError("M4 requires actual EngineCore scheduler")
            gc.collect()
            result["m4"] = {
                "requests": instance._m4_states(),
                "pending": instance._m4_pending_rows(),
                "weak_registry_size": len(instance._m4_registry),
                "next_generation": instance._m4_registry.next_generation,
            }
            result["drained"] = result["drained"] and not result["m4"]["pending"]
            emit(
                "m4_checkpoint",
                layer="gpu",
                case_id=case_id,
                label=label,
                resources=result["scheduler"],
                observer=result["m4"],
            )
            if result["drained"]:
                emit(
                    "m4_generation_drained",
                    layer="gpu",
                    case_id=case_id,
                    label=label,
                    resources=result["scheduler"],
                    observer=result["m4"],
                )
            return result

        def m4_probe(self, request_id, case_id):
            instance = events.get_scheduler()
            if instance is None:
                raise RuntimeError("M4 probe lacks EngineCore scheduler")
            request = instance.requests[request_id]
            flags = {
                "full": request.kv_cache_report_mode == "full",
                "events": instance.enable_kv_cache_events,
                "stats": instance.log_stats,
                "metrics": instance.kv_metrics_collector is not None,
            }
            if not all(flags.values()):
                raise RuntimeError("M4 probe requires full KV events/stats/metrics")
            emit(
                "m4_probe_activity",
                layer="gpu",
                case_id=case_id,
                metric_block_count=len(instance.kv_metrics_collector.block_metrics),
                cached_block_count=sum(
                    b._block_hash is not None for b in instance.kv_cache_manager.block_pool.blocks
                ),
            )
            probe_id = case_id + ":probe"
            before = storage_snapshot(instance, request)
            emit(
                "m4_probe_before",
                layer="gpu",
                case_id=case_id,
                probe_id=probe_id,
                flags=flags,
                snapshot=before,
            )
            score = reusable_prefix_tokens(instance.kv_cache_manager, request)
            after = storage_snapshot(instance, request)
            emit(
                "m4_probe_after",
                layer="gpu",
                case_id=case_id,
                probe_id=probe_id,
                flags=flags,
                snapshot=after,
                score=score,
                diagnostic_only=True,
            )
            return {
                "equal": before == after,
                "score": score,
                "flags": flags,
                "before_sha256": hashlib.sha256(
                    json.dumps(before, sort_keys=True).encode()
                ).hexdigest(),
                "after_sha256": hashlib.sha256(
                    json.dumps(after, sort_keys=True).encode()
                ).hexdigest(),
            }

    for cls in (M4NativeScheduler, M4PrefixScheduler, M4WorkerExtension):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls


try:
    _define_native_classes()
except ModuleNotFoundError as error:
    # Pure weak-identity/snapshot helpers have meaningful local unit tests.
    # Native class lookup still fails explicitly when vLLM is unavailable.
    if error.name != "vllm":
        raise
