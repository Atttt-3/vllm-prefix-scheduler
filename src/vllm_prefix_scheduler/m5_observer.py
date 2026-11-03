"""M5-only same-process scheduler and V2 runner measurement.

The runner position copy is ordered on its native main stream and captured after
native async output materialization. A bounded host buffer keeps those exact
positions until post-drain validation. No device synchronization is added.
"""

import hashlib
import json
import os
import socket
import sys
import time
import weakref
from array import array
from pathlib import Path

from vllm_prefix_scheduler.m3_events import module_source, qualified_type
from vllm_prefix_scheduler.m4_observer import WeakIdentityRegistry


def clock_identity():
    return {
        "clock": "CLOCK_MONOTONIC",
        "host": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "time_namespace": os.readlink("/proc/self/ns/time"),
        "pid": os.getpid(),
    }


_state = None
_scheduler = None
CAPTURE_TOKEN_LIMIT = 2_000_000


def _append(name, row):
    rows = _state[name]
    if len(rows) >= _state["limit"]:
        raise RuntimeError(f"M5 {name} observation buffer exhausted")
    rows.append({"stage": _state["stage"], **row})


def _record_error(operation, error):
    if len(_state["errors"]) < 32:
        _state["errors"].append({"operation": operation, "error": str(error)})


def _observe(operation, callback, *args):
    try:
        return callback(*args)
    except Exception as error:
        _record_error(operation, error)
        return None


def _capture_buffer(limit):
    positions = array("q", [0]) * limit
    if positions.itemsize != 8:
        raise RuntimeError("M5 requires an 8-byte signed host position buffer")
    return {
        "positions": positions,
        "position_bytes": memoryview(positions).cast("B"),
        "position_count": 0,
        "prompt_count": 0,
        "limit": limit,
    }


def _capture_materialized(record, output, slots):
    slot = record.pop("slot")
    if slot is None:
        raise RuntimeError("M5 missing native runner positions")
    try:
        if record.get("position_copy_submitted") is not True:
            raise RuntimeError("M5 native runner position copy was not submitted")
        shape = record.pop("shape")
        capture = _state["_capture"]
        offset = capture["position_count"]
        end = offset + shape["effective_tokens"]
        if end > capture["limit"]:
            raise RuntimeError("M5 raw runner position buffer exhausted")
        # The original AsyncOutput.get_output has already synchronized its event.
        # Detach the exact copied bytes before reusing the pinned slot.
        source = memoryview(slot.numpy())
        if source.ndim != 1 or source.itemsize != 8 or source.format not in ("l", "q"):
            raise RuntimeError("M5 unexpected host runner position format")
        capture["position_bytes"][offset * 8 : end * 8] = source.cast("B")[: (end - offset) * 8]
        batch = record["batch"]
        batch["_raw_runner"] = (offset, shape, tuple(output.req_ids))
        batch["materialized"] = True
        capture["position_count"] = end
    finally:
        slots.append(slot)


def _format_runner(batch, raw, positions):
    offset, shape, output_ids = raw
    if output_ids != tuple(row[0] for row in shape["rows"]):
        raise RuntimeError("M5 native runner output request order changed")
    effective = shape["effective_tokens"]
    if not 0 <= offset < offset + effective <= len(positions):
        raise RuntimeError("M5 runner position capture outside committed host bytes")
    target = positions[offset : offset + effective]
    generation = {row["request_id"]: row["generation"] for row in batch["scheduled"]}
    rows = []
    previous_end = 0
    for rid, lower, upper, prefill_length in shape["rows"]:
        if not previous_end == lower < upper <= effective:
            raise RuntimeError("M5 runner query offsets do not partition effective inputs")
        start = target[lower]
        if any(position != start + i for i, position in enumerate(target[lower + 1 : upper], 1)):
            raise RuntimeError("M5 runner positions are not a contiguous request interval")
        rows.append(
            {
                "request_id": rid,
                "generation": generation[rid],
                "start": start,
                "end": target[upper - 1] + 1,
                "prefill_length": prefill_length,
                "position_source": "runner_positions_d2h",
            }
        )
        previous_end = upper
    if previous_end != effective:
        raise RuntimeError("M5 runner query offsets do not partition effective inputs")
    batch["runner"] = {
        "effective_tokens": effective,
        "padded_tokens": shape["padded_tokens"],
        "rows": rows,
    }


def _format_enqueue(row, raw):
    ids, block_hashes = raw
    row["prompt_sha256"] = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    row["block_hashes"] = [block_hash.hex() for block_hash in block_hashes]


def _flush_capture():
    capture = _state["_capture"]
    if capture is None:
        return
    if _state["stage"] != "done" or _state["pending"]:
        _record_error("export", "M5 observer export before full drain")
    positions = memoryview(capture["positions"])[: capture["position_count"]]
    for row in _state["enqueues"]:
        raw = row.pop("_raw_enqueue")
        _observe("export_enqueue", _format_enqueue, row, raw)
    for batch in _state["batches"]:
        raw = batch.pop("_raw_runner", None)
        if not batch["scheduled"]:
            if raw is not None:
                _record_error("export_runner", "M5 runner capture attached to an empty batch")
            continue
        if raw is None or batch["materialized"] is not True or batch["consumed"] is not True:
            _record_error(
                "export_runner", "M5 native runner capture was not materialized and consumed"
            )
            continue
        _observe("export_runner", _format_runner, batch, raw, positions)
    _state["_capture"] = None


def _install_runner(runner, config):
    import torch

    if qualified_type(runner) != "vllm.v1.worker.gpu.model_runner.GPUModelRunner":
        raise RuntimeError("M5 requires the pinned V2 GPUModelRunner")
    if (
        runner.speculator is not None
        or runner.pcp_manager is not None
        or runner.use_pp
        or qualified_type(runner.model_state)
        != "vllm.v1.worker.gpu.model_states.default.DefaultModelState"
        or runner.model_state.rope_state is not None
    ):
        raise RuntimeError("M5 cannot bind actual positions for this runner topology")
    original_execute = runner.execute_model.__func__
    original_sample = runner.sample_tokens.__func__
    runner_ref = weakref.ref(runner)
    slots = [
        torch.empty(
            config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for _ in range(config.max_concurrent_batches + 1)
    ]
    sample_pending = None

    def copy_positions(active, record):
        execution = active.execute_model_state
        if execution is None or not slots:
            raise RuntimeError("M5 runner forward state or pinned host slot missing")
        batch = execution.input_batch
        effective = int(batch.num_tokens)
        padded = int(batch.num_tokens_after_padding)
        slot = slots[-1]
        if effective <= 0 or effective > slot.numel() or padded < effective:
            raise RuntimeError("M5 invalid actual runner input size")
        if torch.cuda.current_stream(active.device) != active.main_stream:
            raise RuntimeError("M5 runner position copy outside native main stream")
        positions = batch.positions
        if positions.ndim != 1 or positions.dtype != torch.int64:
            raise RuntimeError("M5 unexpected Qwen3 text position dtype or rank")
        offsets = batch.query_start_loc_np
        if int(offsets[0]) != 0 or int(offsets[len(batch.req_ids)]) != effective:
            raise RuntimeError("M5 runner query offsets do not cover effective inputs")
        record["shape"] = {
            "effective_tokens": effective,
            "padded_tokens": padded,
            "rows": [
                (rid, int(offsets[i]), int(offsets[i + 1]), int(batch.prefill_len_np[i]))
                for i, rid in enumerate(batch.req_ids)
            ],
        }
        record["slot"] = slots.pop()
        slot[:effective].copy_(positions[:effective], non_blocking=True)
        record["position_copy_submitted"] = True

    def execute(output, *args, **kwargs):
        nonlocal sample_pending
        active = runner_ref()
        result = original_execute(active, output, *args, **kwargs)
        record = _state["pending"].get(id(output))
        if record is None or record["output"] is not output:
            _record_error("execute", "unbound batch")
        elif output.total_num_scheduled_tokens > 0:
            sample_pending = record
            _observe("copy_positions", copy_positions, active, record)
        else:
            record["batch"]["materialized"] = result is not None
        return result

    def sample(*args, **kwargs):
        nonlocal sample_pending
        record, sample_pending = sample_pending, None
        output = original_sample(runner_ref(), *args, **kwargs)
        if record is None:
            _record_error("sample", "missing execute")
            return output
        if not callable(getattr(output, "get_output", None)):
            _record_error("sample", "expected native AsyncOutput")
            return output
        original = output.get_output.__func__
        output_ref = weakref.ref(output)

        def get_output(*inner_args, **inner_kwargs):
            materialized = original(output_ref(), *inner_args, **inner_kwargs)
            _observe("materialized", _capture_materialized, record, materialized, slots)
            return materialized

        output.get_output = get_output
        return output

    runner.execute_model = execute
    runner.sample_tokens = sample


def _profile(worker):
    import torch

    runner, cfg = worker.model_runner, worker.vllm_config
    backends = {group.backend for groups in runner.attn_groups for group in groups}
    backend = sorted({impl.get_name() for impl in backends})
    parameters = next(runner.model.parameters())
    result = {
        "runner_type": qualified_type(runner),
        "model_type": qualified_type(runner.model),
        "num_gpu_blocks": runner.kv_cache_config.num_blocks,
        "block_size": cfg.cache_config.block_size,
        "attention_backend": backend,
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(worker.device).total_memory,
        "model_dtype": str(parameters.dtype),
        "model_device": str(parameters.device),
        "async_scheduling": cfg.scheduler_config.async_scheduling,
        "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
        "enable_chunked_prefill": cfg.scheduler_config.enable_chunked_prefill,
        "use_v2_model_runner": cfg.use_v2_model_runner,
        "enforce_eager": cfg.model_config.enforce_eager,
        "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": cfg.parallel_config.pipeline_parallel_size,
        "executor_backend": cfg.parallel_config.distributed_executor_backend,
        "scheduler_policy": cfg.scheduler_config.policy,
        "max_concurrent_batches": cfg.max_concurrent_batches,
        "max_num_seqs": cfg.scheduler_config.max_num_seqs,
        "max_num_active_seqs": cfg.scheduler_config.max_num_active_seqs,
        "scheduler_reserve_full_isl": cfg.scheduler_config.scheduler_reserve_full_isl,
        "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
        "max_model_len": cfg.model_config.max_model_len,
        "gpu_memory_utilization": cfg.cache_config.gpu_memory_utilization,
        "scheduler_type": qualified_type(cfg.scheduler_config.get_scheduler_cls()),
        "production_modules": {
            name: module_source(sys.modules[f"vllm_prefix_scheduler.{name}"])
            for name in ("scheduler", "config", "probe", "wait_guard")
            if f"vllm_prefix_scheduler.{name}" in sys.modules
        },
        "sources": {
            name: module_source(sys.modules[name])
            for name in (
                "vllm.v1.core.sched.scheduler",
                "vllm.v1.core.sched.async_scheduler",
                "vllm.v1.worker.gpu.model_runner",
                "vllm.v1.worker.gpu.input_batch",
                "vllm.v1.worker.gpu.states",
                "vllm.v1.worker.gpu.async_utils",
                type(runner.model_state).__module__,
                type(runner.model).__module__,
                *(impl.__module__ for impl in backends),
            )
        },
    }
    if (
        backend != ["FLASH_ATTN"]
        or result["gpu_name"] != "NVIDIA GeForce RTX 3090"
        or result["compute_capability"] != [8, 6]
        or result["model_type"] != "vllm.model_executor.models.qwen3.Qwen3ForCausalLM"
        or result["model_dtype"] != "torch.bfloat16"
        or not result["model_device"].startswith("cuda")
        or result["tensor_parallel_size"] != 1
        or result["pipeline_parallel_size"] != 1
        or result["executor_backend"] != "uni"
        or result["scheduler_policy"] != "fcfs"
        or not all(
            result[key] is True
            for key in (
                "async_scheduling",
                "enable_prefix_caching",
                "enable_chunked_prefill",
                "use_v2_model_runner",
                "enforce_eager",
            )
        )
    ):
        raise RuntimeError("M5 actual runner profile differs from the pinned runtime")
    return result


class M5WorkerExtension:
    """RPC boundaries stay outside the natural-arrival measured path."""

    def m5_arm(self, tier, limit, schema_version=1, capture_token_limit=CAPTURE_TOKEN_LIMIT):
        global _state
        if (
            _state is not None
            or type(limit) is not int
            or limit < 1
            or type(schema_version) is not int
            or schema_version not in (1, 2)
            or type(capture_token_limit) is not int
            or not 1 <= capture_token_limit <= CAPTURE_TOKEN_LIMIT
        ):
            raise RuntimeError("M5 already armed or invalid journal limit")
        profile = _profile(self)
        _state = {
            "schema_version": schema_version,
            "clock": clock_identity(),
            "profile": profile,
            "batches": [],
            "enqueues": [],
            "decisions": [],
            "preemptions": [],
            "allocations": [],
            "evictions": [],
            "errors": [],
            "boundaries": [],
            "overhead": [],
            "probes": [],
            "pending": {},
            "stage": "warmup",
            "tier": tier,
            "limit": limit,
            "_capture": None if tier == "endpoint" else _capture_buffer(capture_token_limit),
        }
        if tier != "endpoint":
            if _scheduler is None or _scheduler() is None:
                raise RuntimeError("M5 scheduler and runner must share EngineCore")
            _install_runner(self.model_runner, self.vllm_config)
        return {"clock": _state["clock"], "engine_ns": time.monotonic_ns(), "profile": profile}

    def m5_clock(self):
        return {"engine_ns": time.monotonic_ns()}

    def m5_boundary(self, label, stage=None):
        from vllm_prefix_scheduler.m3_observer import scheduler_resources

        if _scheduler is None:
            result = {
                "runner_request_ids": list(self.model_runner.req_states.req_id_to_index),
                "runner_pending": self.model_runner.execute_model_state is not None,
            }
        else:
            scheduler = _scheduler()
            result = {
                **scheduler_resources(scheduler),
                "runner_request_ids": list(self.model_runner.req_states.req_id_to_index),
                "runner_pending": self.model_runner.execute_model_state is not None,
                "m5_pending": len(_state["pending"]),
            }
        result.update(label=label, at_ns=time.monotonic_ns())
        _append("boundaries", result)
        if stage is not None:
            _state["stage"] = stage
        return result

    def m5_probe(self, request_ids):
        scheduler = _scheduler()
        manager = scheduler.kv_cache_manager
        results = []
        for rid in request_ids:
            request = scheduler.requests.get(rid)
            if request is None:
                continue
            blocks, hit, _ = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            results.append(
                {
                    "request_id": rid,
                    "hit_tokens": hit,
                    "block_ids": [block.block_id for group in blocks for block in group],
                    "at_ns": time.monotonic_ns(),
                }
            )
        if len(results) == len(request_ids):
            for row in results:
                _append("probes", row)
        return results

    def m5_export(self):
        _flush_capture()
        return {
            key: value
            for key, value in _state.items()
            if key not in ("pending", "limit", "_capture")
        }


def _define_native_classes():
    from vllm.v1.core.kv_cache_utils import get_block_hash
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

    class M5Mixin:
        def __init__(self, *args, **kwargs):
            global _scheduler
            super().__init__(*args, **kwargs)
            _scheduler = weakref.ref(self)
            self._m5_registry = WeakIdentityRegistry()
            self._m5_index = 0
            self._m5_current = None
            self._m5_allocation = None
            self._m5_allocation_id = 0
            self._install_cache_observers()

        def _install_cache_observers(self):
            manager = self.kv_cache_manager
            pool = manager.block_pool
            original_allocate = manager.allocate_slots.__func__
            original_evict = pool._maybe_evict_cached_block.__func__
            original_remove = pool._remove_cached_block_hashes.__func__
            scheduler_ref = weakref.ref(self)
            manager_ref, pool_ref = weakref.ref(manager), weakref.ref(pool)

            def allocate(request, *args, **kwargs):
                if _state is None:
                    return original_allocate(manager_ref(), request, *args, **kwargs)
                scheduler = scheduler_ref()
                before = pool_ref().get_num_free_blocks()
                scheduler._m5_allocation_id += 1
                context = {
                    "allocation_id": scheduler._m5_allocation_id,
                    "request": request,
                    "eviction_count": 0,
                }
                scheduler._m5_allocation = context
                try:
                    result = original_allocate(manager_ref(), request, *args, **kwargs)
                finally:
                    scheduler._m5_allocation = None

                def record():
                    _append(
                        "allocations",
                        {
                            "batch_id": scheduler._m5_index,
                            "request_id": request.request_id,
                            "generation": scheduler._m5_registry.binding(request)["generation"],
                            "free_before": before,
                            "free_after": pool_ref().get_num_free_blocks(),
                            "success": result is not None,
                            "hit_tokens": kwargs.get("num_new_computed_tokens", 0),
                            "at_ns": time.monotonic_ns(),
                            **(
                                {
                                    "allocation_id": context["allocation_id"],
                                    "eviction_count": context["eviction_count"],
                                }
                                if _state["schema_version"] == 2
                                else {}
                            ),
                        },
                    )

                _observe("allocation", record)
                return result

            removed_hashes = None

            def remove(block):
                nonlocal removed_hashes
                result = original_remove(pool_ref(), block)
                removed_hashes = result
                return result

            def evict(block):
                nonlocal removed_hashes
                removed_hashes = None
                result = original_evict(pool_ref(), block)
                if _state is not None and result:
                    scheduler = scheduler_ref()
                    context = scheduler._m5_allocation
                    if context is not None:
                        context["eviction_count"] += 1

                    def record():
                        if _state["schema_version"] == 2 and context is None:
                            # The pinned full-attention path has no connector or checkpoint evictions.
                            raise RuntimeError("M5 native eviction outside an allocation")
                        request = None if context is None else context["request"]
                        keys = [bytes(get_block_hash(key)).hex() for key in removed_hashes]
                        _append(
                            "evictions",
                            {
                                "batch_id": scheduler._m5_index,
                                "request_id": None if request is None else request.request_id,
                                "block_id": block.block_id,
                                "reusable": True,
                                "cache_key": keys[0],
                                "removed_cache_keys": keys,
                                "at_ns": time.monotonic_ns(),
                                **(
                                    {"allocation_id": context["allocation_id"]}
                                    if _state["schema_version"] == 2
                                    else {}
                                ),
                            },
                        )

                    _observe("eviction", record)
                removed_hashes = None
                return result

            manager.allocate_slots = allocate
            pool._maybe_evict_cached_block = evict
            pool._remove_cached_block_hashes = remove

        def add_request(self, request):
            result = super().add_request(request)
            if _state is not None:

                def record():
                    if self.requests.get(request.request_id) is not request:
                        raise RuntimeError("request was not bound to native request object")
                    capture = _state["_capture"]
                    ids = request.prompt_token_ids
                    block_hashes = request.block_hashes
                    prompt_count = capture["prompt_count"] + len(ids)
                    if prompt_count > capture["limit"] or len(block_hashes) > len(ids):
                        raise RuntimeError("M5 raw enqueue prompt buffer exhausted or invalid")
                    _append(
                        "enqueues",
                        {
                            "request_id": request.request_id,
                            "generation": self._m5_registry.binding(request)["generation"],
                            "enqueued_ns": time.monotonic_ns(),
                            "_raw_enqueue": (tuple(ids), tuple(map(bytes, block_hashes))),
                        },
                    )
                    capture["prompt_count"] = prompt_count

                _observe("enqueue", record)
            return result

        def schedule(self, throttle_prefills=False):
            if _state is None:
                return super().schedule(throttle_prefills=throttle_prefills)
            profile = _state["tier"] == "profile"
            envelope_start = (
                time.monotonic_ns() if profile and _state["schema_version"] == 2 else None
            )
            self._m5_index += 1
            self._m5_current = {
                "candidates": [],
                "ordered": [],
                "protected": [],
                "decision_ns": time.monotonic_ns(),
                "window_kind": "production",
            }
            self._m5_timings = (
                dict.fromkeys(
                    ("window_ns", "score_ns", "sort_inclusive_ns", "commit_ns", "reorder_ns"), 0
                )
                if profile
                else None
            )
            prefix = getattr(self, "prefix_config", None)
            if prefix is None or prefix.mode == "off":
                _observe("shadow_window", self._shadow_window)
            start_ns = time.monotonic_ns()
            output = super().schedule(throttle_prefills=throttle_prefills)
            end_ns = time.monotonic_ns()
            _observe("schedule", self._record_batch, output, end_ns, end_ns - start_ns)
            self._m5_current = None
            if envelope_start is not None and not _state["errors"]:
                _state["batches"][-1]["schedule_envelope_ns"] = time.monotonic_ns() - envelope_start
            return output

        def _shadow_window(self):
            from itertools import islice

            ids = []
            for request in islice(self.waiting, int(os.environ["M5_WINDOW"])):
                if not PrefixAsyncScheduler._fresh_request(self, request):
                    break
                ids.append(request.request_id)
            self._m5_current.update(candidates=ids, ordered=ids, window_kind="shadow")

        def _record_batch(self, output, scheduled_ns, schedule_ns):
            batch_id = self._m5_index
            scheduled = []
            for rid, count in output.num_scheduled_tokens.items():
                req = self.requests[rid]
                scheduled.append(
                    {
                        "request_id": rid,
                        "generation": self._m5_registry.binding(req)["generation"],
                        "start": req.num_computed_tokens - count,
                        "end": req.num_computed_tokens,
                        "prompt_length": req.num_prompt_tokens,
                        "preemptions": req.num_preemptions,
                    }
                )
            _append(
                "batches",
                {
                    "batch_id": batch_id,
                    "scheduled_ns": scheduled_ns,
                    "schedule_ns": schedule_ns,
                    "scheduled": scheduled,
                    "runner": None,
                    "materialized": False,
                    "consumed": False,
                    "outputs": [],
                    "has_candidates": bool(self._m5_current["candidates"]),
                },
            )
            if id(output) in _state["pending"]:
                raise RuntimeError("M5 duplicate in-flight output binding")
            _state["pending"][id(output)] = {
                "output": output,
                "batch": _state["batches"][-1],
                "slot": None,
            }
            if self._m5_current["candidates"]:
                _append("decisions", {"batch_id": batch_id, **self._m5_current})
            if _state["tier"] == "profile":
                _append("overhead", {"batch_id": batch_id, **self._m5_timings})

        def update_from_output(self, scheduler_output, model_runner_output):
            result = super().update_from_output(scheduler_output, model_runner_output)
            if _state is not None:

                def record():
                    row = _state["pending"].pop(id(scheduler_output))
                    if row["output"] is not scheduler_output:
                        raise RuntimeError("M5 native consumption identity mismatch")
                    batch = row["batch"]
                    generations = {r["request_id"]: r["generation"] for r in batch["scheduled"]}
                    batch["outputs"] = [
                        {
                            "request_id": out.request_id,
                            "generation": generations[out.request_id],
                            "token_ids": list(out.new_token_ids),
                        }
                        for group in result.values()
                        for out in group.outputs
                        if out.new_token_ids
                    ]
                    batch["consumed"] = True

                _observe("consumed", record)
            return result

        def _preempt_request(self, request, timestamp, drop_stale_output=False):
            result = super()._preempt_request(request, timestamp, drop_stale_output)
            if _state is not None:

                def record():
                    _append(
                        "preemptions",
                        {
                            "batch_id": self._m5_index,
                            "request_id": request.request_id,
                            "generation": self._m5_registry.binding(request)["generation"],
                            "preemptions": request.num_preemptions,
                            "at_ns": time.monotonic_ns(),
                        },
                    )

                _observe("preempt", record)
            return result

    class M5NativeScheduler(M5Mixin, AsyncScheduler):
        pass

    class M5PrefixScheduler(M5Mixin, PrefixAsyncScheduler):
        def _reorder_waiting(self):
            if _state is None or _state["tier"] != "profile":
                return super()._reorder_waiting()
            start = time.monotonic_ns()
            queue = self.waiting
            original_pop = queue.pop_request
            original_prepend = queue.prepend_requests

            def timed(method):
                def call(*args, **kwargs):
                    begin = time.monotonic_ns()
                    try:
                        return method(*args, **kwargs)
                    finally:
                        self._m5_timings["commit_ns"] += time.monotonic_ns() - begin

                return call

            queue.pop_request = timed(original_pop)
            queue.prepend_requests = timed(original_prepend)
            try:
                return super()._reorder_waiting()
            finally:
                del queue.pop_request
                del queue.prepend_requests
                self._m5_timings["reorder_ns"] = time.monotonic_ns() - start

        def _candidate_window(self):
            profile = _state is not None and _state["tier"] == "profile"
            start = time.monotonic_ns() if profile else 0
            candidates = super()._candidate_window()
            if profile:
                self._m5_timings["window_ns"] = time.monotonic_ns() - start
            if self._m5_current is not None:
                ids = [r.request_id for r in candidates]
                self._m5_current.update(candidates=ids, ordered=ids)
            return candidates

        def _score_request(self, request):
            if _state is None or _state["tier"] != "profile":
                return super()._score_request(request)
            start = time.monotonic_ns()
            try:
                return super()._score_request(request)
            finally:
                self._m5_timings["score_ns"] += time.monotonic_ns() - start

        def _sort_window(self, candidates):
            profile = _state is not None and _state["tier"] == "profile"
            start = time.monotonic_ns() if profile else 0
            result = super()._sort_window(candidates)
            if profile:
                self._m5_timings["sort_inclusive_ns"] = time.monotonic_ns() - start
            if self._m5_current is not None:
                self._m5_current["ordered"] = [r.request_id for r in result]
                guard = getattr(self, "_wait_guard", None)
                if guard is not None:
                    self._m5_current.update(
                        decision_ns=guard.last_ns,
                        guard_enqueued_ns={
                            r.request_id: guard.records[r].enqueued_ns for r in candidates
                        },
                        protected=[
                            r.request_id
                            for r in result
                            if guard.last_ns - guard.records[r].enqueued_ns >= guard.threshold_ns
                        ],
                    )
            return result

    M5NativeScheduler.__qualname__ = M5NativeScheduler.__name__
    M5PrefixScheduler.__qualname__ = M5PrefixScheduler.__name__
    return M5NativeScheduler, M5PrefixScheduler


def __getattr__(name):
    if name in ("M5NativeScheduler", "M5PrefixScheduler"):
        native, prefix = _define_native_classes()
        globals().update(M5NativeScheduler=native, M5PrefixScheduler=prefix)
        return globals()[name]
    raise AttributeError(name)
