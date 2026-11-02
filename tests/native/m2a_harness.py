"""Unmocked pinned scheduler metadata regression; no model/KV tensor execution.

Only ModelRunnerOutput values are synthetic. Construction follows the pinned
vLLM tests/v1/core/utils.py; FIFO delivery follows engine/core.py. M1 is read-only.
"""

import functools
import hashlib
import inspect
import json
import random
from collections import deque
from pathlib import Path

from m2a_snapshot import compact_request, snapshot, value

from vllm_prefix_scheduler.m2a_contract import expected_cases, require, require_assertions
from vllm_prefix_scheduler.native_replay import DispatchLedger


def create_scheduler(
    config,
    mode,
    model_path,
    *,
    capacity=10,
    budget=None,
    active=None,
    apc=True,
    window=None,
    mutate=None,
):
    import torch
    from vllm.config import (
        CacheConfig,
        KVEventsConfig,
        ModelConfig,
        ObservabilityConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_prefix_scheduler.scheduler import PrefixAsyncScheduler

    random.seed(config["seed"])
    model = ModelConfig(
        model=str(model_path),
        dtype="bfloat16",
        seed=config["seed"],
        skip_tokenizer_init=True,
        max_model_len=config["max_model_len"],
        enforce_eager=True,
    )
    scheduling = SchedulerConfig(
        max_num_seqs=active or config["max_num_seqs"],
        max_num_active_seqs=active or config["max_num_active_seqs"],
        max_num_batched_tokens=budget or config["max_num_batched_tokens"],
        max_model_len=config["max_model_len"],
        is_encoder_decoder=model.is_encoder_decoder,
        enable_chunked_prefill=True,
        async_scheduling=True,
        policy="fcfs",
        watermark=0.0,
        scheduler_reserve_full_isl=True,
        long_prefill_token_threshold=0,
        long_prefill_token_threshold_adaptive=False,
        scheduler_cls=None
        if mode == "native"
        else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler",
    )
    cache = CacheConfig(
        block_size=config["block_size"],
        enable_prefix_caching=apc,
        prefix_caching_hash_algo="sha256",
    )
    cache.num_gpu_blocks = capacity
    vc = VllmConfig(
        model_config=model,
        scheduler_config=scheduling,
        cache_config=cache,
        parallel_config=ParallelConfig(),
        kv_events_config=KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
        observability_config=ObservabilityConfig(
            kv_cache_metrics=True, kv_cache_metrics_sample=1.0
        ),
        additional_config={}
        if mode == "native"
        else {
            "prefix_scheduler": {
                "mode": mode,
                "window": window if window is not None else config["window"],
                "max_wait_ms": config["max_wait_ms"],
            }
        },
    )
    register_all_kvcache_specs(vc)
    spec = FullAttentionSpec(
        block_size=config["block_size"], num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv = KVCacheConfig(
        num_blocks=capacity,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["fixture_layer"], spec)],
    )
    args = {
        "vllm_config": vc,
        "kv_cache_config": kv,
        "block_size": config["block_size"],
        "structured_output_manager": StructuredOutputManager(vc),
        "log_stats": True,
    }
    if mutate is not None:
        mutate(vc, kv, args)
    resolved = vc.scheduler_config.get_scheduler_cls()
    scheduler = resolved(**args)
    require(
        type(scheduler) is (AsyncScheduler if mode == "native" else PrefixAsyncScheduler),
        "resolved scheduler has unexpected actual class",
    )
    require(
        isinstance(scheduler, AsyncScheduler) and vc.scheduler_config.async_scheduling is True,
        "actual asynchronous scheduler is required",
    )
    require(scheduler.use_v2_model_runner, "V2 is required by the frozen M2A runtime")
    return scheduler


def request(config, rid, tokens, *, arrival=0.0, skip_read=False, **kwargs):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        request_id=rid,
        prompt_token_ids=list(tokens),
        arrival_time=arrival,
        block_hasher=get_request_block_hasher(config["block_size"], sha256),
        pooling_params=None,
        sampling_params=SamplingParams(
            max_tokens=config["max_tokens"],
            ignore_eos=True,
            temperature=0,
            skip_reading_prefix_cache=skip_read,
            extra_args={"kv_cache_report_mode": "full"},
        ),
        **kwargs,
    )


def source_record(scheduler):
    classes = [type(scheduler), *type(scheduler).__mro__[1:]]
    return {
        "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "mro": [f"{c.__module__}.{c.__qualname__}" for c in classes],
        "sources": {
            f"{c.__module__}.{c.__qualname__}": {
                "path": inspect.getfile(c),
                "sha256": hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest(),
            }
            for c in classes
            if c.__module__ != "builtins"
        },
        "async_scheduling": scheduler.scheduler_config.async_scheduling,
        "use_v2_model_runner": scheduler.use_v2_model_runner,
        "max_concurrent_batches": scheduler.vllm_config.max_concurrent_batches,
        "max_in_flight_tokens": scheduler.vllm_config.max_in_flight_tokens,
        "instance_fields": sorted(scheduler.__dict__),
        "effective_config": {
            "budget": scheduler.scheduler_config.max_num_batched_tokens,
            "active": scheduler.max_num_active_reqs,
            "window": getattr(getattr(scheduler, "prefix_config", None), "window", None),
            "max_wait_ms": getattr(getattr(scheduler, "prefix_config", None), "max_wait_ms", None),
            "parallel": {
                k: getattr(scheduler.parallel_config, k)
                for k in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "decode_context_parallel_size",
                    "prefill_context_parallel_size",
                )
            },
        },
        "apc": scheduler.cache_config.enable_prefix_caching,
        "log_stats": scheduler.log_stats,
        "kv_events": scheduler.enable_kv_cache_events,
        "kv_metrics": scheduler.kv_metrics_collector is not None,
        "publisher": type(scheduler.kv_event_publisher).__name__,
        "block_size": scheduler.block_size,
        "capacity": scheduler.kv_cache_config.num_blocks,
    }


class Driver:
    def __init__(self, config, scheduler, emit):
        self.config, self.scheduler, self.emit = config, scheduler, emit
        self.requests, self.generations, self.current_keys = {}, {}, {}
        self.pending = deque()
        self.step, self.call_id, self.max_pending = 0, 0, 0
        self.owner = None
        self.originals = []
        self.ledger = DispatchLedger()
        self.dispatches, self.published, self.output_stats, self.windows = [], [], [], []
        self.step_counts = {}
        self.step_candidates, self.step_scores = {}, {}
        self.event("native_objects", **source_record(scheduler))
        for name in ("get_computed_blocks", "allocate_slots", "cache_blocks", "free"):
            self.wrap_native(scheduler.kv_cache_manager, name)
        for name in ("get_new_blocks", "_maybe_evict_cached_block", "touch", "free_blocks"):
            self.wrap_native(scheduler.kv_cache_manager.block_pool, name)
        self.wrap_native(scheduler, "_preempt_request")
        self.install(scheduler, "_update_after_schedule", self.advance_wrapper)
        self.install(scheduler.kv_event_publisher, "publish", self.publish_wrapper)
        if hasattr(scheduler, "_reorder_waiting"):
            self.install(scheduler, "_reorder_waiting", self.window_wrapper)
            self.install(scheduler, "_candidate_window", self.candidate_wrapper)
            self.install(scheduler, "_fresh_request", self.fresh_wrapper)
            self.install(scheduler, "_score_request", self.score_wrapper)
            self.install(scheduler, "_sort_window", self.sort_wrapper)
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        self.install(AsyncScheduler, "schedule", self.super_wrapper)

    def install(self, obj, name, factory):
        own = name in getattr(obj, "__dict__", {})
        original = getattr(obj, name)
        self.originals.append((obj, name, original, own))
        setattr(obj, name, factory(original))

    def close(self):
        for obj, name, original, own in reversed(self.originals):
            if own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.scheduler.shutdown()

    def event(self, event, **data):
        self.emit({"event": event, "step": self.step, **data})

    def state(self, full=False):
        return snapshot(self.scheduler, self.requests, self.generations, full)

    def counts(self):
        return self.step_counts.setdefault(
            self.step,
            {
                "checks": 0,
                "candidates": 0,
                "probes": 0,
                "sorts": 0,
                "reorders": 0,
                "super": 0,
                "window_calls": 0,
            },
        )

    def super_wrapper(self, original):
        driver = self

        @functools.wraps(original)
        def wrapped(instance, *args, **kwargs):
            if instance is driver.scheduler:
                driver.counts()["super"] += 1
                driver.event(
                    "native_schedule_enter", state=driver.state(), counts=driver.counts().copy()
                )
            return original(instance, *args, **kwargs)

        return wrapped

    def fresh_wrapper(self, original):
        def wrapped(r):
            self.counts()["checks"] += 1
            return original(r)

        return wrapped

    def candidate_wrapper(self, original):
        def wrapped():
            candidates = original()
            self.counts()["candidates"] += len(candidates)
            self.step_candidates[self.step] = [r.request_id for r in candidates]
            self.event("candidates", ids=[r.request_id for r in candidates], count=len(candidates))
            return candidates

        return wrapped

    def score_wrapper(self, original):
        def wrapped(r):
            self.counts()["probes"] += 1
            score = original(r)
            self.step_scores.setdefault(self.step, []).append([r.request_id, score])
            self.event("score", request_id=r.request_id, score=score)
            return score

        return wrapped

    def sort_wrapper(self, original):
        def wrapped(candidates):
            self.counts()["sorts"] += 1
            return original(candidates)

        return wrapped

    def window_wrapper(self, original):
        def wrapped():
            self.counts()["window_calls"] += 1
            before = [r.request_id for r in self.scheduler.waiting]
            self.event("window_before", waiting=before, state=self.state())
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
            self.event("window_after", **row, state=self.state())
            return result

        return wrapped

    def publish_wrapper(self, original):
        def wrapped(batch):
            serialized = value(batch)
            self.published.append(serialized)
            self.event("published", batch=serialized)
            return original(batch)

        return wrapped

    def advance_wrapper(self, original):
        def wrapped(output):
            intervals = []
            for rid, count in output.num_scheduled_tokens.items():
                r = self.scheduler.requests[rid]
                generation = self.generations[rid]
                interval = {
                    "request_id": rid,
                    "generation": generation,
                    "prompt_length": r.num_prompt_tokens,
                    "step": self.step,
                    "preemptions": r.num_preemptions,
                    **self.ledger.record(
                        f"{rid}@{generation}", r.num_computed_tokens, count, r.num_prompt_tokens
                    ),
                }
                intervals.append(interval)
                self.dispatches.append(interval)
            self.event("dispatch", intervals=intervals)
            result = original(output)
            self.event("advance", state=self.state())
            return result

        return wrapped

    def wrap_native(self, obj, name):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                old_owner = self.owner
                if name in {
                    "get_computed_blocks",
                    "allocate_slots",
                    "cache_blocks",
                    "free",
                    "_preempt_request",
                }:
                    self.owner = args[0].request_id
                self.call_id += 1
                call = self.call_id
                details = {}
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
                finally:
                    self.owner = old_owner

            return wrapped

        self.install(obj, name, factory)

    def add(self, r):
        rid = r.request_id
        generation = self.generations.get(rid, -1) + 1
        self.generations[rid] = generation
        key = rid if generation == 0 else f"{rid}@{generation}"
        self.current_keys[rid] = key
        self.requests[key] = r
        self.scheduler.add_request(r)
        self.event(
            "enqueue",
            ids=[rid],
            generation=generation,
            arrivals={rid: r.arrival_time},
            tokens=list(r.prompt_token_ids),
            state=self.state(),
        )

    def schedule(self, phase="main"):
        from vllm.v1.outputs import ModelRunnerOutput

        require(len(self.pending) < self.config["pending_output_depth"], "pending FIFO overflow")
        self.step += 1
        self.event(
            "schedule_before",
            phase=phase,
            pending_outputs=len(self.pending),
            state=self.state(full=True),
        )
        output = self.scheduler.schedule()
        ids = list(output.num_scheduled_tokens)
        sampled = [
            []
            if self.scheduler.requests[rid].is_prefill_chunk
            else [self.config["synthetic_output_token_id"]]
            for rid in ids
        ]
        synthetic = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        self.pending.append((self.step, output, synthetic))
        self.max_pending = max(self.max_pending, len(self.pending))
        self.event(
            "schedule_after",
            phase=phase,
            admitted=[r.req_id for r in output.scheduled_new_reqs],
            scheduled=output.num_scheduled_tokens,
            sampled_token_ids=sampled,
            preempted_ids=sorted(output.preempted_req_ids or ()),
            finished_ids=sorted(output.finished_req_ids),
            pending_outputs=len(self.pending),
            counts=self.counts().copy(),
            output=value(output),
            state=self.state(full=True),
        )
        require(self.counts()["super"] == 1, "native schedule must be called exactly once")
        require(
            self.counts()["window_calls"] <= 1 and self.counts()["sorts"] <= 1,
            "repeated window pass",
        )
        return output

    def consume(self):
        step, output, synthetic = self.pending.popleft()
        self.event(
            "output_before",
            output_step=step,
            pending_outputs=len(self.pending) + 1,
            state=self.state(),
        )
        result = self.scheduler.update_from_output(output, synthetic)
        serialized = value(result)
        self.output_stats.append(serialized)
        self.event("output_stats", output_step=step, outputs=serialized)
        self.event(
            "output_after",
            output_step=step,
            pending_outputs=len(self.pending),
            state=self.state(full=True),
        )

    def drain(self, phase="main"):
        for _ in range(self.config["max_steps_per_phase"]):
            if not self.scheduler.has_requests() and not self.pending:
                self.event("drained", phase=phase, pending_outputs=0, state=self.state(full=True))
                return
            if (
                self.scheduler.has_requests()
                and len(self.pending) < self.config["pending_output_depth"]
            ):
                output = self.schedule(phase)
                if (
                    len(self.pending) < self.config["pending_output_depth"]
                    and output.total_num_scheduled_tokens
                ):
                    continue
            if self.pending:
                self.consume()
        raise RuntimeError(f"{phase}: finite step limit exceeded")

    def cancel(self, rid):
        from vllm.v1.request import RequestStatus

        self.scheduler.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
        self.event("cancel", request_id=rid, state=self.state(full=True))

    def finish(self):
        state = self.state(full=True)
        require(
            not self.pending
            and not self.scheduler.requests
            and not self.scheduler.running
            and not self.scheduler.waiting
            and not self.scheduler.kv_holding_waiting
            and not self.scheduler.deferred_waiting,
            "live state did not drain",
        )
        require(
            all(b["ref_cnt"] == 0 for b in state["blocks"] if not b["null"]), "references leaked"
        )
        require(
            len(state["free_queue"]) == self.scheduler.kv_cache_config.num_blocks - 1,
            "free queue leaked",
        )
        for r in self.requests.values():
            require(r.is_finished(), "request unfinished")
            if r.status.name != "FINISHED_ABORTED":
                require(
                    r.num_in_flight_tokens
                    == r.num_output_placeholders
                    == r.num_stale_output_tokens
                    == 0,
                    "normal terminal request has undrained async counters",
                )
        totals = {}
        for key, r in self.requests.items():
            generation = 0 if "@" not in key else int(key.rsplit("@", 1)[1])
            rows = [
                d
                for d in self.dispatches
                if d["request_id"] == r.request_id and d["generation"] == generation
            ]
            totals[key] = {
                name: sum(d[name] for d in rows)
                for name in (
                    "prefill_tokens",
                    "decode_tokens",
                    "repeated_prefill_tokens",
                    "repeated_decode_tokens",
                )
            }
            totals[key].update(preemptions=r.num_preemptions, final_status=r.status.name)
        return {
            "status": "passed",
            "requests": totals,
            "max_pending_outputs": self.max_pending,
            "final_pending_outputs": 0,
            "all_references_released": True,
            "published_batches": len(self.published),
            "output_updates": len(self.output_stats),
            "state": state,
        }


def _abc(config):
    return {
        x["id"]: request(config, x["id"], x["tokens"], arrival=x["arrival_time"])
        for x in config["requests"]
    }


def _warm(d, config):
    a = _abc(config)["A"]
    d.add(a)
    d.drain("warmup")
    cached = [b for b in d.state()["blocks"] if b["hash"]]
    require(
        len(cached) == 2 and all(b["ref_cnt"] == 0 for b in cached), "warm cache is not evictable"
    )
    d.event("A_cached_and_evictable", cached_blocks=cached, state=d.state(full=True))


def bc_case(d, config, apc):
    if apc:
        _warm(d, config)
    else:
        d.add(_abc(config)["A"])
        d.drain("warmup")
    requests = _abc(config)
    d.add(requests["B"])
    d.add(requests["C"])
    d.event("bc_enqueued", ids=["B", "C"], state=d.state(full=True))
    d.drain("BC")


def probe_case(d, config):
    from vllm_prefix_scheduler.probe import reusable_prefix_tokens

    _warm(d, config)
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
        r = request(config, label, tokens, skip_read=skip)
        d.requests[label] = r
        before = d.state(full=True)
        d.event("probe_before", probe_id=label, state=before, expected=expected)
        score = reusable_prefix_tokens(d.scheduler.kv_cache_manager, r)
        after = d.state(full=True)
        d.event("probe_after", probe_id=label, state=after, score=score, expected=expected)
        require(before == after and score == expected, f"probe {label} changed state or score")
    # Native-effect positive controls prove snapshots can see each mutation.
    manager = d.scheduler.kv_cache_manager
    pool = manager.block_pool
    r = d.requests["full"]
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
    # Detached lookup requests never entered native lifetime; remove them from
    # the driver's terminal registry, then schedule the same ordinary full request.
    d.requests = {"A": d.requests["A"]}
    d.add(r)
    d.drain("post_probe_native")
    require(d.published and d.output_stats, "native legal event/stat output missing")


def chunk_case(d, config):
    _warm(d, config)
    cold = request(config, "D", range(1000, 1065), arrival=3.0)
    d.add(cold)
    d.schedule("chunk_first")
    require(cold.is_prefill_chunk, "actual chunk did not occur")
    d.add(_abc(config)["C"])
    d.drain("chunk")
    prompt_rows = [x for x in d.dispatches if x["request_id"] == "D" and x["prefill_tokens"]]
    require(len(prompt_rows) >= 5, "chunk must dispatch prompt in multiple steps")
    d.event("check", name="real_chunk", intervals=prompt_rows)


def preemption_case(d, config):
    first = request(config, "R0", range(1000, 1032))
    second = request(config, "R1", range(2000, 2032), arrival=1.0)
    d.add(first)
    d.add(second)
    d.schedule("both_prefill")
    require(len(d.scheduler.running) == 2, "preemption fixture failed initial joint admission")
    d.schedule("natural_preemption")
    require(
        second.num_preemptions > 0
        and second.num_computed_tokens == 0
        and second.num_stale_output_tokens > 0,
        "native pending-output preemption did not occur",
    )
    d.event(
        "check", name="real_preemption", victim=compact_request(second), state=d.state(full=True)
    )
    d.consume()
    d.add(request(config, "H", list(range(1000, 1032)) + [3000], arrival=2.0))
    require(
        [r.request_id for r in d.scheduler.waiting][:2] == ["R1", "H"], "preemption barrier missing"
    )
    d.schedule("preempted_barrier")
    d.drain("preemption")


def cancellation_case(d, config):
    d.add(request(config, "running_cancel", range(1000, 1033)))
    d.schedule("pending_cancel")
    d.add(request(config, "waiting_cancel", range(2000, 2033)))
    d.cancel("waiting_cancel")
    d.cancel("running_cancel")
    d.drain("cancelled")
    d.add(request(config, "survivor", range(3000, 3033)))
    d.drain("survivor")


def id_reuse_case(d, config):
    d.add(request(config, "reuse", range(1000, 1033)))
    d.drain("generation0")
    d.event("reuse_boundary", pending_outputs=len(d.pending), state=d.state(full=True))
    d.add(request(config, "reuse", range(2000, 2033), arrival=10.0))
    d.drain("generation1")
    d.add(request(config, "cancel_reuse", range(3000, 3033)))
    d.schedule("cancel_generation0")
    d.cancel("cancel_reuse")
    d.drain("cancel_generation0")
    d.event("reuse_boundary", pending_outputs=len(d.pending), state=d.state(full=True))
    d.add(request(config, "cancel_reuse", range(4000, 4033), arrival=20.0))
    d.drain("cancel_generation1")


def score_eviction_case(d, config):
    _warm(d, config)
    d.add(request(config, "R", range(1000, 1032)))
    d.schedule("running_prefill")
    # Two live R blocks leave only the two cached A blocks evictable. The next
    # running decode allocates one of these AFTER the extension scores C=32.
    d.add(request(config, "cold", range(2000, 2033), arrival=2.0))
    d.add(_abc(config)["C"])
    d.schedule("score_then_evict")
    d.drain("score_eviction")


def window_case(config, model_path, emit):
    from vllm.v1.request import RequestStatus

    labels = [
        "stable_ties",
        "all_cold",
        "window_one",
        "window_limit",
        "head_barrier",
        "middle_barrier",
        "preempt_history",
        "partial_progress",
        "placeholder",
        "in_flight",
        "stale",
        "resumable",
        "unsupported",
        "prepend_order",
        "score_exception",
        "pop_exception",
        "prepend_exception",
        "repeated_step",
        "other_queues",
    ]
    results = []
    for label in labels:
        window = 1 if label == "window_one" else 2 if label == "window_limit" else 16
        scheduler = create_scheduler(config, "prefix", model_path, window=window)

        def sub_emit(row, label=label):
            emit({"subcase": label, **row})

        d = Driver(config, scheduler, sub_emit)
        try:
            _warm(d, config)
            hot = list(range(100, 132)) + [201]
            specs = [("cold", list(range(700, 733))), ("hot", hot), ("tail", hot)]
            if label == "stable_ties":
                specs = [("first", hot), ("second", hot), ("tail", hot)]
            elif label == "all_cold":
                specs = [("first", list(range(700, 733))), ("second", list(range(800, 833)))]
            reqs = [
                request(
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
                scheduler.running.append(request(config, "other_running", [42]))
                scheduler.kv_holding_waiting.add_request(request(config, "other_holding", [43]))
                scheduler.deferred_waiting.add(request(config, "other_deferred", [44]))
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
            results.append(d.finish())
        finally:
            d.close()
    return {"status": "passed", "subcases": labels, "results": results}


def global_rejection_case(config, model_path, emit):
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
        "encoder_decoder": lambda vc, kv, args: setattr(
            vc.model_config.hf_config, "is_encoder_decoder", True
        ),
        "diffusion": lambda vc, kv, args: setattr(vc.model_config.hf_config, "canvas_length", 1),
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
    rejected = []
    for label, mutate in tests.items():
        caught = None
        try:
            scheduler = create_scheduler(config, "prefix", model_path, mutate=mutate)
        except (ValueError, RuntimeError, NotImplementedError) as error:
            caught = str(error)
        else:
            scheduler.shutdown()
        require(
            caught is not None and "prefix_scheduler:" in caught,
            f"incompatible config {label} did not fail in the extension: {caught}",
        )
        rejected.append(label)
        emit({"event": "check", "step": 0, "name": label, "rejected": True, "error": caught})
    for mode, window in [("prefix", 1), ("guarded", 16)]:
        caught = None
        try:
            scheduler = create_scheduler(config, mode, model_path, apc=False, window=window)
        except (ValueError, RuntimeError, NotImplementedError) as error:
            caught = str(error)
        else:
            scheduler.shutdown()
        require(caught is not None, "fast-path configuration bypassed validation")
        label = "window1_apc_off" if window == 1 else "guarded"
        rejected.append(label)
        emit({"event": "check", "step": 0, "name": label, "rejected": True, "error": caught})
    return {"status": "passed", "rejected": rejected}


def run_matrix(config, model_path, output_path):
    require_assertions()
    cases = expected_cases(config)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    results = []
    with (output_path / "trace.jsonl").open("x") as stream:
        for case, metadata in cases.items():

            def emit(row, case=case, metadata=metadata):
                stream.write(json.dumps({"case": case, **metadata, **row}, sort_keys=True) + "\n")
                stream.flush()

            emit({"event": "case_begin", "step": 0, "seed": config["seed"]})
            scenario, mode = metadata["scenario"], metadata["mode"]
            if scenario == "window":
                result = window_case(config, model_path, emit)
            elif scenario == "global_rejection":
                result = global_rejection_case(config, model_path, emit)
            else:
                capacity = config["capacities"][
                    "pressure"
                    if scenario in {"bc_pressure", "preemption", "score_eviction"}
                    else "ample"
                ]
                scheduler = create_scheduler(
                    config,
                    mode,
                    model_path,
                    capacity=capacity,
                    budget=16 if scenario == "chunk" else None,
                    active=1 if scenario == "chunk" else None,
                    apc=scenario != "apc_off",
                )
                d = Driver(config, scheduler, emit)
                try:
                    if scenario in {"bc_pressure", "bc_ample", "apc_off"}:
                        bc_case(d, config, scenario != "apc_off")
                    else:
                        {
                            "probe": probe_case,
                            "chunk": chunk_case,
                            "preemption": preemption_case,
                            "cancellation": cancellation_case,
                            "id_reuse": id_reuse_case,
                            "score_eviction": score_eviction_case,
                        }[scenario](d, config)
                    result = d.finish()
                finally:
                    d.close()
            result = {"case": case, **metadata, **result}
            results.append(result)
            emit({"event": "case_end", "step": 0, "result": result})
            (output_path / "summary.json").write_text(
                json.dumps({"verified": False, "results": results}, indent=2) + "\n"
            )
    return results
