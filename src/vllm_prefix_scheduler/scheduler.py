"""Native AsyncScheduler integration for the pinned vLLM 0.32.0 runtime.

The supported native interface and lookup semantics were inspected at upstream
5df63a7203cdb9328d824a27b682e77c7b9928be (Apache-2.0, vLLM contributors).
See docs/research/m2a-prefix-scheduler.md. Native scheduling is never substituted.
"""

import time
from collections import deque

import vllm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request, RequestStatus

from vllm_prefix_scheduler.config import parse_prefix_config
from vllm_prefix_scheduler.probe import reusable_prefix_tokens
from vllm_prefix_scheduler.wait_guard import WaitGuard, WaitGuardError


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"prefix_scheduler: {reason}")


def _validate_prefix_config(vconfig, kv_config, block_size, hash_block_size) -> None:
    """Reject unaudited global semantics before native resource construction."""
    _require(vconfig.device_config.device_type == "cuda", "prefix requires CUDA")
    parallel = vconfig.parallel_config
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "world_size",
    ):
        _require(getattr(parallel, field) == 1, f"prefix requires {field}=1")
    for field in (
        "enable_expert_parallel",
        "enable_dbo",
        "data_parallel_external_lb",
        "data_parallel_hybrid_lb",
    ):
        _require(not getattr(parallel, field), f"prefix does not support {field}")
    _require(vconfig.scheduler_config.policy == "fcfs", "prefix requires FCFS")
    _require(vconfig.use_v2_model_runner, "prefix requires Model Runner V2")
    _require(vconfig.cache_config.enable_prefix_caching, "prefix requires APC enabled")
    _require(
        vconfig.cache_config.kv_offloading_size is None, "prefix does not support KV offloading"
    )
    for field in ("speculative_config", "lora_config", "kv_transfer_config", "ec_transfer_config"):
        _require(getattr(vconfig, field) is None, f"prefix does not support {field}")
    _require(not vconfig.aux_output_config.enabled, "prefix does not support auxiliary output")
    _require(vconfig.offload_config.uva.cpu_offload_gb == 0, "prefix does not support CPU offload")
    _require(
        vconfig.offload_config.prefetch.offload_group_size == 0,
        "prefix does not support weight prefetch offload",
    )
    model = vconfig.model_config
    _require(model.runner_type == "generate", "prefix requires ordinary text generation")
    for field in ("is_encoder_decoder", "is_diffusion", "is_hybrid"):
        _require(not getattr(model, field), f"prefix does not support {field}")
    _require(model.multimodal_config is None, "prefix does not support multimodal models")
    groups = kv_config.kv_cache_groups
    _require(len(groups) == 1, "prefix requires exactly one full-attention KV group")
    group = groups[0]
    spec = group.kv_cache_spec
    _require(type(spec) is FullAttentionSpec, "prefix requires exact FullAttentionSpec")
    _require(not group.is_eagle_group, "prefix does not support EAGLE groups")
    _require(
        spec.sliding_window is None and spec.attention_chunk_size is None and not spec.non_causal,
        "prefix requires causal full attention without local windows",
    )
    effective_hash_size = block_size if hash_block_size is None else hash_block_size
    _require(
        block_size == effective_hash_size == spec.block_size,
        "prefix requires identical scheduler, hash and physical block sizes",
    )


class PrefixAsyncScheduler(AsyncScheduler):
    """A fixed-runtime extension with native allocation and async output handling."""

    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size,
        hash_block_size=None,
        mm_registry=MULTIMODAL_REGISTRY,
        include_finished_set=False,
        log_stats=False,
    ):
        self.prefix_config = parse_prefix_config(vllm_config.additional_config)
        _require(vllm.__version__ == "0.32.0", "extension requires pinned vLLM 0.32.0")
        _require(
            vllm_config.scheduler_config.async_scheduling is True,
            "extension requires effective async_scheduling=True",
        )
        if self.prefix_config.mode in {"prefix", "guarded"}:
            _validate_prefix_config(vllm_config, kv_cache_config, block_size, hash_block_size)
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        if self.prefix_config.mode in {"prefix", "guarded"}:
            try:
                self._validate_native_objects()
            except BaseException as error:
                try:
                    super().shutdown()
                except BaseException as cleanup_error:
                    error.add_note(f"Native shutdown after postcheck failure: {cleanup_error}")
                raise
        if self.prefix_config.mode == "guarded":
            self._wait_guard = WaitGuard(self.prefix_config.max_wait_ms, self._wait_clock_ns)

    def _validate_native_objects(self):
        manager = self.kv_cache_manager
        coordinator = manager.coordinator
        _require(type(self.waiting) is FCFSRequestQueue, "prefix requires native FCFSRequestQueue")
        _require(
            type(coordinator) is UnitaryKVCacheCoordinator,
            "prefix requires native UnitaryKVCacheCoordinator",
        )
        _require(
            len(coordinator.single_type_managers) == 1
            and type(coordinator.single_type_managers[0]) is FullAttentionManager,
            "prefix requires native FullAttentionManager",
        )
        single = coordinator.single_type_managers[0]
        _require(
            self.block_size
            == self.hash_block_size
            == coordinator.block_size
            == single.block_size
            == manager.block_pool.hash_block_size,
            "native cache block sizes differ from the supported contract",
        )
        _require(self.use_v2_model_runner, "native Model Runner V2 is not active")

    def schedule(self, throttle_prefills: bool = False):
        if self.prefix_config.mode == "guarded":
            guard = self._wait_guard
            guard.ensure_active()
            # Keep M2A's recoverable queue transaction outside native failure handling.
            # Even a one-item window must validate its original enqueue record.
            self._reorder_waiting()
            try:
                output = super().schedule(throttle_prefills=throttle_prefills)
                # The full AsyncScheduler parent must return successfully. Its final
                # update hook alone is too early (a profiling context can still fail).
                for request_id, tokens in output.num_scheduled_tokens.items():
                    if tokens > 0:
                        request = self.requests.get(request_id)
                        if request is not None:
                            guard.discard(request, "first_dispatch")
                return output
            except BaseException:
                guard.fail("native_schedule")
                raise
        if self.prefix_config.mode == "prefix" and self.prefix_config.window > 1:
            self._reorder_waiting()
        # Deliberately outside the queue transaction: native effects are not rolled back.
        return super().schedule(throttle_prefills=throttle_prefills)

    def _wait_clock_ns(self) -> int:
        """Private injection point; never replaces any native clock."""
        return time.monotonic_ns()

    def add_request(self, request: Request) -> None:
        if self.prefix_config.mode != "guarded":
            return super().add_request(request)
        guard = self._wait_guard
        guard.ensure_active()
        existing = self.requests.get(request.request_id)
        try:
            result = super().add_request(request)
        except BaseException:
            # Native add is not transactional: enqueue may precede the ID binding.
            # Existing-ID duplicate/session failures never refresh an old record.
            if existing is None:
                guard.fail("native_add")
            raise
        if existing is None:
            if self.requests.get(request.request_id) is not request:
                guard.fail("admission_identity")
                raise WaitGuardError("wait_guard: admission_identity")
            if self._fresh_request(request):
                guard.register(request)
        return result

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        if self.prefix_config.mode != "guarded":
            return super()._preempt_request(request, timestamp, drop_stale_output)
        guard = self._wait_guard
        status_before = request.status
        preemptions_before = request.num_preemptions
        try:
            return super()._preempt_request(request, timestamp, drop_stale_output)
        except BaseException:
            # This hook is also called outside schedule (e.g. reset_prefix_cache).
            guard.fail("native_preempt")
            raise
        finally:
            if (
                status_before is not RequestStatus.PREEMPTED
                and request.status is RequestStatus.PREEMPTED
            ) or request.num_preemptions > preemptions_before:
                guard.discard(request, "preempted")

    def _free_request(self, request, delay_free_blocks=False):
        if self.prefix_config.mode != "guarded":
            return super()._free_request(request, delay_free_blocks)
        guard = self._wait_guard
        try:
            return super()._free_request(request, delay_free_blocks)
        except BaseException:
            guard.fail("native_free")
            raise
        finally:
            if request.is_finished():
                guard.discard(request, "terminal")

    def shutdown(self):
        if self.prefix_config.mode != "guarded":
            return super().shutdown()
        try:
            return super().shutdown()
        finally:
            self._wait_guard.close()

    def _fresh_request(self, request) -> bool:
        """Conservative read-only predicate; do not advance native blocked handlers."""
        if type(request) is not Request or request.status is not RequestStatus.WAITING:
            return False
        if request in self.deferred_waiting or request in self._inflight_prefills:
            return False
        if any(
            (
                request.num_computed_tokens,
                request.num_preemptions,
                request.num_output_tokens,
                request.num_output_placeholders,
                request.num_in_flight_tokens,
                request.num_stale_output_tokens,
                request.next_decode_eligible_step,
                request.last_sched_seq,
                request.shared_prefix_boundary,
                request.replay_start,
            )
        ):
            return False
        if (
            request.is_prefill_chunk
            or request.drop_stale_output
            or request.spec_token_ids
            or request.resumable
            or request.streaming_queue is not None
            or request.session_id is not None
        ):
            return False
        return (
            bool(request.prompt_token_ids)
            and request.sampling_params is not None
            and request.pooling_params is None
            and request.prompt_embeds is None
            and request.prompt_is_token_ids is None
            and not request.mm_features
            and request.lora_request is None
            and request.structured_output_request is None
            and request.kv_transfer_params is None
            and request.ec_transfer_params is None
            and request.kv_hints is None
            and not request.abort_immediately
            and request.stop_reason is None
        )

    def _candidate_window(self):
        """Read at most window queue entries, stopping immediately at the first barrier."""
        candidates = []
        iterator = iter(self.waiting)
        # Unlike islice, this also supports arbitrarily large positive Python integers.
        for _ in range(min(self.prefix_config.window, len(self.waiting))):
            request = next(iterator)
            if not self._fresh_request(request):
                break
            candidates.append(request)
        return candidates

    def _score_request(self, request) -> int:
        return reusable_prefix_tokens(self.kv_cache_manager, request)

    def _sort_window(self, candidates):
        if self.prefix_config.mode == "guarded":
            guard = self._wait_guard
            now = guard.read_clock("decision")

            def key(request):
                if guard.is_protected(request, now):
                    return (0, guard.require_record(request).sequence)
                return (1, -self._score_request(request))

            return sorted(candidates, key=key)
        # sorted evaluates each key once and preserves current window order for ties.
        return sorted(candidates, key=self._score_request, reverse=True)

    def _reorder_waiting(self) -> None:
        original = self._candidate_window()
        if self.prefix_config.mode == "guarded":
            for request in original:
                self._wait_guard.require_record(request)
        if len(original) < 2:
            return
        ordered = self._sort_window(original)
        if all(left is right for left, right in zip(original, ordered, strict=True)):
            return
        queue = self.waiting
        tail_len = len(queue) - len(original)
        try:
            for _ in original:
                queue.pop_request()
            # FCFS prepend_requests uses extendleft, which reverses its input.
            queue.prepend_requests(reversed(ordered))
        except BaseException:
            # Native scheduling is serialized. The untouched suffix still has tail_len
            # entries even if a pop/prepend raised after a partial modification. Bypass
            # failing wrappers and restore just the head with the actual deque primitives.
            for _ in range(deque.__len__(queue) - tail_len):
                deque.popleft(queue)
            deque.extendleft(queue, reversed(original))
            raise
