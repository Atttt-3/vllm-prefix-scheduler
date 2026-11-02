"""Native AsyncScheduler integration for the pinned vLLM 0.32.0 runtime.

The supported native interface and lookup semantics were inspected at upstream
5df63a7203cdb9328d824a27b682e77c7b9928be (Apache-2.0, vLLM contributors).
See docs/research/m2a-prefix-scheduler.md. Native scheduling is never substituted.
"""

import vllm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_prefix_scheduler.config import parse_prefix_config


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
    """A fixed-runtime extension; native allocation and lifecycle stay inherited."""

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
        if self.prefix_config.mode == "prefix":
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
        if self.prefix_config.mode == "prefix":
            try:
                self._validate_native_objects()
            except BaseException as error:
                try:
                    super().shutdown()
                except BaseException as cleanup_error:
                    error.add_note(f"Native shutdown after postcheck failure: {cleanup_error}")
                raise

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
        return super().schedule(throttle_prefills=throttle_prefills)
