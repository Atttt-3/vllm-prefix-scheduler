"""Read-only named worker RPC extension; no scheduling or execution changes."""


class ReadinessWorkerExtension:
    def readiness_snapshot(self):
        import torch

        torch.cuda.synchronize()
        config = self.vllm_config
        runner = self.model_runner
        kv = runner.kv_cache_config
        groups = []
        for group in kv.kv_cache_groups:
            spec = group.kv_cache_spec
            groups.append(
                {
                    "spec_class": type(spec).__name__,
                    "block_size": spec.block_size,
                    "layer_count": len(group.layer_names),
                    "page_size_bytes": spec.page_size_bytes,
                    "dtype": str(getattr(spec, "dtype", None)),
                }
            )
        scheduler = config.scheduler_config.get_scheduler_cls()
        return {
            "model_parameter_dtype": str(next(runner.model.parameters()).dtype),
            "model_config_dtype": str(config.model_config.dtype),
            "async_scheduling": config.scheduler_config.async_scheduling,
            "scheduler_class": f"{scheduler.__module__}.{scheduler.__name__}",
            "enable_prefix_caching": config.cache_config.enable_prefix_caching,
            "num_gpu_blocks": kv.num_blocks,
            "kv_cache_groups": groups,
            "kv_allocation_bytes": sum(tensor.size for tensor in kv.kv_cache_tensors),
            "torch_memory_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_memory_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "model_runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
        }
