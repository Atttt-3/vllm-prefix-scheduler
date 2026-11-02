"""Named worker RPC and passive V2 runner observation for M3 only.

Pinned upstream anchors: gpu_worker.py:1293-1340; gpu/model_runner.py:
1719-1748,1997-2084,2088-2184; gpu/async_utils.py:115-184;
executor/uniproc_executor.py:26-45,90-125. No device synchronization is
introduced. The original async-output object is returned without conversion.
"""

import os
import sys
import weakref
from collections import Counter

from vllm_prefix_scheduler import m3_events as events


def _result_kind(output):
    if output is None:
        return "none"
    if callable(getattr(output, "get_output", None)):
        return "async"
    return "output"


def _record_return(event, batch_id, output, **fields):
    kind = _result_kind(output)
    events.emit(
        event,
        batch_id=batch_id,
        result_kind=kind,
        output_type=events.qualified_type(output),
        **fields,
    )
    if kind == "async":
        events.observe_materialization(output, batch_id)
    elif kind == "output":
        events.emit(
            "model_output_materialized", batch_id=batch_id, **events.model_output_fields(output)
        )


def install_runner_observer(runner):
    """Install once after model startup; all parent calls remain exactly once."""
    if getattr(runner, "_m3_observer_installed", False):
        return
    if events.qualified_type(runner) != "vllm.v1.worker.gpu.model_runner.GPUModelRunner":
        raise RuntimeError("M3 diagnostics require the pinned active V2 GPUModelRunner")
    # Keep the native function and only a weak instance reference: keeping a
    # bound method in an instance-owned closure creates a GPU-resource cycle.
    original_execute = runner.execute_model.__func__
    original_sample = runner.sample_tokens.__func__
    runner_ref = weakref.ref(runner)
    state = {"sample_batch": None}

    def execute_model(scheduler_output, *args, **kwargs):
        active_runner = runner_ref()
        if active_runner is None:
            raise ReferenceError("native model runner no longer exists")
        batch_id = events.observe(
            "runner_batch_binding", lambda: events.batch_id_for(scheduler_output)
        )
        events.observe(
            "runner_execute_start",
            lambda: events.emit(
                "runner_execute_start",
                batch_id=batch_id,
                **events.output_descriptor(scheduler_output),
            ),
        )
        try:
            result = original_execute(active_runner, scheduler_output, *args, **kwargs)
        except BaseException as error:
            events.emit(
                "runner_execute_error", batch_id=batch_id, error_type=events.qualified_type(error)
            )
            raise
        state["sample_batch"] = (
            batch_id if scheduler_output.total_num_scheduled_tokens > 0 else None
        )

        def record():
            execution = active_runner.execute_model_state
            batch = None if execution is None else execution.input_batch
            _record_return(
                "runner_execute_success",
                batch_id,
                result,
                forward_tokens=None if batch is None else int(batch.num_tokens),
                forward_tokens_padded=None
                if batch is None
                else int(batch.num_tokens_after_padding),
                forward_req_ids=[] if batch is None else list(batch.req_ids),
                forward_observed=execution is not None,
            )

        events.observe("runner_execute_success", record)
        return result

    def sample_tokens(*args, **kwargs):
        active_runner = runner_ref()
        if active_runner is None:
            raise ReferenceError("native model runner no longer exists")
        batch_id = state["sample_batch"]
        events.emit("runner_sample_start", batch_id=batch_id)
        try:
            result = original_sample(active_runner, *args, **kwargs)
        except BaseException as error:
            events.emit(
                "runner_sample_error", batch_id=batch_id, error_type=events.qualified_type(error)
            )
            raise
        events.observe(
            "runner_sample_success",
            lambda: _record_return("runner_sample_success", batch_id, result),
        )
        state["sample_batch"] = None
        return result

    runner.execute_model = execute_model
    runner.sample_tokens = sample_tokens
    runner._m3_observer_installed = True
    events.emit("runner_observer_installed", runner_type=events.qualified_type(runner))


def _spec_fields(spec):
    result = {
        "type": events.qualified_type(spec),
        "block_size": spec.block_size,
        "page_size_bytes": spec.page_size_bytes,
    }
    for field in (
        "dtype",
        "num_kv_heads",
        "head_size",
        "sliding_window",
        "attention_chunk_size",
        "non_causal",
    ):
        if hasattr(spec, field):
            value = getattr(spec, field)
            result[field] = str(value) if field == "dtype" else value
    if hasattr(spec, "kv_cache_specs"):
        result["per_layer_specs"] = {
            name: _spec_fields(layer_spec) for name, layer_spec in spec.kv_cache_specs.items()
        }
    return result


class M3WorkerExtension:
    """Read-only snapshots; diagnostic hook installation is explicitly reported."""

    def m3_snapshot(self, label="boundary", case_id=None):
        import torch

        runner = self.model_runner
        cfg = self.vllm_config
        diagnostic = os.environ.get("M3_PHASE") == "diagnostic"
        if diagnostic:
            # TP=PP=1 defaults to UniProcExecutor: the runner and scheduler share
            # the real EngineCore process and SchedulerOutput object. Refuse any
            # different topology rather than inventing cross-process IDs.
            if cfg.parallel_config.distributed_executor_backend != "uni":
                raise RuntimeError("M3 diagnostic observer requires the native uni executor")
            if events.get_scheduler() is None:
                raise RuntimeError("M3 diagnostic scheduler is not registered in EngineCore")
            install_runner_observer(runner)

        kv = runner.kv_cache_config
        tensors = []
        storage_bytes = {}
        for tensor in runner.kv_caches:
            storage = tensor.untyped_storage()
            identity = f"{tensor.device}:{storage.data_ptr():x}"
            storage_bytes[identity] = storage.nbytes()
            tensors.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "numel": tensor.numel(),
                    "view_bytes": tensor.numel() * tensor.element_size(),
                    "storage_id": identity,
                    "storage_bytes": storage.nbytes(),
                }
            )
        parameter_dtypes = Counter()
        parameter_devices = Counter()
        for parameter in runner.model.parameters():
            parameter_dtypes[str(parameter.dtype)] += parameter.numel()
            parameter_devices[str(parameter.device)] += parameter.numel()
        module_names = [
            "vllm.v1.engine.core",
            "vllm.v1.core.sched.scheduler",
            "vllm.v1.core.sched.async_scheduler",
            "vllm.v1.worker.gpu_worker",
            "vllm.v1.worker.gpu.model_runner",
            "vllm.v1.worker.gpu.async_utils",
            "vllm.v1.executor.uniproc_executor",
        ]
        snapshot = {
            "pid": os.getpid(),
            "label": label,
            "case_id": case_id,
            "worker_type": events.qualified_type(self),
            "runner_type": events.qualified_type(runner),
            "runner_observer_installed": bool(getattr(runner, "_m3_observer_installed", False)),
            "model_parameter_dtypes": dict(parameter_dtypes),
            "model_parameter_devices": dict(parameter_devices),
            "model_type": events.qualified_type(runner.model),
            "model_config_dtype": str(cfg.model_config.dtype),
            "gpu_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(self.device).total_memory,
            "torch_memory_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_memory_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "num_gpu_blocks": kv.num_blocks,
            "kv_cache_groups": [
                {"layer_names": list(group.layer_names), "spec": _spec_fields(group.kv_cache_spec)}
                for group in kv.kv_cache_groups
            ],
            "kv_tensors": tensors,
            "kv_unique_storage_bytes": sum(storage_bytes.values()),
            "kv_unique_storage_count": len(storage_bytes),
            "kv_tensor_descriptor_bytes": [tensor.size for tensor in kv.kv_cache_tensors],
            "runner_request_ids": sorted(runner.req_states.req_id_to_index),
            "runner_execute_state_pending": runner.execute_model_state is not None,
            "configured_scheduler_type": events.qualified_type(
                cfg.scheduler_config.get_scheduler_cls()
            ),
            "configured_scheduler_module": events.module_source(
                sys.modules[cfg.scheduler_config.get_scheduler_cls().__module__]
            ),
            "effective_config": {
                "async_scheduling": cfg.scheduler_config.async_scheduling,
                "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
                "enable_chunked_prefill": cfg.scheduler_config.enable_chunked_prefill,
                "use_v2_model_runner": cfg.use_v2_model_runner,
                "max_num_seqs": cfg.scheduler_config.max_num_seqs,
                "max_num_active_seqs": cfg.scheduler_config.max_num_active_seqs,
                "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
                "max_model_len": cfg.model_config.max_model_len,
                "gpu_memory_utilization": cfg.cache_config.gpu_memory_utilization,
                "enforce_eager": cfg.model_config.enforce_eager,
                "executor_backend": cfg.parallel_config.distributed_executor_backend,
                "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
                "pipeline_parallel_size": cfg.parallel_config.pipeline_parallel_size,
                "max_concurrent_batches": cfg.max_concurrent_batches,
            },
            "modules": {name: events.module_source(sys.modules[name]) for name in module_names},
            "observer_modules": {
                "events": events.module_source(events),
                "worker": events.module_source(sys.modules[__name__]),
            },
            "production_modules": {
                name: events.module_source(sys.modules[f"vllm_prefix_scheduler.{name}"])
                for name in ("scheduler", "config", "probe", "wait_guard")
                if f"vllm_prefix_scheduler.{name}" in sys.modules
            },
            "scheduler": None,
            "observation_errors": events.observation_errors(),
        }
        if diagnostic:
            from vllm_prefix_scheduler.m3_observer import scheduler_checkpoint

            snapshot["scheduler"] = scheduler_checkpoint(events.get_scheduler(), label, case_id)
        # A progress convenience only: the independent reviewer recomputes all
        # component gates instead of trusting this boolean.
        drained = (
            not snapshot["runner_request_ids"] and not snapshot["runner_execute_state_pending"]
        )
        if diagnostic:
            resource = snapshot["scheduler"]
            empty_fields = (
                "requests",
                "waiting",
                "running",
                "holding",
                "deferred",
                "finished_req_ids",
                "reset_preempted_req_ids",
                "inflight_prefills",
                "pending_batches",
                "guard_records",
                "deferred_frees",
                "request_blocks",
                "non_null_ref_sum",
                "inflight_tokens",
                "stale_tokens",
                "output_placeholders",
            )
            drained = drained and all(not resource[field] for field in empty_fields)
            drained = (
                drained
                and len(
                    {
                        resource[field]
                        for field in (
                            "free_count",
                            "free_unique_count",
                            "free_traversed_count",
                            "total_non_null_blocks",
                        )
                    }
                )
                == 1
            )
            drained = drained and resource["guard_fault_code"] is None
            drained = drained and not resource["observation_errors"]
        snapshot["drained"] = drained and not snapshot["observation_errors"]
        events.emit("worker_snapshot", **snapshot)
        return snapshot
