"""Verify effective runtime values and module hashes against independent pins."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m5_seal_test", ROOT / "scripts/m5_seal.py")
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


def test_active_profile_source_gate():
    config = json.loads((ROOT / "configs/m5-performance.json").read_text())
    upstream = json.loads((ROOT / "configs/native-m2b.json").read_text())["source_sha256"]
    upstream.update(config["extra_upstream_sha256"])
    names = (
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.input_batch",
        "vllm.v1.worker.gpu.states",
        "vllm.v1.worker.gpu.async_utils",
        "vllm.v1.worker.gpu.model_states.default",
        "vllm.model_executor.models.qwen3",
        "vllm.v1.attention.backends.flash_attn",
    )
    profile = {
        **json.loads((ROOT / config["baseline_config"]).read_text()),
        "block_size": 16,
        "attention_backend": ["FLASH_ATTN"],
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "compute_capability": [8, 6],
        "model_type": "vllm.model_executor.models.qwen3.Qwen3ForCausalLM",
        "model_dtype": "torch.bfloat16",
        "model_device": "cuda:0",
        "use_v2_model_runner": True,
        "pipeline_parallel_size": 1,
        "executor_backend": "uni",
        "scheduler_policy": "fcfs",
        "runner_type": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "scheduler_type": "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "num_gpu_blocks": 5553,
        "sources": {name: {"sha256": upstream[name.replace(".", "/") + ".py"]} for name in names},
        "production_modules": {},
    }
    run = {"mode": "native", "tier": "endpoint", "mechanism": None}
    source = {"upstream_sha256": upstream}
    seal.validate_profile(profile, run, config, source, {})
    with pytest.raises(ValueError, match="runtime profile differs"):
        seal.validate_profile({**profile, "max_num_batched_tokens": 4096}, run, config, source, {})
    profile["sources"][names[0]]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime module hashes differ"):
        seal.validate_profile(profile, run, config, source, {})
