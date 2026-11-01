"""Check the internal consistency of a published native readiness record."""

import argparse
import json
from pathlib import Path


def check_readiness(record: dict, pins: dict) -> dict:
    """Reject unsupported claims or inconsistent cache/accounting measurements."""
    errors = []
    if record.get("status") != "passed":
        errors.append("the native run did not pass")
    if record.get("vllm") != pins["vllm"]["version"]:
        errors.append("vLLM version differs from the pinned runtime")
    if record.get("model_revision") != pins["model"]["revision"]:
        errors.append("model revision differs from the pinned snapshot")

    config = record.get("resolved_config", {})
    if config.get("dtype") != "torch.bfloat16":
        errors.append("BF16 model configuration was not observed")
    if config.get("async_scheduling") is not True:
        errors.append("async scheduling was not enabled")
    if not config.get("scheduler_class", "").endswith(".AsyncScheduler"):
        errors.append("the resolved scheduler is not AsyncScheduler")
    if config.get("enable_prefix_caching") is not True:
        errors.append("APC was not enabled")

    workers = record.get("worker_after_requests", [])
    if len(workers) != 1:
        errors.append("the record is not a single-worker readiness run")
    else:
        worker = workers[0]
        if worker.get("model_parameter_dtype") != "torch.bfloat16":
            errors.append("the actual worker parameters were not BF16")
        if worker.get("async_scheduling") is not True:
            errors.append("the worker did not confirm async scheduling")
        if worker.get("enable_prefix_caching") is not True:
            errors.append("the worker did not confirm APC")
        groups = worker.get("kv_cache_groups", [])
        capacity = record.get("kv_capacity", {})
        if len(groups) != 1 or groups[0].get("spec_class") != "FullAttentionSpec":
            errors.append("a single full-attention KV group was not observed")
        else:
            group = groups[0]
            blocks = worker.get("num_gpu_blocks", 0)
            expected_tokens = blocks * group["block_size"]
            expected_bytes = blocks * group["page_size_bytes"] * group["layer_count"]
            if blocks <= 0 or capacity.get("num_gpu_blocks") != blocks:
                errors.append("KV block count is inconsistent")
            if capacity.get("capacity_tokens") != expected_tokens:
                errors.append("KV token capacity is inconsistent")
            if capacity.get("allocation_bytes") != expected_bytes:
                errors.append("KV byte capacity is inconsistent")
            if worker.get("kv_allocation_bytes") != expected_bytes:
                errors.append("worker KV allocation is inconsistent")

    requests = record.get("requests", [])
    if not requests or any(
        not request.get("finished")
        or request.get("output_tokens", 0) <= 0
        or not request.get("text", "").strip()
        for request in requests
    ):
        errors.append("some requests did not finish with nonempty output")
    by_label = {request["label"]: request for request in requests}
    cold = by_label.get("prefix_cold", {})
    warm = by_label.get("prefix_warm_identical_prompt", {})
    cold_hits = cold.get("num_cached_tokens")
    warm_hits = warm.get("num_cached_tokens")
    if not (
        isinstance(cold_hits, int)
        and isinstance(warm_hits, int)
        and 0 <= cold_hits < warm_hits <= warm.get("prompt_tokens", 0)
        and cold.get("prompt_tokens") == warm.get("prompt_tokens")
        and record.get("apc_behavior_verified") is True
    ):
        errors.append("the cold/warm request pair does not demonstrate APC reuse")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "passed",
        "completed_requests": len(requests),
        "cold_cached_tokens": cold_hits,
        "warm_cached_tokens": warm_hits,
        "kv_capacity_tokens": record["kv_capacity"]["capacity_tokens"],
        "kv_allocation_gib": record["kv_capacity"]["allocation_bytes"] / 1024**3,
        "gpu_sampled_peak_mib": record["gpu_memory_sampled_peak_used_mib"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--pins", type=Path, default=Path("configs/upstream.json"))
    args = parser.parse_args()
    result = check_readiness(json.loads(args.record.read_text()), json.loads(args.pins.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
