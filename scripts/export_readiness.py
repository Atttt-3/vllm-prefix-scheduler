"""Export synthetic readiness evidence with an explicit field allowlist.

This exporter is for this project's synthetic readiness schema, not arbitrary
production prompts, request traces or cluster logs. Review the exported diff.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

TOP_FIELDS = {
    "status",
    "started_at_utc",
    "finished_at_utc",
    "vllm",
    "model_revision",
    "project_commit",
    "scope",
    "kv_capacity",
    "apc_behavior_verified",
    "gpu_memory_sampling_interval_seconds",
    "gpu_memory_sample_count",
    "gpu_memory_sampled_peak_used_mib",
}
ENGINE_FIELDS = {
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enable_prefix_caching",
    "async_scheduling",
    "enable_chunked_prefill",
    "enforce_eager",
    "seed",
    "trust_remote_code",
    "disable_log_stats",
    "worker_extension_cls",
    "scheduler_class",
}
WORKER_FIELDS = {
    "model_parameter_dtype",
    "model_config_dtype",
    "async_scheduling",
    "scheduler_class",
    "enable_prefix_caching",
    "num_gpu_blocks",
    "kv_cache_groups",
    "kv_allocation_bytes",
    "torch_memory_allocated_bytes",
    "torch_memory_reserved_bytes",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "gpu_name",
    "compute_capability",
    "model_runner_class",
}
REQUEST_FIELDS = {
    "label",
    "request_id",
    "finished",
    "prompt_tokens",
    "output_tokens",
    "num_cached_tokens",
    "finish_reason",
    "text",
    "call_wall_seconds_not_benchmark",
}
GPU_FIELDS = {"name", "total_mib", "used_mib", "free_mib", "utilization_percent"}
ENV_FIELDS = {
    "vllm",
    "torch",
    "torch_cuda",
    "cuda_available",
    "gpu_name",
    "compute_capability",
    "bf16_supported",
    "requested_vllm_upstream_commit",
    "installation",
    "flashinfer_python",
    "flashinfer_cubin",
    "flashinfer_jit_cache",
    "dependency_check",
}
CAPACITY_FIELDS = {
    "num_gpu_blocks",
    "block_size_tokens",
    "capacity_tokens",
    "allocation_bytes",
    "allocation_gib",
    "full_16k_sequences_theoretical",
}
GROUP_FIELDS = {"spec_class", "block_size", "layer_count", "page_size_bytes", "dtype"}


def select(record, fields):
    return {key: value for key, value in record.items() if key in fields}


def export_record(record):
    if record.get("scope") != (
        "native minimal inference/APC/async readiness, not optimization benchmark"
    ):
        raise ValueError("Only the synthetic readiness schema is supported")
    result = select(record, TOP_FIELDS)
    result["kv_capacity"] = select(record["kv_capacity"], CAPACITY_FIELDS)
    result["python_version"] = record["python"].split()[0]
    result["model_id"] = "Qwen/Qwen3-4B"
    result["requested_engine_args"] = select(record["requested_engine_args"], ENGINE_FIELDS)
    result["resolved_config"] = select(record["resolved_config"], ENGINE_FIELDS)
    for key in ("worker_after_init", "worker_after_requests"):
        result[key] = [select(worker, WORKER_FIELDS) for worker in record[key]]
        for worker in result[key]:
            worker["kv_cache_groups"] = [
                select(group, GROUP_FIELDS) for group in worker["kv_cache_groups"]
            ]
    result["requests"] = [select(request, REQUEST_FIELDS) for request in record["requests"]]
    for key in ("gpu_before_engine", "gpu_after_engine_init", "gpu_after_requests"):
        result[key] = {
            "utc": record[key]["utc"],
            "gpus": [select(gpu, GPU_FIELDS) for gpu in record[key]["gpus"]],
        }
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing evidence: {args.output}")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vllm_prefix_scheduler.evidence import check_readiness

    raw_path = args.run_dir / "validation.json"
    record = export_record(json.loads(raw_path.read_text()))
    pins = json.loads((root / "configs" / "upstream.json").read_text())
    summary = check_readiness(record, pins)
    files = {"validation.json": record, "summary.json": summary}
    hashes = {"validation.json": digest(raw_path)}
    env_path = args.run_dir / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text())
        public_env = select(env, ENV_FIELDS)
        public_env["python_version"] = env["python"].split()[0]
        files["environment.json"] = public_env
        hashes["environment.json"] = digest(env_path)
    source_path = args.run_dir / "pinned-source-verification.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        if source.get("commit") != pins["vllm"]["commit"] or source.get("all_match") is not True:
            raise ValueError("Pinned-source verification does not match configured upstream")
        files["pinned-source-verification.json"] = {
            "commit": source["commit"],
            "files": [
                select(item, {"path", "installed_sha256", "upstream_sha256", "matches"})
                for item in source["files"]
            ],
            "all_match": source["all_match"],
        }
        hashes["pinned-source-verification.json"] = digest(source_path)
    provenance = {
        "source_file_sha256": hashes,
        "exporter": "scripts/export_readiness.py",
        "retained": "Numerical measurements, configuration, generated synthetic outputs and UTC times",
        "removed": [
            "cluster hostname and username",
            "absolute filesystem paths",
            "Slurm job ID",
            "GPU UUID and physical index",
            "unfiltered logs, process information and filesystem mount details",
        ],
        "timing_scope": "Call wall times are readiness diagnostics, not performance measurements",
    }
    args.output.mkdir(parents=True)
    for name, value in files.items():
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    provenance["exported_file_sha256"] = {name: digest(args.output / name) for name in files}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
