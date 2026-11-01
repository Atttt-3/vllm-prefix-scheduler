"""Minimal native vLLM validation; timings here are not a performance benchmark."""

import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path


def gpu_snapshot():
    query = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    rows = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        rows.append(
            dict(
                index=int(row[0]),
                uuid=row[1],
                name=row[2],
                total_mib=int(row[3]),
                used_mib=int(row[4]),
                free_mib=int(row[5]),
                utilization_percent=int(row[6]),
            )
        )
    return {"utc": datetime.now(UTC).isoformat(), "gpus": rows}


def output_record(output, label, elapsed):
    completion = output.outputs[0]
    return {
        "label": label,
        "request_id": output.request_id,
        "finished": output.finished,
        "prompt_tokens": len(output.prompt_token_ids),
        "output_tokens": len(completion.token_ids),
        "num_cached_tokens": output.num_cached_tokens,
        "finish_reason": completion.finish_reason,
        "text": completion.text,
        "call_wall_seconds_not_benchmark": elapsed,
    }


def main():
    from importlib.metadata import version

    from vllm import LLM, SamplingParams

    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
    pins = json.loads((root / "configs" / "upstream.json").read_text())
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(os.environ.get("RUN_DIR", root / "runs" / "readiness" / run_id)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "validation.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {result_path}")
    samples_path = run_dir / "gpu-memory-samples.jsonl"
    result = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version,
        "vllm": version("vllm"),
        "model_revision": pins["model"]["revision"],
        "project_commit": os.environ.get("PROJECT_COMMIT"),
        "project_dirty": os.environ.get("PROJECT_DIRTY"),
        "scope": "native minimal inference/APC/async readiness, not optimization benchmark",
        "requests": [],
    }

    def save():
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")

    stop = threading.Event()
    samples = []

    def sample_memory():
        with samples_path.open("w") as output:
            while not stop.is_set():
                try:
                    sample = gpu_snapshot()
                    samples.append(sample)
                    output.write(json.dumps(sample) + "\n")
                    output.flush()
                except Exception as exc:
                    output.write(json.dumps({"sampling_error": repr(exc)}) + "\n")
                stop.wait(0.5)

    result["gpu_before_engine"] = gpu_snapshot()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    llm = None
    try:
        args = json.loads((root / "configs" / "baseline-3090.json").read_text())
        args.update(
            model=str(root / pins["model"]["directory"]),
            tokenizer=str(root / pins["model"]["directory"]),
            worker_extension_cls="diagnostic_worker.ReadinessWorkerExtension",
        )
        result["requested_engine_args"] = args
        save()
        llm = LLM(**args)
        config = llm.llm_engine.vllm_config
        scheduler = config.scheduler_config.get_scheduler_cls()
        result["resolved_config"] = {
            "dtype": str(config.model_config.dtype),
            "max_model_len": config.model_config.max_model_len,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "scheduler_class": f"{scheduler.__module__}.{scheduler.__name__}",
            "enable_prefix_caching": config.cache_config.enable_prefix_caching,
            "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "enforce_eager": config.model_config.enforce_eager,
            "gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
        }
        result["worker_after_init"] = llm.collective_rpc("readiness_snapshot")
        result["gpu_after_engine_init"] = gpu_snapshot()
        worker = result["worker_after_init"][0]
        assert worker["model_parameter_dtype"] == "torch.bfloat16", worker
        assert worker["async_scheduling"] is True, worker
        assert worker["scheduler_class"].endswith(".AsyncScheduler"), worker
        assert worker["enable_prefix_caching"] is True, worker
        assert len(worker["kv_cache_groups"]) == 1, worker
        group = worker["kv_cache_groups"][0]
        assert group["spec_class"] == "FullAttentionSpec", group
        result["kv_capacity"] = {
            "num_gpu_blocks": worker["num_gpu_blocks"],
            "block_size_tokens": group["block_size"],
            "capacity_tokens": worker["num_gpu_blocks"] * group["block_size"],
            "allocation_bytes": worker["kv_allocation_bytes"],
            "allocation_gib": worker["kv_allocation_bytes"] / 1024**3,
            "full_16k_sequences_theoretical": worker["num_gpu_blocks"]
            * group["block_size"]
            / 16384,
        }
        assert result["kv_capacity"]["capacity_tokens"] >= 16384
        save()
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0, max_tokens=64, seed=42)

        def chat_prompt(text):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        def generate_one(prompt, label):
            started = time.monotonic()
            output = llm.generate([prompt], sampling, use_tqdm=False)[0]
            record = output_record(output, label, time.monotonic() - started)
            assert record["finished"] and record["output_tokens"] > 0, record
            assert record["text"].strip(), record
            result["requests"].append(record)
            save()
            print("REQUEST_RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
            return record

        generate_one(chat_prompt("请用一句简短的中文解释 GPU 的用途。"), "basic_smoke")
        llm.reset_prefix_cache()
        prefix = (
            "Public synthetic reference: GPU inference reuses identical prompt prefixes. "
            "This paragraph is repeated solely to test automatic prefix caching.\n"
        ) * 96
        prompt = chat_prompt(prefix + "\nSummarize the reference in one short sentence.")
        cold = generate_one(prompt, "prefix_cold")
        warm = generate_one(prompt, "prefix_warm_identical_prompt")
        assert warm["num_cached_tokens"] > 0, warm
        assert warm["num_cached_tokens"] > cold["num_cached_tokens"], (cold, warm)
        result["apc_behavior_verified"] = True
        batched_prompts = [
            chat_prompt(prefix + f"\nGive one short GPU fact. Query number {i}.") for i in range(4)
        ]
        started = time.monotonic()
        outputs = llm.generate(batched_prompts, sampling, use_tqdm=False)
        elapsed = time.monotonic() - started
        assert len(outputs) == 4
        for index, output in enumerate(outputs):
            record = output_record(output, f"batch_four_{index}", elapsed)
            assert record["finished"] and record["output_tokens"] > 0 and record["text"].strip(), (
                record
            )
            result["requests"].append(record)
        result["worker_after_requests"] = llm.collective_rpc("readiness_snapshot")
        result["gpu_after_requests"] = gpu_snapshot()
        result["status"] = "passed"
        save()
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as exc:
                result["shutdown_error"] = repr(exc)
        stop.set()
        sampler.join(timeout=12)
        result["gpu_memory_sampling_interval_seconds"] = 0.5
        result["gpu_memory_sample_count"] = len(samples)
        result["gpu_memory_sampled_peak_used_mib"] = max(
            (gpu["used_mib"] for sample in samples for gpu in sample["gpus"]),
            default=None,
        )
        result["finished_at_utc"] = datetime.now(UTC).isoformat()
        save()
        print("FINAL_RESULT " + json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
