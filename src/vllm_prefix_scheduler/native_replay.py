"""Pinned host-side AsyncScheduler experiment; never computes KV tensors.

Native construction follows vLLM tests/v1/core/utils.py and FIFO output delivery
follows vllm/v1/engine/core.py:step_with_batch_queue, at the commit in the config.
No scheduler/cache algorithm is copied or substituted. See the research record.
"""

import argparse
import functools
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DispatchLedger:
    """Count dispatched positions, including overlaps after native preemption."""

    def __init__(self):
        self.seen = {}

    def record(self, request_id, start, count, prompt_length):
        if start < 0 or count <= 0:
            raise ValueError("Invalid dispatched interval")
        seen = self.seen.setdefault(request_id, set())
        positions = set(range(start, start + count))
        prompt = {p for p in positions if p < prompt_length}
        decode = positions - prompt
        result = {
            "start": start,
            "end": start + count,
            "scheduled_tokens": count,
            "prefill_tokens": len(prompt),
            "repeated_prefill_tokens": len(prompt & seen),
            "decode_tokens": len(decode),
            "repeated_decode_tokens": len(decode & seen),
        }
        seen.update(positions)
        return result


def request_state(request):
    return {
        "status": request.status.name,
        "computed_optimistic": request.num_computed_tokens,
        "in_flight": request.num_in_flight_tokens,
        "output_placeholders": request.num_output_placeholders,
        "stale_output_tokens": request.num_stale_output_tokens,
        "output_tokens": list(request.output_token_ids),
        "preemptions": request.num_preemptions,
        "is_prefill_chunk": request.is_prefill_chunk,
    }


class Observation:
    """Instance wrappers call each native method exactly once, unchanged.

    Snapshots read fields and the free-list traversal helper. They never perform
    a cache lookup, touch, allocation, statistics reset, or event collection.
    """

    def __init__(self, scheduler, requests, emit):
        self.scheduler = scheduler
        self.requests = requests
        self.emit = emit
        self.step = 0
        self.owner = None
        self.ledger = DispatchLedger()
        self.dispatches = []
        self.calls = 0
        self.originals = []
        manager = scheduler.kv_cache_manager
        for name in ("get_computed_blocks", "allocate_slots", "cache_blocks", "free"):
            self.wrap(manager, name)
        for name in ("get_new_blocks", "_maybe_evict_cached_block", "touch", "free_blocks"):
            self.wrap(manager.block_pool, name)
        self.wrap(scheduler, "_preempt_request")
        original = scheduler._update_after_schedule

        @functools.wraps(original)
        def advance(output):
            intervals = []
            for rid, count in output.num_scheduled_tokens.items():
                request = scheduler.requests[rid]
                interval = {
                    "request_id": rid,
                    "step": self.step,
                    "preemptions": request.num_preemptions,
                    **self.ledger.record(
                        rid, request.num_computed_tokens, count, request.num_prompt_tokens
                    ),
                }
                intervals.append(interval)
                self.dispatches.append(interval)
            self.event("dispatch", intervals=intervals)
            result = original(output)
            self.event("advance", requests=self.states())
            return result

        self.originals.append((scheduler, "_update_after_schedule", original))
        scheduler._update_after_schedule = advance

    def event(self, kind, **fields):
        self.emit({"event": kind, "step": self.step, **fields})

    def states(self):
        return {rid: request_state(r) for rid, r in self.requests.items()}

    def snapshot(self):
        scheduler = self.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        mappings = {}
        for key, blocks in pool.cached_block_hash_to_block._cache.items():
            mappings[key.hex()] = sorted(blocks) if isinstance(blocks, dict) else [blocks.block_id]
        return {
            "waiting": [r.request_id for r in scheduler.waiting],
            "kv_holding_waiting": [r.request_id for r in scheduler.kv_holding_waiting],
            "deferred_waiting": sorted(r.request_id for r in scheduler.deferred_waiting),
            "running": [r.request_id for r in scheduler.running],
            "requests": self.states(),
            "blocks": [
                {
                    "id": b.block_id,
                    "ref_cnt": b.ref_cnt,
                    "hash": b.block_hash.hex() if b.block_hash is not None else None,
                    "null": b.is_null,
                }
                for b in pool.blocks
            ],
            "hash_mapping": mappings,
            "free_queue": [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
            "request_blocks": [
                {rid: [b.block_id for b in bs] for rid, bs in m.req_to_blocks.items()}
                for m in scheduler.kv_cache_manager.coordinator.single_type_managers
            ],
        }

    def wrap(self, obj, name):
        original = getattr(obj, name)
        self.originals.append((obj, name, original))

        @functools.wraps(original)
        def observed(*args, **kwargs):
            previous = self.owner
            if name in {"allocate_slots", "get_computed_blocks", "cache_blocks", "free"}:
                self.owner = args[0].request_id
            self.calls += 1
            call = self.calls
            details = {}
            if name == "_maybe_evict_cached_block":
                details["block_id"] = args[0].block_id
            if name == "get_new_blocks":
                details["num_blocks"] = args[0]
            if name == "allocate_slots":
                details["num_new_tokens"] = kwargs.get(
                    "num_new_tokens", args[1] if len(args) > 1 else None
                )
                details["num_new_computed_tokens"] = kwargs.get("num_new_computed_tokens", 0)
            self.event(
                "call",
                method=name,
                call=call,
                owner=self.owner,
                details=details,
                state=self.snapshot(),
            )
            try:
                result = original(*args, **kwargs)
                returned = None
                if name == "get_computed_blocks":
                    returned = {
                        "blocks": result[0].get_block_ids(),
                        "hit_tokens": result[1],
                        "shared_prefix_boundary": result[2],
                    }
                elif name == "allocate_slots":
                    returned = {
                        "success": result is not None,
                        "blocks": result.get_block_ids() if result is not None else None,
                    }
                elif name == "get_new_blocks":
                    returned = [b.block_id for b in result]
                elif name == "_maybe_evict_cached_block":
                    returned = result
                self.event(
                    "return",
                    method=name,
                    call=call,
                    owner=self.owner,
                    result=returned,
                    state=self.snapshot(),
                )
                return result
            finally:
                self.owner = previous

        setattr(obj, name, observed)

    def restore(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def runtime_record(config, model_path):
    import vllm

    root = Path(vllm.__file__).resolve().parent.parent
    modules = {}
    for relative, expected in config["source_sha256"].items():
        path = root / relative
        actual = sha256_file(path)
        modules[relative] = {
            "path": str(path),
            "sha256": actual,
            "pinned_sha256": expected,
            "matches": actual == expected,
        }
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "vllm_version": vllm.__version__,
        "vllm_file": vllm.__file__,
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_commit": config["vllm_commit"],
        "modules": modules,
        "model_config": {
            "path": str(model_path / "config.json"),
            "sha256": sha256_file(model_path / "config.json"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "VLLM_USE_V2_MODEL_RUNNER",
                "HF_HUB_OFFLINE",
                "SLURM_JOB_ID",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "substitutions": [
            "Synthetic ModelRunnerOutput only; no model runner, model weights, tokenizer, or KV tensors",
            "One-layer FullAttentionSpec metadata fixture (not model-derived KV geometry)",
            "One recorded waiting-queue permutation only in C,B test fixture",
        ],
        "mocks_or_stubs": [],
    }


def create_scheduler(config, capacity, model_path):
    import torch
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    model = ModelConfig(
        model=str(model_path),
        dtype="bfloat16",
        seed=config["seed"],
        skip_tokenizer_init=True,
        max_model_len=config["max_model_len"],
        enforce_eager=True,
    )
    scheduling = SchedulerConfig(
        max_num_seqs=config["max_num_seqs"],
        max_num_active_seqs=config["max_num_active_seqs"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_model_len=config["max_model_len"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        async_scheduling=True,
        is_encoder_decoder=model.is_encoder_decoder,
        policy="fcfs",
        watermark=config["watermark"],
        scheduler_reserve_full_isl=config["scheduler_reserve_full_isl"],
        long_prefill_token_threshold=config["long_prefill_token_threshold"],
        long_prefill_token_threshold_adaptive=config["long_prefill_token_threshold_adaptive"],
    )
    cache = CacheConfig(
        block_size=config["block_size"],
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256",
    )
    cache.num_gpu_blocks = capacity
    vconfig = VllmConfig(
        model_config=model,
        scheduler_config=scheduling,
        cache_config=cache,
        parallel_config=ParallelConfig(),
    )
    register_all_kvcache_specs(vconfig)
    spec = FullAttentionSpec(
        block_size=config["block_size"], num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv = KVCacheConfig(
        num_blocks=capacity,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["fixture_layer"], spec)],
    )
    scheduler = AsyncScheduler(
        vllm_config=vconfig,
        kv_cache_config=kv,
        block_size=config["block_size"],
        structured_output_manager=StructuredOutputManager(vconfig),
        log_stats=False,
    )
    assert type(scheduler) is AsyncScheduler and scheduler.use_v2_model_runner
    return scheduler


def create_requests(config):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hasher = get_request_block_hasher(config["block_size"], sha256)
    return {
        item["id"]: Request(
            request_id=item["id"],
            prompt_token_ids=item["tokens"].copy(),
            arrival_time=item["arrival_time"],
            block_hasher=hasher,
            pooling_params=None,
            sampling_params=SamplingParams(
                max_tokens=config["max_tokens"], ignore_eos=config["ignore_eos"], temperature=0
            ),
        )
        for item in config["requests"]
    }


def drive(config, observation, phase):
    """Bounded FIFO of synthetic outputs with two outstanding native steps."""
    from vllm.v1.outputs import ModelRunnerOutput

    scheduler = observation.scheduler
    pending = deque()
    maximum = 0
    for _ in range(config["max_steps_per_phase"]):
        if not scheduler.has_requests() and not pending:
            observation.event(
                "drained", phase=phase, pending_outputs=0, state=observation.snapshot()
            )
            return maximum
        if scheduler.has_requests() and len(pending) < config["pending_output_depth"]:
            observation.step += 1
            observation.event(
                "schedule_before",
                phase=phase,
                pending_outputs=len(pending),
                state=observation.snapshot(),
            )
            output = scheduler.schedule()
            ids = list(output.num_scheduled_tokens)
            # Capture per-step chunk state NOW; mutable request state can advance
            # before this output is delivered.
            sampled = [
                []
                if scheduler.requests[rid].is_prefill_chunk
                else [config["synthetic_output_token_id"]]
                for rid in ids
            ]
            pending.append(
                (
                    observation.step,
                    output,
                    ModelRunnerOutput(
                        req_ids=ids,
                        req_id_to_index={rid: i for i, rid in enumerate(ids)},
                        sampled_token_ids=sampled,
                        logprobs=None,
                        prompt_logprobs_dict={},
                        pooler_output=[],
                    ),
                )
            )
            maximum = max(maximum, len(pending))
            observation.event(
                "schedule_after",
                phase=phase,
                admitted=[r.req_id for r in output.scheduled_new_reqs],
                scheduled=output.num_scheduled_tokens,
                finished_ids=sorted(output.finished_req_ids),
                sampled_token_ids=sampled,
                pending_outputs=len(pending),
                state=observation.snapshot(),
            )
            if len(pending) < config["pending_output_depth"] and output.total_num_scheduled_tokens:
                continue
        if pending:
            step, output, synthetic = pending.popleft()
            observation.event(
                "output_before",
                output_step=step,
                pending_outputs=len(pending) + 1,
                state=observation.snapshot(),
            )
            scheduler.update_from_output(output, synthetic)
            observation.event(
                "output_after",
                output_step=step,
                pending_outputs=len(pending),
                state=observation.snapshot(),
            )
    raise AssertionError(f"{phase}: finite step limit exceeded; trace retained")


def replay_case(config, capacity_name, order, model_path, emit):
    scheduler = create_scheduler(config, config["capacities"][capacity_name], model_path)
    requests = create_requests(config)
    observation = Observation(scheduler, requests, emit)
    sources = {}
    for obj in (
        scheduler,
        requests["A"],
        scheduler.kv_cache_manager,
        scheduler.kv_cache_manager.block_pool,
        scheduler.waiting,
    ):
        cls = type(obj)
        sources[cls.__name__] = {
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "path": inspect.getfile(cls),
            "sha256": sha256_file(inspect.getfile(cls)),
        }
    observation.event("native_objects", sources=sources)
    observation.event(
        "effective_config",
        scheduler={
            key: getattr(scheduler.scheduler_config, key)
            for key in (
                "max_num_seqs",
                "max_num_active_seqs",
                "max_num_batched_tokens",
                "max_num_scheduled_tokens",
                "enable_chunked_prefill",
                "async_scheduling",
                "policy",
                "watermark",
                "scheduler_reserve_full_isl",
                "long_prefill_token_threshold",
                "long_prefill_token_threshold_adaptive",
            )
        },
        cache={
            "block_size": scheduler.block_size,
            "num_blocks": scheduler.kv_cache_config.num_blocks,
            "prefix_caching": scheduler.cache_config.enable_prefix_caching,
            "kv_events": scheduler.enable_kv_cache_events,
        },
        parallel={
            "tensor": scheduler.parallel_config.tensor_parallel_size,
            "pipeline": scheduler.parallel_config.pipeline_parallel_size,
            "decode_context": scheduler.dcp_world_size,
            "prefill_context": scheduler.pcp_world_size,
        },
        use_v2_model_runner=scheduler.use_v2_model_runner,
        max_concurrent_batches=scheduler.vllm_config.max_concurrent_batches,
        max_in_flight_tokens=scheduler.vllm_config.max_in_flight_tokens,
        max_model_len=scheduler.max_model_len,
        speculative_tokens=scheduler.num_spec_tokens,
        lookahead_tokens=scheduler.num_lookahead_tokens,
    )
    try:
        scheduler.add_request(requests["A"])
        observation.event("enqueue", ids=["A"], arrivals={"A": requests["A"].arrival_time})
        warm_depth = drive(config, observation, "warmup")
        warm = observation.snapshot()
        a_blocks = [b for b in warm["blocks"] if b["hash"] is not None]
        assert len(a_blocks) == 2
        assert all(
            b["ref_cnt"] == 0
            and b["id"] in warm["free_queue"]
            and b["id"] in warm["hash_mapping"][b["hash"]]
            for b in a_blocks
        )
        observation.event("A_cached_and_evictable", cached_blocks=a_blocks, state=warm)
        for rid in ("B", "C"):
            scheduler.add_request(requests[rid])
        observation.event(
            "enqueue",
            ids=["B", "C"],
            arrivals={rid: requests[rid].arrival_time for rid in ("B", "C")},
            state=observation.snapshot(),
        )
        if order == ["C", "B"]:
            b = scheduler.waiting.pop_request()
            c = scheduler.waiting.pop_request()
            assert [b.request_id, c.request_id] == ["B", "C"]
            scheduler.waiting.add_request(c)
            scheduler.waiting.add_request(b)
            observation.event(
                "test_only_reorder",
                before=["B", "C"],
                after=order,
                count=1,
                state=observation.snapshot(),
            )
        depth = drive(config, observation, "BC")
        final = observation.snapshot()
        assert not scheduler.requests and not scheduler.running and not scheduler.waiting
        assert not scheduler.kv_holding_waiting and not scheduler.deferred_waiting
        assert all(not blocks for group in final["request_blocks"] for blocks in group.values())
        assert len(final["free_queue"]) == config["capacities"][capacity_name] - 1
        assert all(b["ref_cnt"] == 0 for b in final["blocks"] if not b["null"])
        assert all(
            r.is_finished()
            and r.num_output_tokens == config["max_tokens"]
            and r.num_output_placeholders == 0
            and r.num_in_flight_tokens == 0
            and r.num_stale_output_tokens == 0
            for r in requests.values()
        )
        per_request = {}
        for rid, request in requests.items():
            dispatches = [d for d in observation.dispatches if d["request_id"] == rid]
            per_request[rid] = {
                key: sum(d[key] for d in dispatches)
                for key in (
                    "prefill_tokens",
                    "decode_tokens",
                    "repeated_prefill_tokens",
                    "repeated_decode_tokens",
                )
            }
            per_request[rid].update(
                preemptions=request.num_preemptions, final_status=request.status.name
            )
        return {
            "capacity": capacity_name,
            "num_blocks": config["capacities"][capacity_name],
            "order": order,
            "requests": per_request,
            "max_pending_outputs": max(depth, warm_depth),
            "final_pending_outputs": 0,
            "all_references_released": True,
        }
    finally:
        observation.restore()
        scheduler.shutdown()


def verify_matrix(results):
    """Check repeated dispatch counts; causal trace acceptance is a separate gate."""
    expected = {
        ("pressure", "BC"): 33,
        ("pressure", "CB"): 1,
        ("ample", "BC"): 1,
        ("ample", "CB"): 1,
    }
    for result in results:
        assert (
            result["requests"]["C"]["prefill_tokens"]
            == expected[(result["capacity"], "".join(result["order"]))]
        )
        assert result["requests"]["A"]["prefill_tokens"] == 33
        assert result["requests"]["B"]["prefill_tokens"] == 49
        assert all(
            r["decode_tokens"] == 2
            and r["repeated_prefill_tokens"] == 0
            and r["repeated_decode_tokens"] == 0
            and r["preemptions"] == 0
            for r in result["requests"].values()
        )
    for capacity in ("pressure", "ample"):
        for order in (["B", "C"], ["C", "B"]):
            repeated = [r for r in results if r["capacity"] == capacity and r["order"] == order]
            assert len(repeated) >= 3 and all(
                r["requests"] == repeated[0]["requests"] for r in repeated
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/native-bc-replay.json"))
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Existing local model directory; reads metadata only",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    write_json(args.output / "inputs.json", config)
    results = []
    try:
        if platform.system() != "Linux":
            raise RuntimeError("Native replay requires the pinned Linux runtime; no skip-success")
        for key, expected in {
            "PYTHONHASHSEED": str(config["seed"]),
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "HF_HUB_OFFLINE": "1",
        }.items():
            if os.environ.get(key) != expected:
                raise RuntimeError(f"Set {key}={expected} before launching the interpreter")
        runtime = runtime_record(config, args.model_config)
        write_json(args.output / "runtime.json", runtime)
        assert runtime["vllm_version"] == config["vllm_version"]
        assert runtime["model_config"]["sha256"] == config["model_config_sha256"]
        assert all(item["matches"] for item in runtime["modules"].values()), (
            "Pinned source mismatch"
        )
        with (args.output / "trace.jsonl").open("x") as trace:
            for repeat in range(config["repeats"]):
                for capacity in config["capacities"]:
                    for order in config["orders"]:
                        case = f"{repeat}-{capacity}-{''.join(order)}"

                        def emit(event, case=case):
                            trace.write(json.dumps({"case": case, **event}, sort_keys=True) + "\n")
                            trace.flush()

                        result = replay_case(config, capacity, order, args.model_config, emit)
                        result.update(case=case, repeat=repeat)
                        results.append(result)
                        write_json(
                            args.output / "summary.json",
                            {"metric": config["metric"], "cases": results, "verified": False},
                        )
        verify_matrix(results)
        write_json(
            args.output / "summary.json",
            {"metric": config["metric"], "cases": results, "verified": True},
        )
        print(json.dumps({"native_cases": len(results), "matrix_verified": True}))
    except BaseException:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
