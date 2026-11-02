"""One fresh real AsyncLLM process for M3. Never supplies synthetic model output."""

import argparse
import asyncio
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/m3-real-model.json"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["native", "off", "prefix", "guarded"], required=True)
    p.add_argument("--phase", choices=["smoke", "diagnostic"], required=True)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


async def run(args):
    from m3_sources import sha256
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_prefix_scheduler.m3_events import emit, observation_errors

    config = json.loads(args.config.read_text())
    sources = json.loads(args.sources.read_text())
    frozen_inputs = json.loads(args.inputs.read_text())
    mode, phase = args.mode, args.phase
    cls = (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        if mode == "native"
        else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
    )
    actual_cls = cls
    if phase == "diagnostic":
        actual_cls = "vllm_prefix_scheduler.m3_observer." + (
            "ObserverNativeScheduler" if mode == "native" else "ObserverPrefixScheduler"
        )
    profile = config["baseline" if phase == "smoke" else "functional"].copy()
    profile.update(
        model=str(args.model),
        tokenizer=str(args.model),
        scheduler_cls=actual_cls,
        worker_extension_cls="vllm_prefix_scheduler.m3_worker.M3WorkerExtension",
    )
    if mode != "native":
        profile["additional_config"] = {
            "prefix_scheduler": {
                "mode": mode,
                "window": config["window"],
                "max_wait_ms": config["max_wait_ms"],
            }
        }
    emit(
        "engine_start",
        phase=phase,
        production_class=cls,
        scheduler_class=actual_cls,
        config_sha256=sha256(args.config),
        inputs_sha256=sha256(args.inputs),
        project_commit=sources["project_commit"],
        source_sha256=sources["source_sha256"],
        upstream_sha256=sources["upstream_sha256"],
        python_version=sources["python_version"],
        vllm_version=sources["packages"]["vllm"],
        requested_config=profile,
    )
    llm = None
    active = set()
    results = {}
    try:
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**profile))
        tokenizer = llm.get_tokenizer()
        cfg = llm.vllm_config
        resolved = {
            k: getattr(cfg.scheduler_config, k)
            for k in [
                "async_scheduling",
                "max_num_seqs",
                "max_num_active_seqs",
                "max_num_batched_tokens",
                "enable_chunked_prefill",
            ]
        }
        resolved.update(
            model=cfg.model_config.model,
            tokenizer=cfg.model_config.tokenizer,
            dtype=str(cfg.model_config.dtype),
            max_model_len=cfg.model_config.max_model_len,
            enable_prefix_caching=cfg.cache_config.enable_prefix_caching,
            enforce_eager=cfg.model_config.enforce_eager,
            gpu_memory_utilization=cfg.cache_config.gpu_memory_utilization,
            scheduler_class=f"{cfg.scheduler_config.get_scheduler_cls().__module__}.{cfg.scheduler_config.get_scheduler_cls().__name__}",
        )
        worker = await llm.collective_rpc("m3_snapshot", timeout=10)
        emit(
            "engine_ready",
            resolved_config=resolved,
            worker=worker,
            sources=sources,
            tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
            observation_errors=observation_errors(),
            client_unfinished=llm.get_num_unfinished_requests(),
        )
        if (
            resolved["async_scheduling"] is not True
            or resolved["enable_prefix_caching"] is not True
        ):
            raise RuntimeError("Required APC/async disabled")

        def tokenize(text):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise RuntimeError("Tokenizer did not return nonempty integer token IDs")
            expected = frozen_inputs["by_text"].get(text)
            if ids != expected:
                raise RuntimeError("Tokenizer IDs differ from pre-run frozen inputs")
            return {"text": text, "token_ids": list(ids)}

        common = {k: tokenize(v["text"]) for k, v in config["requests"].items()}

        async def consume(case_id, role, value, sampling, cancel=False):
            request_id = case_id + "::" + role
            active.add(request_id)
            emit("client_submit", request_id=request_id, case_id=case_id)
            last = None
            aborted = False
            try:
                async for output in llm.generate(
                    {"prompt_token_ids": value["token_ids"]},
                    SamplingParams(**sampling),
                    request_id=request_id,
                ):
                    part = output.outputs[0]
                    last = {
                        "token_ids": list(part.token_ids),
                        "finish_reason": part.finish_reason,
                        "finished": output.finished,
                        "text": part.text,
                        "num_cached_tokens": output.num_cached_tokens,
                    }
                    emit("client_output", request_id=request_id, case_id=case_id, **last)
                    if cancel and not aborted and part.token_ids:
                        if output.finished:
                            raise RuntimeError(
                                "Cancellation invalid: natural completion preceded abort"
                            )
                        emit("client_abort", request_id=request_id, case_id=case_id)
                        await llm.abort(request_id)
                        aborted = True
                if last is None or not last["finished"] or not last["token_ids"]:
                    raise RuntimeError("Missing real completed generated output")
                if cancel and not aborted:
                    raise RuntimeError("Cancellation did not occur")
                results[request_id] = last
                emit("client_done", request_id=request_id, case_id=case_id, **last)
            finally:
                active.discard(request_id)

        async def drain(case_id, label):
            deadline = asyncio.get_running_loop().time() + config["timing"]["drain_timeout_s"]
            while True:
                snapshots = await llm.collective_rpc(
                    "m3_snapshot",
                    timeout=10,
                    kwargs={
                        "label": f"{case_id}:{label}" if case_id else label,
                        "case_id": case_id,
                    },
                )
                client_count = llm.get_num_unfinished_requests()
                # Diagnostic RPC reads the actual local scheduler and runner in UniProcExecutor.
                drained = (
                    all(s.get("drained", False) for s in snapshots)
                    if phase == "diagnostic"
                    else all(s.get("drained", False) for s in snapshots)
                )
                if client_count == 0 and drained:
                    emit(
                        "client_drain",
                        case_id=case_id,
                        label=label,
                        unfinished=client_count,
                        snapshots=snapshots,
                    )
                    return
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError("Native resources did not drain")
                await asyncio.sleep(0.05)

        async def enqueued(request_id):
            deadline = asyncio.get_running_loop().time() + config["timing"]["enqueue_timeout_s"]
            while True:
                for path in args.output.glob("*.jsonl"):
                    for line in path.read_text().splitlines():
                        try:
                            e = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if e.get("event") == "enqueue" and e.get("request_id") == request_id:
                            emit(
                                "client_enqueue_confirmed",
                                request_id=request_id,
                                engine_pid=e["pid"],
                                engine_seq=e["seq"],
                            )
                            return
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError("No successful EngineCore add acknowledgement")
                await asyncio.sleep(0.01)

        async def case(scenario, variant, repeat):
            case_id = (
                f"smoke:{mode}"
                if scenario == "smoke"
                else "cancel:guarded"
                if scenario == "cancel"
                else f"mechanism:{variant}:{repeat}"
            )
            sampling = dict(config["sampling"])
            if scenario == "smoke":
                inputs = {
                    f"S{i}": tokenize(f"Give one short fact about {x}.")
                    for i, x in enumerate(["GPUs", "oceans", "planets"])
                }
            elif scenario == "cancel":
                inputs = {"X": common["B"]}
                sampling["max_tokens"] = 512
            else:
                inputs = common
            fields = dict(
                case_id=case_id,
                scenario=scenario,
                variant=(
                    "production"
                    if scenario == "smoke"
                    else "client_abort"
                    if scenario == "cancel"
                    else variant
                ),
                repeat=repeat,
            )
            emit("case_begin", inputs=inputs, sampling=sampling, **fields)
            tasks = []
            try:
                await drain(case_id, "before_reset")
                if not await llm.reset_prefix_cache():
                    raise RuntimeError("Native cache reset failed")
                emit("cache_reset_success", case_id=case_id)
                await drain(case_id, "initial")
                if scenario == "smoke":
                    tasks = [
                        asyncio.create_task(consume(case_id, role, value, sampling))
                        for role, value in inputs.items()
                    ]
                    await asyncio.gather(*tasks)
                elif scenario == "cancel":
                    await consume(case_id, "X", inputs["X"], sampling, cancel=True)
                else:
                    await consume(case_id, "A", inputs["A"], sampling)
                    await drain(case_id, "warm_drained")
                    await llm.pause_generation(mode="keep", clear_cache=False)
                    emit("pause_success", case_id=case_id)
                    tasks.append(asyncio.create_task(consume(case_id, "B", inputs["B"], sampling)))
                    await enqueued(case_id + "::B")
                    if variant == "guarded_expired":
                        await asyncio.sleep(config["timing"]["old_wait_s"])
                    tasks.append(asyncio.create_task(consume(case_id, "C", inputs["C"], sampling)))
                    await enqueued(case_id + "::C")
                    await llm.resume_generation()
                    emit("resume_success", case_id=case_id)
                    await asyncio.gather(*tasks)
                await drain(case_id, "drained")
                emit("case_end", status="passed", **fields)
            except BaseException as error:
                emit(
                    "case_end",
                    status="failed",
                    error_type=type(error).__name__,
                    error=str(error),
                    **fields,
                )
                for request_id in tuple(active):
                    await llm.abort(request_id)
                await llm.resume_generation()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                raise

        if phase == "smoke":
            await asyncio.wait_for(case("smoke", mode, 0), config["timing"]["case_timeout_s"])
        else:
            variants = ["guarded_unexpired", "guarded_expired"] if mode == "guarded" else [mode]
            for variant in variants:
                for repeat in range(args.repeats):
                    await asyncio.wait_for(
                        case("mechanism", variant, repeat), config["timing"]["case_timeout_s"]
                    )
            if mode == "guarded":
                await asyncio.wait_for(
                    case("cancel", "guarded", 0), config["timing"]["case_timeout_s"]
                )
        await drain(None, "before_shutdown")
        emit("engine_completed", request_count=len(results), results=results)
    finally:
        if llm is not None:
            for request_id in tuple(active):
                await llm.abort(request_id)
            llm.shutdown(timeout=15)
            emit(
                "engine_shutdown",
                client_unfinished=llm.get_num_unfinished_requests(),
                observation_errors=observation_errors(),
            )
            if observation_errors():
                raise RuntimeError("M3 observer errors invalidate evidence")


def main():
    if not __debug__:
        raise RuntimeError("Optimized interpreter prohibited for M3 validation")
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("M3 inference requires an existing Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        M3_TRACE_DIR=str(args.output.resolve()),
        M3_ENGINE_ID=f"{args.phase}-{args.mode}",
        M3_MODE=args.mode,
        M3_PHASE=args.phase,
    )
    config = json.loads(args.config.read_text())
    for name, value in config["environment"].items():
        os.environ[name] = value
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
