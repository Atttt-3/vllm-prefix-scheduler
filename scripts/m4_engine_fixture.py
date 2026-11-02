"""One real M4 AsyncLLM engine: no synthetic outputs or active observer controls."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/m4-stability.json"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("native", "off", "prefix", "guarded"), required=True)
    p.add_argument("--calibration", action="store_true")
    return p


class JournalReader:
    """Read appended passive events without slowing or synchronizing EngineCore."""

    def __init__(self, root):
        self.root = root
        self.positions = {}

    def read(self):
        rows = []
        for path in self.root.glob("*.jsonl"):
            position = self.positions.get(path, 0)
            with path.open() as handle:
                handle.seek(position)
                while True:
                    start = handle.tell()
                    line = handle.readline()
                    if not line or not line.endswith("\n"):
                        handle.seek(start)
                        break
                    rows.append(json.loads(line))
                self.positions[path] = handle.tell()
        return rows


async def run(args):
    from vllm import SamplingParams
    from vllm.config import KVEventsConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_prefix_scheduler import m3_events as events

    config = json.loads(args.config.read_text())
    sources = json.loads(args.sources.read_text())
    frozen = json.loads(args.inputs.read_text())
    inputs = frozen["requests"]
    phase = "calibration" if args.calibration else "formal"

    def emit(event, **fields):
        events.emit(event, task_id=config["task_id"], layer="gpu", phase=phase, **fields)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for name, value in inputs.items():
        if not value["token_ids"] or len(value["token_ids"]) != value["length"]:
            raise RuntimeError(f"invalid frozen tokens: {name}")
    cls = "vllm_prefix_scheduler.m4_observer." + (
        "M4NativeScheduler" if args.mode == "native" else "M4PrefixScheduler"
    )
    profile = dict(config["gpu_profile"])
    profile.update(
        model=str(args.model),
        tokenizer=str(args.model),
        scheduler_cls=cls,
        worker_extension_cls="vllm_prefix_scheduler.m4_observer.M4WorkerExtension",
    )
    if args.mode != "native":
        profile["additional_config"] = {
            "prefix_scheduler": {
                "mode": args.mode,
                "window": config["window"],
                "max_wait_ms": config["max_wait_ms"],
            }
        }
    emit(
        "m4_engine_start",
        mode=args.mode,
        config_sha256=sha(args.config),
        inputs_sha256=sha(args.inputs),
        project_commit=sources["project_commit"],
        source_sha256=sources["source_sha256"],
        upstream_sha256=sources["upstream_sha256"],
        python_version=sources["python_version"],
        vllm_version=sources["packages"]["vllm"],
        requested_config=profile,
        model_revision=config["model_revision"],
        tokenizer_revision=config["tokenizer_revision"],
    )
    llm = None
    active = set()
    tasks = []
    journal = JournalReader(args.output)
    observations = []
    try:
        engine_profile = dict(profile)
        engine_profile["kv_events_config"] = KVEventsConfig(**profile["kv_events_config"])
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_profile))
        worker = await llm.collective_rpc("m4_snapshot", timeout=15)
        emit(
            "m4_engine_ready",
            mode=args.mode,
            worker=worker,
            sources=sources,
            actual_tokenizer_class=f"{type(llm.get_tokenizer()).__module__}.{type(llm.get_tokenizer()).__name__}",
        )

        async def consume(case_id, rid, input_key, sampling, generation_label):
            active.add(rid)
            value = inputs[input_key]
            emit(
                "m4_client_submit",
                case_id=case_id,
                request_id=rid,
                input_key=input_key,
                token_ids=value["token_ids"],
                sampling=sampling,
                generation_label=generation_label,
            )
            last = None
            try:
                async for output in llm.generate(
                    {"prompt_token_ids": value["token_ids"]},
                    SamplingParams(**sampling),
                    request_id=rid,
                ):
                    part = output.outputs[0]
                    last = {
                        "token_ids": list(part.token_ids),
                        "finished": output.finished,
                        "finish_reason": part.finish_reason,
                        "num_cached_tokens": output.num_cached_tokens,
                    }
                    emit(
                        "m4_client_output",
                        case_id=case_id,
                        request_id=rid,
                        generation_label=generation_label,
                        **last,
                    )
                if last is None or not last["finished"]:
                    raise RuntimeError("real client stream did not reach a terminal output")
                emit(
                    "m4_client_done",
                    case_id=case_id,
                    request_id=rid,
                    generation_label=generation_label,
                    **last,
                )
                return last
            finally:
                active.discard(rid)

        async def until(predicate, label):
            deadline = asyncio.get_running_loop().time() + config["timing"]["drain_timeout_s"]
            while True:
                new = journal.read()
                observations.extend(new)
                for row in observations:
                    if predicate(row):
                        return row
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"M4 passive condition missing: {label}")
                await asyncio.sleep(0.002)

        async def enqueue(rid):
            return await until(
                lambda row: (
                    row.get("event") == "m4_request_bound"
                    and row.get("request", {}).get("request_id") == rid
                ),
                "actual enqueue",
            )

        async def drain(case_id, label):
            deadline = asyncio.get_running_loop().time() + config["timing"]["drain_timeout_s"]
            while True:
                snapshots = await llm.collective_rpc(
                    "m4_snapshot", timeout=15, kwargs={"label": label, "case_id": case_id}
                )
                if llm.get_num_unfinished_requests() == 0 and all(s["drained"] for s in snapshots):
                    for snapshot in snapshots:
                        emit(
                            "m4_client_drain",
                            case_id=case_id,
                            label=label,
                            resources=snapshot["scheduler"],
                            observer=snapshot["m4"],
                        )
                    return snapshots[0]["scheduler"]
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("M4 actual native resources failed to drain")
                await asyncio.sleep(0.02)

        async def do_case(case):
            nonlocal tasks, observations
            scenario = case["scenario"]
            case_id = f"{scenario}:{case['mode']}:{case['repeat']}"
            tasks = []
            observations = []
            journal.read()
            sampling = {**config["sampling"], "seed": case["seed"]}
            try:
                await llm.collective_rpc(
                    "m4_set_case", timeout=15, kwargs={"case_id": case_id, "metadata": case}
                )
                await drain(case_id, "before_reset")
                if not await llm.reset_prefix_cache():
                    raise RuntimeError("M4 idle native cache reset failed")
                emit(
                    "m4_cache_reset",
                    case_id=case_id,
                    active_client_count=llm.get_num_unfinished_requests(),
                )
                initial_resources = await drain(case_id, "initial")
                emit(
                    "m4_case_begin",
                    case_id=case_id,
                    **case,
                    sampling=sampling,
                    initial_resources=initial_resources,
                    interventions=["idle_native_cache_reset"]
                    + (
                        ["native_pause_keep"]
                        if scenario
                        in {"capacity_preempt_resume", "waiting_cancel_reuse", "full_events_probe"}
                        else []
                    )
                    + (["normal_client_abort"] if "cancel" in scenario else []),
                )
                if scenario == "chunk_complete":
                    await consume(case_id, case_id + "::chunk", "long_a", sampling, "initial")
                elif scenario == "capacity_preempt_resume":
                    await llm.pause_generation(mode="keep", clear_cache=False)
                    emit("m4_pause_success", case_id=case_id, intervention="native_pause_keep")
                    for role, key in (("a", "long_a"), ("b", "long_b")):
                        rid = case_id + "::" + role
                        tasks.append(
                            asyncio.create_task(consume(case_id, rid, key, sampling, "initial"))
                        )
                        await enqueue(rid)
                    await llm.resume_generation()
                    emit("m4_resume_success", case_id=case_id)
                    await asyncio.gather(*tasks)
                elif scenario in {"waiting_cancel_reuse", "pending_cancel_reuse"}:
                    rid = case_id + "::reused"
                    if scenario == "waiting_cancel_reuse":
                        await llm.pause_generation(mode="keep", clear_cache=False)
                        emit("m4_pause_success", case_id=case_id, intervention="native_pause_keep")
                    long_sampling = {
                        **sampling,
                        "max_tokens": config["inputs"]["pending_cancel_max_output"],
                    }
                    task = asyncio.create_task(
                        consume(case_id, rid, "long_a", long_sampling, "cancelled")
                    )
                    tasks.append(task)
                    await enqueue(rid)
                    if scenario == "pending_cancel_reuse":
                        await until(
                            lambda row: (
                                row.get("event") == "m4_schedule"
                                and any(r["request_id"] == rid for r in row.get("intervals", []))
                            ),
                            "real pending dispatch",
                        )
                    emit(
                        "m4_client_abort",
                        case_id=case_id,
                        request_id=rid,
                        intervention="normal_client_abort",
                    )
                    await llm.abort(rid)
                    if scenario == "waiting_cancel_reuse":
                        await llm.resume_generation()
                        emit("m4_resume_success", case_id=case_id)
                    result = await task
                    if result["finish_reason"] != "abort":
                        raise RuntimeError("M4 cancellation raced natural completion")
                    await drain(case_id, "old_generation")
                    # The actual abort-entry condition is assessed from its raw row.
                    observations.extend(journal.read())
                    entries = [
                        row
                        for row in observations
                        if row.get("event") == "m4_cancel_start"
                        and any(r["request_id"] == rid for r in row.get("requests", []))
                    ]
                    if not entries:
                        raise RuntimeError("no actual EngineCore cancellation entry")
                    valid = False
                    for entry in entries:
                        request = next(r for r in entry["requests"] if r["request_id"] == rid)
                        matching_pending = any(
                            any(r["request_id"] == rid for r in batch["intervals"])
                            for batch in entry["pending"]
                        )
                        valid |= (
                            request["status"] == "WAITING" and request["num_computed_tokens"] == 0
                            if scenario == "waiting_cancel_reuse"
                            else request["status"] == "RUNNING"
                            and request["num_in_flight_tokens"] > 0
                            and matching_pending
                        )
                    if not valid:
                        raise RuntimeError("actual cancellation precondition was not met")
                    await consume(case_id, rid, "short", sampling, "reused")
                elif scenario == "full_events_probe":
                    await consume(case_id, case_id + "::warm", "hot", sampling, "warm")
                    await drain(case_id, "warm")
                    await llm.pause_generation(mode="keep", clear_cache=False)
                    emit("m4_pause_success", case_id=case_id, intervention="native_pause_keep")
                    rid = case_id + "::probe"
                    tasks.append(
                        asyncio.create_task(consume(case_id, rid, "hot", sampling, "probe"))
                    )
                    await enqueue(rid)
                    probes = await llm.collective_rpc(
                        "m4_probe", timeout=20, kwargs={"case_id": case_id, "request_id": rid}
                    )
                    if not all(p["equal"] and p["score"] > 0 for p in probes):
                        raise RuntimeError("M4 real full-event probe changed storage or had no hit")
                    await llm.resume_generation()
                    emit("m4_resume_success", case_id=case_id)
                    await asyncio.gather(*tasks)
                else:
                    raise ValueError(f"unknown frozen GPU scenario {scenario}")
                await drain(case_id, "main_complete")
                await consume(
                    case_id, case_id + "::continuation", "short", sampling, "continuation"
                )
                resources = await drain(case_id, "final")
                emit("m4_case_end", case_id=case_id, **case, resources=resources, status="passed")
            except BaseException as error:
                emit(
                    "m4_case_failure",
                    case_id=case_id,
                    **case,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise

        selected = [
            case
            for case in config["gpu_matrix"]
            if case["mode"] == args.mode and (not args.calibration or case["repeat"] == 0)
        ]
        if not selected:
            raise RuntimeError("M4 GPU matrix is empty")
        for case in selected:
            await asyncio.wait_for(do_case(case), config["timing"]["case_timeout_s"])
        await drain(None, "before_shutdown")
        emit("m4_engine_completed", mode=args.mode, case_count=len(selected))
    finally:
        if llm is not None:
            for rid in tuple(active):
                await llm.abort(rid)
            await asyncio.gather(*tasks, return_exceptions=True)
            llm.shutdown(timeout=15)
            emit(
                "m4_engine_shutdown",
                mode=args.mode,
                client_unfinished=llm.get_num_unfinished_requests(),
                observation_errors=events.observation_errors(),
            )
            if events.observation_errors():
                raise RuntimeError("M4 passive observation errors invalidate evidence")


def main():
    if not __debug__:
        raise RuntimeError("M4 refuses optimized interpreter")
    args = args_parser().parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("M4 GPU fixture requires an existing Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    os.environ.update(config["environment"])
    os.environ.update(
        M3_TRACE_DIR=str(args.output.resolve()),
        M3_ENGINE_ID=f"m4-{args.mode}",
        M3_MODE=args.mode,
        M3_PHASE="diagnostic",
        M4_PHASE="calibration" if args.calibration else "formal",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
