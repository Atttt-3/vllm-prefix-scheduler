"""Replay one frozen M5 workload against one freshly initialized native AsyncLLM."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from m3_sources import collect_sources, sha256

from vllm_prefix_scheduler.m5_campaign import (
    D2R110_ROLE,
    D2R_ROLE,
    D3_ROLE,
    D8_ROLE,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_d2 import d2r110_plan, d2r_plan, d3_plan
from vllm_prefix_scheduler.m5_d3 import (
    bind_loaded_sources,
    monotonic_clock_info,
    snapshot_native_metrics,
    verify_clock_brackets,
)
from vllm_prefix_scheduler.m5_d4 import d4_plan
from vllm_prefix_scheduler.m5_d5 import d5_plan, d5_profile
from vllm_prefix_scheduler.m5_d5r import d5r_plan
from vllm_prefix_scheduler.m5_d6 import D6_RUNS, d6_plan, d6_sources
from vllm_prefix_scheduler.m5_d7 import D7_RUNS, d7_plan
from vllm_prefix_scheduler.m5_d8 import d8_plan
from vllm_prefix_scheduler.m5_workload import MODES

ROOT = Path(__file__).resolve().parents[1]
TIER = ("endpoint", "light", "profile")


def _dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _single_worker(values):
    if len(values) != 1:
        raise RuntimeError("M5 replay requires exactly one worker")
    return values[0]


def _binding(args, config):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not __debug__ or sys.platform != "linux" or not job_id.isdigit():
        raise RuntimeError("M5 replay requires unoptimized Linux and an owned Slurm allocation")
    if args.sources is None:
        source = collect_sources(ROOT, args.model, config)
    else:
        source = json.loads(args.sources.read_text())
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"])
        if (
            dirty
            or source["project_commit"] != head
            or source["project_dirty"] is not False
            or source["python_version"] != "3.12.16"
            or source["packages"]["vllm"] != "0.32.0"
            or source["upstream_commit"] != config["upstream_commit"]
            or source["model"]["revision"] != config["model_revision"]
            or source["model"]["tokenizer_revision"] != config["tokenizer_revision"]
            or Path(source["model"]["directory"]).resolve() != args.model.resolve()
        ):
            raise RuntimeError("M5 precomputed source document does not bind the replay runtime")
    return {**source, "job_id": job_id}


def _entry(plan, planned_ns):
    return {
        "request_id": plan["request_id"],
        "planned_ns": planned_ns,
        "submit_ns": None,
        "first_token_ns": None,
        "last_token_ns": None,
        "end_ns": None,
        "status": "unfinished",
        "token_ids": [],
        "cached_tokens": None,
        "chunk_tokens": [],
    }


async def _consume(llm, plan, row, params, deadline_ns, *, native_stats_type=None):
    # Stamp on the executing client task, directly before invoking generate.
    row["submit_ns"] = time.monotonic_ns()
    remaining = (deadline_ns - row["submit_ns"]) / 1_000_000_000
    if remaining <= 0:
        row.update(submit_ns=None, status="timeout", end_ns=time.monotonic_ns())
        return
    finished = False
    try:
        async with asyncio.timeout(remaining):
            async for output in llm.generate(
                {"prompt_token_ids": plan["prompt_token_ids"], "cache_salt": plan["cache_salt"]},
                params,
                request_id=plan["request_id"],
            ):
                now = time.monotonic_ns()
                chunk = list(output.outputs[0].token_ids) if output.outputs else []
                if chunk:
                    first = row["first_token_ns"] is None
                    if first:
                        row["first_token_ns"] = now
                    row["last_token_ns"] = now
                    row["chunk_tokens"].append({"at_ns": now, "token_ids": chunk})
                    row["token_ids"].extend(chunk)
                    if native_stats_type is not None:
                        last = output.finished or len(row["token_ids"]) >= params.max_tokens
                        if first or last:
                            snapshot = snapshot_native_metrics(
                                getattr(output, "metrics", None), native_stats_type, now
                            )
                            if first:
                                row["native_metrics"]["first"] = snapshot
                            if last:
                                row["native_metrics"]["last"] = snapshot
                if output.num_cached_tokens is not None:
                    row["cached_tokens"] = output.num_cached_tokens
                if output.finished:
                    finished = True
            row["status"] = "completed" if finished else "failed"
    except TimeoutError:
        row["status"] = "timeout"
    except asyncio.CancelledError:
        row["status"] = "cancelled"
        raise
    except Exception as error:
        row.update(status="failed", error=type(error).__name__)
    finally:
        row["end_ns"] = time.monotonic_ns()
        if row["status"] == "completed" and row["end_ns"] > deadline_ns:
            row["status"] = "timeout"


async def _sleep_until(deadline_ns):
    while (remaining := deadline_ns - time.monotonic_ns()) > 0:
        await asyncio.sleep(remaining / 1_000_000_000)


async def _open_loop(plans, rows, launch):
    """Stamp and spawn from absolute deadlines, independent of prior request results."""
    for plan, row in zip(plans, rows, strict=True):
        await _sleep_until(row["planned_ns"])
        launch(plan, row)


def _drained(row):
    fields = (
        "requests",
        "waiting",
        "running",
        "holding",
        "deferred",
        "finished_req_ids",
        "reset_preempted_req_ids",
        "inflight_prefills",
        "deferred_frees",
        "request_blocks",
        "non_null_ref_sum",
        "pending_batches",
        "inflight_tokens",
        "stale_tokens",
        "output_placeholders",
        "guard_records",
        "guard_fault_code",
        "runner_request_ids",
        "runner_pending",
        "m5_pending",
        "observation_errors",
    )
    return not any(row.get(key) for key in fields) and (
        "free_count" not in row
        or len(
            {
                row[key]
                for key in (
                    "free_count",
                    "free_unique_count",
                    "free_traversed_count",
                    "total_non_null_blocks",
                )
            }
        )
        == 1
    )


async def _d2_replay_wave(llm, plans, rows, params, cutoff_ns, deadline_ns):
    tasks = []

    def launch(plan, row):
        deadline = min(cutoff_ns, row["planned_ns"] + deadline_ns)
        tasks.append(asyncio.create_task(_consume(llm, plan, row, params, deadline)))

    try:
        await _open_loop(plans, rows, launch)
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _d2_condition(
    args,
    workload,
    llm,
    params,
    drain,
    thermal,
    conditioning,
    deadline_ns,
    *,
    pilot_temperature_range=None,
):
    from m5_d2_thermal import NS, first_plateau, read_samples, target_from_plateau

    prospective = thermal["schema_version"] == 2
    started_ns = time.monotonic_ns()
    thermal["conditioning_start_ns"] = started_ns
    cap_s = conditioning["pilot_cap_s" if args.d2_role == "pilot" else "per_measured_engine_cap_s"]
    cutoff_ns = started_ns + cap_s * NS
    if prospective:
        thermal["conditioning_cutoff_ns"] = cutoff_ns
    timeout_s = (cutoff_ns - time.monotonic_ns()) / NS if prospective else cap_s
    selection = (
        {"pilot_temperature_range": pilot_temperature_range}
        if args.d2_role == "pilot" and pilot_temperature_range is not None
        else {}
    )
    try:
        async with asyncio.timeout(max(0, timeout_s)):
            while time.monotonic_ns() < cutoff_ns:
                wave = len(thermal["waves"]) + 1
                start_ns = time.monotonic_ns() + 20_000_000
                if prospective and start_ns >= cutoff_ns:
                    break
                plans = [
                    {**plan, "request_id": f"d2-preheat-{wave}-{plan['request_id']}"}
                    for plan in workload["requests"]
                ]
                rows = [_entry(plan, start_ns + plan["offset_ns"]) for plan in plans]
                record = {
                    "wave": wave,
                    "started_ns": start_ns,
                    "requests": rows,
                    "drained_at_ns": None,
                    "reset_after_ns": None,
                    "sample_observed_through_ns": None,
                }
                if prospective:
                    record["plateau_decision_ns"] = None
                thermal["waves"].append(record)
                await _d2_replay_wave(llm, plans, rows, params, cutoff_ns, deadline_ns)
                if not all(
                    row["status"] == "completed"
                    and len(row["token_ids"]) == workload["sampling"]["max_tokens"]
                    for row in rows
                ):
                    thermal["reason"] = "preheat wave did not complete"
                    return
                drained = await drain(f"d2-preheat-{wave}")
                record["drained_at_ns"] = drained["at_ns"]
                if not await llm.reset_prefix_cache(
                    reset_running_requests=False, reset_connector=False
                ):
                    thermal["reason"] = "preheat APC reset failed"
                    return
                record["reset_after_ns"] = time.monotonic_ns()
                if record["reset_after_ns"] > cutoff_ns:
                    break
                telemetry = read_samples(args.d2_samples, record["reset_after_ns"])
                if telemetry:
                    record["sample_observed_through_ns"] = telemetry[-1]["monotonic_ns"]
                if (
                    thermal["target_gpu_uuid"] is not None
                    and telemetry
                    and telemetry[-1]["gpu_uuid"] != thermal["target_gpu_uuid"]
                ):
                    thermal["reason"] = "conditioning GPU differs from pilot"
                    return
                plateau = first_plateau(
                    telemetry, started_ns, conditioning, thermal["target"], **selection
                )
                if prospective:
                    record["plateau_decision_ns"] = time.monotonic_ns()
                    if record["plateau_decision_ns"] > cutoff_ns:
                        break
                if plateau is not None:
                    thermal["plateau"] = plateau
                    thermal["reason"] = None
                    if args.d2_role == "pilot":
                        thermal["target"] = target_from_plateau(plateau)
                        thermal["target_gpu_uuid"] = plateau["gpu_uuid"]
                        thermal["status"] = "target-frozen"
                    return
    except TimeoutError:
        pass
    except (OSError, KeyError, TypeError, ValueError) as error:
        thermal["reason"] = f"conditioning telemetry invalid: {type(error).__name__}"
        return
    finally:
        thermal["conditioning_end_ns"] = time.monotonic_ns()
    thermal["reason"] = "conditioning deadline without an admissible plateau"


def _check_d5_profile(profile, expected):
    require(
        all(
            type(profile.get(key)) is int and profile[key] == value
            for key, value in expected.items()
        ),
        "D5 active engine differs from frozen admission cap, runner slots, token budget or KV blocks",
    )


async def _run(
    args,
    config,
    workload,
    client,
    *,
    native_stats_type=None,
    preheat_workload=None,
    d5_expected_profile=None,
    pilot_temperature_range=None,
    trace_plan=None,
    d10_placement_mode=None,
):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_prefix_scheduler.m5_observer import clock_identity

    options = json.loads((ROOT / config["baseline_config"]).read_text())
    if native_stats_type is not None:
        if options.get("disable_log_stats") is not False:
            raise RuntimeError("D3 requires the existing native log_stats path")
        client["native_metrics_clock"] = monotonic_clock_info()
    options.update(
        model=str(args.model.resolve()),
        tokenizer=str(args.model.resolve()),
        block_size=config["block_size"],
        attention_config={"backend": config["attention_backend"]},
        worker_extension_cls="vllm_prefix_scheduler.m5_observer.M5WorkerExtension",
    )
    stepped = trace_plan is not None and _trace_capture(trace_plan, args.pair_id) == "step"
    trace_armed = False
    if stepped:
        trace_role = trace_plan["role"]
        trace_key = trace_role.lower()
        options["worker_extension_cls"] = (
            f"vllm_prefix_scheduler.m5_{trace_key}_observer.{trace_role}WorkerExtension"
        )
    if args.tier == "endpoint":
        if args.mode != "native":
            options["scheduler_cls"] = "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
    else:
        observer = "M5NativeScheduler" if args.mode == "native" else "M5PrefixScheduler"
        options["scheduler_cls"] = f"vllm_prefix_scheduler.m5_observer.{observer}"
    if args.mode != "native":
        options["additional_config"] = {
            "prefix_scheduler": {
                "mode": args.mode,
                "window": config["window"],
                "max_wait_ms": config["max_wait_ms"],
            }
        }
    if args.mechanism is not None:
        diag = config["scenarios"]["mechanism"]
        options.update(
            num_gpu_blocks_override=diag[f"{args.mechanism}_blocks"],
            max_model_len=diag["max_model_len"],
        )
    if d5_expected_profile is not None:
        options.update(d5_expected_profile)
        options["num_gpu_blocks_override"] = options.pop("num_gpu_blocks")

    llm = None
    tasks = []
    params = SamplingParams(output_kind=RequestOutputKind.DELTA, **workload["sampling"])
    phases = client["phases"]
    deadline_ns = client["deadline_ns"]
    client["clock"] = clock_identity()
    d2_thermal = None
    host_placement = None
    if args.d2_role is not None:
        from m5_d2_thermal import target_from_plateau

        plan = json.loads(args.d2_plan.read_text())
        prospective = plan["role"] in (D2R_ROLE, D2R110_ROLE, D3_ROLE, D8_ROLE)
        conditioning = plan["conditioning"]
        plan_sha = sha256(
            getattr(args, "d10_plan", None)
            or getattr(args, "d9_plan", None)
            or getattr(args, "d7_plan", None)
            or getattr(args, "d6_plan", None)
            or getattr(args, "d5r_plan", None)
            or args.d5_plan
            or args.d4_plan
            or args.d2_plan
        )
        target, pilot_gpu = None, None
        if args.d2_role == "measure":
            pilot = json.loads(args.d2_target.read_text())
            if (
                pilot["role"] != "pilot"
                or pilot["status"] != "target-frozen"
                or (prospective and pilot["schema_version"] != 2)
                or pilot["plan_sha256"] != plan_sha
                or pilot["target"] != target_from_plateau(pilot["plateau"])
            ):
                raise RuntimeError("D2 pilot target does not bind this plan")
            target = pilot["target"]
            if pilot_temperature_range is not None:
                require(
                    pilot_temperature_range[0]
                    <= target["temperature_c"]
                    <= pilot_temperature_range[1],
                    "D5-R pilot target temperature differs from the frozen band",
                )
            pilot_gpu = pilot["plateau"]["gpu_uuid"]
        d2_thermal = {
            "schema_version": 2 if prospective else 1,
            "role": args.d2_role,
            "status": "invalid",
            "reason": "D2 conditioning did not finish",
            "plan_sha256": plan_sha,
            "conditioning_start_ns": None,
            "conditioning_end_ns": None,
            "target": target,
            "target_gpu_uuid": pilot_gpu,
            "plateau": None,
            "waves": [],
            "preheat_boundaries": [],
            "pre_t0": None,
        }
        if prospective:
            d2_thermal["conditioning_cutoff_ns"] = None

    async def rpc(name, **kwargs):
        return _single_worker(await llm.collective_rpc(name, timeout=15, kwargs=kwargs))

    async def read_clock_samples():
        samples = []
        for _ in range(5):
            send_ns = time.monotonic_ns()
            clock = await rpc("m5_clock")
            recv_ns = time.monotonic_ns()
            samples.append(
                {"send_ns": send_ns, "engine_ns": clock["engine_ns"], "recv_ns": recv_ns}
            )
        return samples

    async def drain(label, stage=None):
        expires = time.monotonic_ns() + deadline_ns
        while True:
            row = await rpc("m5_boundary", label=f"{label}-poll")
            if _drained(row) and llm.get_num_unfinished_requests() == 0:
                return await rpc("m5_boundary", label=label, stage=stage)
            if time.monotonic_ns() >= expires:
                raise TimeoutError(f"native engine failed to drain at {label}")
            await asyncio.sleep(0.01)

    def launch(plan, row):
        tasks.append(
            asyncio.create_task(
                _consume(
                    llm,
                    plan,
                    row,
                    params,
                    row["planned_ns"] + deadline_ns,
                    native_stats_type=native_stats_type,
                )
            )
        )

    try:
        phases["load_start_ns"] = time.monotonic_ns()
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        phases["load_end_ns"] = time.monotonic_ns()
        limit = config.get("observation_buffer_rows", 250_000)
        armed = await rpc(
            "m5_arm", tier=args.tier, limit=limit, schema_version=client["schema_version"]
        )
        if d5_expected_profile is not None:
            _check_d5_profile(armed["profile"], d5_expected_profile)
        samples = await read_clock_samples()
        shortest = min(samples, key=lambda row: row["recv_ns"] - row["send_ns"])
        send_ns, engine_ns, recv_ns = (shortest[key] for key in ("send_ns", "engine_ns", "recv_ns"))
        client["clock_check"] = shortest
        client["clock_samples"] = samples
        same_domain = ("clock", "host", "boot_id", "time_namespace")
        if (
            not all(client["clock"][key] == armed["clock"][key] for key in same_domain)
            or not send_ns <= engine_ns <= recv_ns
            or recv_ns - send_ns > config["gates"]["max_clock_rtt_ns"]
        ):
            raise RuntimeError("M5 could not verify a shared CLOCK_MONOTONIC domain")
        if native_stats_type is not None:
            if (
                llm.log_stats is not True
                or llm.output_processor.log_stats is not True
                or client["clock"]["pid"] == armed["clock"]["pid"]
            ):
                raise RuntimeError("D3 native stats or distinct EngineCore process missing")
            verify_clock_brackets(samples, config["gates"]["max_clock_rtt_ns"])
        if d10_placement_mode is not None:
            from vllm_prefix_scheduler.m5_d10_placement import HostPlacement

            host_placement = HostPlacement(armed["clock"]["pid"], d10_placement_mode)
            client["d10_placement"] = host_placement.evidence
            host_placement.setup()
        if armed["profile"]["block_size"] != workload["block_size"]:
            raise RuntimeError("M5 workload and active cache block sizes differ")
        if (
            args.mechanism is not None
            and armed["profile"]["num_gpu_blocks"] != options["num_gpu_blocks_override"]
        ):
            raise RuntimeError("M5 mechanism override not reflected in actual block count")

        if d2_thermal is not None:
            conditioning_workload = workload if preheat_workload is None else preheat_workload
            thermal_options = (
                {"pilot_temperature_range": pilot_temperature_range}
                if args.d2_role == "pilot" and pilot_temperature_range is not None
                else {}
            )
            await _d2_condition(
                args,
                conditioning_workload,
                llm,
                params,
                drain,
                d2_thermal,
                conditioning,
                deadline_ns,
                **thermal_options,
            )
            if args.d2_role == "pilot" or d2_thermal["plateau"] is None:
                return

        phases["warmup_start_ns"] = time.monotonic_ns()
        primer = {**workload["requests"][0], "request_id": "runtime-primer"}
        row = _entry(primer, time.monotonic_ns())
        await _consume(llm, primer, row, params, row["planned_ns"] + deadline_ns)
        client["primer"] = row
        if row["status"] != "completed":
            raise RuntimeError("M5 native runtime primer failed")
        await drain("pre-reset")
        if not await llm.reset_prefix_cache(reset_running_requests=False, reset_connector=False):
            raise RuntimeError("M5 idle prefix cache reset failed")
        cleared = await drain("cache-reset")
        if args.tier != "endpoint" and cleared["cached_block_count"] != 0:
            raise RuntimeError("M5 cache reset left native APC entries")
        for plan in workload["warmup"]:
            row = _entry(plan, time.monotonic_ns())
            client["warmup"].append(row)
            await _consume(llm, plan, row, params, row["planned_ns"] + deadline_ns)
            if row["status"] != "completed":
                raise RuntimeError("M5 fixed cache priming failed")
        await drain("measure-start", stage="measure")
        phases["warmup_end_ns"] = time.monotonic_ns()
        if host_placement is not None:
            host_placement.observe("before_measure")

        if args.mechanism is not None:
            await llm.pause_generation(mode="keep", clear_cache=False)
        if d2_thermal is None:
            phases["measure_start_ns"] = time.monotonic_ns() + 20_000_000
        else:
            from m5_d2_thermal import pre_t0_admitted, read_samples

            decision_ns = time.monotonic_ns()
            try:
                samples = read_samples(args.d2_samples, decision_ns)
            except (OSError, KeyError, TypeError, ValueError) as error:
                d2_thermal["reason"] = f"pre-t0 telemetry invalid: {type(error).__name__}"
                return
            sample = samples[-1] if samples else None
            lead = 20_000_000 if trace_plan is None else trace_plan["trace"]["arm_lead_ns"]
            measure_start_ns = time.monotonic_ns() + lead
            d2_thermal["pre_t0"] = {
                "decision_ns": decision_ns,
                "measure_start_ns": measure_start_ns,
                "sample": sample,
            }
            if (
                sample is None
                or sample["gpu_uuid"] != d2_thermal["target_gpu_uuid"]
                or not pre_t0_admitted(
                    sample,
                    measure_start_ns,
                    d2_thermal["target"],
                    conditioning["per_measured_engine"],
                )
            ):
                d2_thermal["reason"] = (
                    "pre-t0 temperature, freshness or GPU identity outside limits"
                )
                return
            if stepped:
                trace_armed = True
                arm_arguments = _trace_arm_arguments(trace_plan, measure_start_ns)
                await rpc(f"m5_{trace_key}_arm", **arm_arguments)
                client[f"{trace_key}_arm_return_ns"] = time.monotonic_ns()
                if client[f"{trace_key}_arm_return_ns"] >= measure_start_ns:
                    d2_thermal["reason"] = f"{trace_role} trace arm reached the planned start"
                    return
            d2_thermal["status"] = "admitted"
            d2_thermal["reason"] = None
            phases["measure_start_ns"] = measure_start_ns
        client["requests"] = [
            _entry(plan, phases["measure_start_ns"] + plan["offset_ns"])
            for plan in workload["requests"]
        ]
        if native_stats_type is not None:
            for row in client["requests"]:
                row["native_metrics"] = {"first": None, "last": None}
        phases["arrival_end_ns"] = client["requests"][-1]["planned_ns"]
        client["drain_deadline_ns"] = phases["arrival_end_ns"] + deadline_ns
        await _open_loop(workload["requests"], client["requests"], launch)
        if args.mechanism is not None:
            expires = time.monotonic_ns() + 2_000_000_000
            while True:
                probes = await rpc("m5_probe", request_ids=["B", "C"])
                if len(probes) == 2:
                    break
                if time.monotonic_ns() >= expires:
                    raise RuntimeError("mechanism did not enqueue both requests while paused")
                await asyncio.sleep(0.002)
            await llm.resume_generation()
        await asyncio.gather(*tasks)
        phases["drain_end_ns"] = time.monotonic_ns()
        await drain("measure-drained", stage="done")
        if host_placement is not None:
            host_placement.observe("after_measure")
        if stepped and trace_role == "D9":
            horizon_ns = phases["measure_start_ns"] + trace_plan["trace"]["windows_ns"][-1][1]
            await _wait_until(horizon_ns)
        if native_stats_type is not None:
            native_clock = client["native_metrics_clock"]
            native_clock["post_clock_samples"] = await read_clock_samples()
            native_clock["post_clock_check"] = verify_clock_brackets(
                native_clock["post_clock_samples"],
                config["gates"]["max_clock_rtt_ns"],
                after_ns=phases["drain_end_ns"],
            )
    finally:
        if llm is not None:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        if stepped:
            trace = {"schema_version": 1, "status": "not-started"}
            if trace_armed:
                try:
                    trace = await rpc(f"m5_{trace_key}_export")
                except Exception as error:
                    trace = {
                        "schema_version": 1,
                        "status": "incomplete",
                        "errors": [f"trace export failed: {type(error).__name__}"],
                    }
            trace_path = args.output / f"{trace_key}-step.json"
            if trace_role == "D9":
                from vllm_prefix_scheduler.m5_d9_observer import write_d9_trace

                write_d9_trace(trace_path, trace)
            else:
                _dump(trace_path, trace)
        if llm is not None:
            try:
                engine = await rpc("m5_export")
                if d2_thermal is not None:
                    retained = []
                    for boundary in engine["boundaries"]:
                        if boundary["label"].startswith("d2-preheat-"):
                            d2_thermal["preheat_boundaries"].append(boundary)
                        else:
                            retained.append(boundary)
                    engine["boundaries"] = retained
                _dump(args.output / "engine.json", engine)
                if engine["errors"]:
                    raise RuntimeError("M5 failed to collect trustworthy native evidence")
            finally:
                try:
                    llm.shutdown()
                finally:
                    try:
                        if host_placement is not None:
                            host_placement.restore_client()
                    finally:
                        if d2_thermal is not None:
                            _dump(args.output / "d2-thermal.json", d2_thermal)
        elif d2_thermal is not None:
            _dump(args.output / "d2-thermal.json", d2_thermal)


def _d4_replay_inputs(args, workload):
    proposal, inherited = d4_plan(args.d4_plan, args.d2_plan)
    target = proposal["target"]
    preheat_rate = target["preheat_rate_rps"]
    rates = dict(
        zip(
            inherited["target"]["fresh_engines"],
            [preheat_rate, *target["measure_order_rps"]],
            strict=True,
        )
    )
    require(
        args.pair_id in rates and (args.d2_role == "pilot") == (args.pair_id == "thermal-pilot"),
        "D4 leaf identity differs from the frozen order",
    )
    measure_rate = rates[args.pair_id]
    require(
        sha256(args.input) == target["arrival_tables"][str(measure_rate)]["sha256"]
        and workload["scenario"] == target["scenario"]
        and workload["seed"] == target["seed"]
        and workload["rate_rps"] == measure_rate
        and len(workload["requests"]) == target["count_per_engine"],
        "D4 measured input differs from the frozen leaf",
    )
    require(
        sha256(args.d4_preheat_input) == target["arrival_tables"][str(preheat_rate)]["sha256"],
        "D4 preheat input differs from the frozen 8 rps table",
    )
    preheat = read_json(args.d4_preheat_input)
    require(
        preheat["rate_rps"] == preheat_rate
        and preheat["scenario"] == workload["scenario"]
        and preheat["seed"] == workload["seed"]
        and preheat["sampling"] == workload["sampling"]
        and len(preheat["requests"]) == len(workload["requests"]),
        "D4 preheat workload differs from the frozen input contract",
    )
    return inherited["native_timing"], preheat


def _d5_replay_inputs(args, workload):
    temperature_range = None
    trace_plan = None
    if args.d7_plan is not None:
        trace_plan, retry, plan, inherited = d7_plan(
            args.d7_plan, args.d6_plan, args.d5r_plan, args.d5_plan, args.d2_plan
        )
        temperature_range = retry["pilot_target_median_c"]
        leaves = D7_RUNS
    elif args.d6_plan is not None:
        trace_plan, retry, plan, inherited = d6_plan(
            args.d6_plan, args.d5r_plan, args.d5_plan, args.d2_plan
        )
        temperature_range = retry["pilot_target_median_c"]
        leaves = D6_RUNS
    elif args.d5r_plan is None:
        plan, inherited = d5_plan(args.d5_plan, args.d2_plan)
    else:
        retry, plan, inherited = d5r_plan(args.d5r_plan, args.d5_plan, args.d2_plan)
        temperature_range = retry["pilot_target_median_c"]
    target = inherited["target"]
    require(
        args.pair_id in (leaves if trace_plan is not None else target["fresh_engines"])
        and (args.d2_role == "pilot") == (args.pair_id == "thermal-pilot"),
        "D5 leaf identity differs from the frozen order",
    )
    require(
        sha256(args.input) == target["input_sha256"]
        and workload["scenario"] == target["scenario"]
        and workload["seed"] == target["seed"]
        and workload["rate_rps"] == target["arrival_rate_rps"]
        and len(workload["requests"]) == target["count_per_engine"],
        "D5 input differs from the frozen original 8 rps table",
    )
    profile = d5_profile(plan, "E1" if trace_plan is not None else args.pair_id)
    return inherited["native_timing"], profile, temperature_range, trace_plan


def _trace_arm_arguments(plan, t0_ns):
    limits = plan["trace"]
    if plan["role"] == "D9":
        return {
            "t0_ns": t0_ns,
            "windows_ns": [[t0_ns + start, t0_ns + end] for start, end in limits["windows_ns"]],
        }
    start_ns = t0_ns + limits.get("start_offset_ns", 0)
    arguments = {"start_ns": start_ns, "end_ns": start_ns + limits["window_ns"]}
    if plan["role"] == "D7":
        arguments["t0_ns"] = t0_ns
    return arguments


def _trace_capture(plan, pair_id):
    if plan["role"] == "D9":
        runs = plan["runs"]
    else:
        runs = D7_RUNS if plan["role"] == "D7" else D6_RUNS
    return plan["capture"][runs.index(pair_id)]


async def _wait_until(horizon_ns):
    while (remaining_ns := horizon_ns - time.monotonic_ns()) > 0:
        await asyncio.sleep(remaining_ns / 1_000_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "input", "model", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--tier", choices=TIER, required=True)
    parser.add_argument("--phase", choices=("formal", "calibration", "diagnostic"), required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--mechanism", choices=("pressure", "ample"))
    parser.add_argument("--d2-role", choices=("pilot", "measure"))
    for field in (
        "d2-plan",
        "d2-samples",
        "d2-target",
        "d4-plan",
        "d4-preheat-input",
        "d5-plan",
        "d5r-plan",
        "d6-plan",
        "d7-plan",
        "d8-inherited-plan",
        "d9-plan",
        "d10-plan",
    ):
        parser.add_argument(f"--{field}", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workload = json.loads(args.input.read_text())
    if (
        workload["task_id"] != config["task_id"]
        or workload["tokenizer_revision"] != config["tokenizer_revision"]
        or (workload["scenario"] == "mechanism") != (args.mechanism is not None)
        or workload["mechanism"] != args.mechanism
        or (args.mechanism is not None and (args.tier != "profile" or args.phase == "formal"))
        or (args.phase == "formal" and args.tier != "light")
    ):
        parser.error("M5 input, phase and mode do not match the frozen design")
    d2_files = (args.d2_plan, args.d2_samples, args.d2_target)
    if (args.d4_plan is None) != (args.d4_preheat_input is None) or (
        args.d4_plan is not None and args.d2_role is None
    ):
        parser.error("D4 requires its plan and 8 rps preheat input with a thermal replay role")
    if args.d5_plan is not None and (args.d2_role is None or args.d4_plan is not None):
        parser.error("D5 requires a thermal replay role and its own plan")
    if args.d5r_plan is not None and args.d5_plan is None:
        parser.error("D5-R requires its inherited D5 plan")
    if args.d6_plan is not None and args.d5r_plan is None:
        parser.error("D6 requires its inherited D5-R plan")
    if args.d7_plan is not None and args.d6_plan is None:
        parser.error("D7 requires its inherited D6 plan")
    if args.d8_inherited_plan is not None and (
        args.d2_role is None or args.d4_plan is not None or args.d5_plan is not None
    ):
        parser.error("D8 inherited plan requires the original endpoint replay")
    if args.d9_plan is not None and args.d8_inherited_plan is None:
        parser.error("D9 requires the original D8 and inherited D2-R110 plans")
    if args.d10_plan is not None and (args.d8_inherited_plan is None or args.d9_plan is not None):
        parser.error("D10 requires the original D8 and inherited D2-R110 plans")
    native_timing = None
    preheat_workload = None
    d5_expected_profile = None
    thermal_options = {}
    trace_plan = None
    if args.d2_role is None:
        if any(path is not None for path in d2_files):
            parser.error("D2 evidence paths require --d2-role")
    elif (
        args.d2_plan is None
        or args.d2_samples is None
        or (args.d2_target is not None) != (args.d2_role == "measure")
        or (args.phase, args.mode, args.tier, args.mechanism)
        != ("calibration", "guarded", "endpoint", None)
    ):
        parser.error("D2 requires the guarded endpoint calibration and thermal inputs")
    elif args.d5_plan is not None:
        try:
            native_timing, d5_expected_profile, temperature_range, trace_plan = _d5_replay_inputs(
                args, workload
            )
            if temperature_range is not None:
                thermal_options["pilot_temperature_range"] = temperature_range
            if trace_plan is not None:
                thermal_options["trace_plan"] = trace_plan
        except (KeyError, OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    elif args.d4_plan is not None:
        try:
            native_timing, preheat_workload = _d4_replay_inputs(args, workload)
        except (KeyError, OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    else:
        plan = json.loads(args.d2_plan.read_text())
        if (plan["role"] == D8_ROLE) != (args.d8_inherited_plan is not None):
            parser.error("D8 endpoint replay requires its inherited D2-R110 plan")
        if plan["role"] == D2R_ROLE:
            d2r_plan(plan)
        elif plan["role"] == D2R110_ROLE:
            d2r110_plan(plan)
        elif plan["role"] == D3_ROLE:
            native_timing = d3_plan(plan)["native_timing"]
        elif plan["role"] == D8_ROLE:
            try:
                if args.d10_plan is not None:
                    from vllm_prefix_scheduler.m5_d10 import d10_plan

                    placement_plan = d10_plan(args.d10_plan, args.d2_plan, args.d8_inherited_plan)
                    placements = dict(
                        zip(
                            placement_plan["runs"],
                            ("inherited", *placement_plan["placement_order"]),
                            strict=True,
                        )
                    )
                    thermal_options["d10_placement_mode"] = placements[args.pair_id]
                    if args.d2_role == "measure":
                        native_timing = placement_plan["native_timing"]
                elif args.d9_plan is None:
                    d8_plan(args.d2_plan, args.d8_inherited_plan)
                else:
                    from vllm_prefix_scheduler.m5_d9 import d9_plan

                    trace_plan = d9_plan(args.d9_plan, args.d2_plan, args.d8_inherited_plan)
                    thermal_options["trace_plan"] = trace_plan
            except (KeyError, OSError, TypeError, ValueError) as error:
                parser.error(str(error))
        planned = plan["target"]
        expected_id = (
            args.pair_id == "thermal-pilot"
            if args.d2_role == "pilot"
            else args.pair_id in planned["fresh_engines"][1:]
        )
        if (
            not expected_id
            or sha256(args.input) != planned["input_sha256"]
            or workload["seed"] != planned["seed"]
            or len(workload["requests"]) != planned["count_per_engine"]
            or workload["rate_rps"] != planned["arrival_rate_rps"]
        ):
            parser.error("D2 input or run ID differs from the reviewed plan")
        if trace_plan is not None and _trace_capture(trace_plan, args.pair_id) == "step":
            native_timing = trace_plan["native_timing"]
    os.environ.update(config["environment"])
    os.environ["M5_WINDOW"] = str(config["window"])
    args.output.mkdir(parents=True, exist_ok=False)
    client = {
        "schema_version": config.get("measurement_schema_version", 1),
        "task_id": config["task_id"],
        "pair_id": args.pair_id,
        "scenario": workload["scenario"],
        "seed": workload["seed"],
        "mode": args.mode,
        "tier": args.tier,
        "phase": args.phase,
        "input_sha256": sha256(args.input),
        "config_sha256": sha256(args.config),
        "deadline_ns": config["gates"]["deadline_ns"],
        "phases": {},
        "requests": (
            [_entry(plan, None) for plan in workload["requests"]] if args.d2_role is None else []
        ),
        "warmup": [],
    }
    if native_timing is not None:
        client["native_metrics_schema_version"] = native_timing["schema_version"]
    try:
        source = _binding(args, config)
        if trace_plan is not None:
            d6_sources(trace_plan, source)
            role = trace_plan["role"]
            client[f"{role.lower()}_capture"] = _trace_capture(trace_plan, args.pair_id)
        client["execution_commit"] = source["project_commit"]
        client["job_id"] = source["job_id"]
        _dump(args.output / "sources.json", source)
        native_stats_type = None
        if native_timing is not None:
            native_stats_type = bind_loaded_sources(source, native_timing["source_sha256"])
        asyncio.run(
            _run(
                args,
                config,
                workload,
                client,
                native_stats_type=native_stats_type,
                preheat_workload=preheat_workload,
                d5_expected_profile=d5_expected_profile,
                **thermal_options,
            )
        )
    finally:
        _dump(args.output / "client.json", client)


if __name__ == "__main__":
    main()
