"""Verify D5 runtime admission caps before any thermal work."""

import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m5_observer
from vllm_prefix_scheduler.m5_campaign import digest, read_json
from vllm_prefix_scheduler.m5_d5 import d5_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("leaf", "mismatch"),
    (
        ("thermal-pilot", None),
        ("E2", None),
        ("E2", "max_num_active_seqs"),
        ("thermal-pilot", "num_gpu_blocks"),
        ("E4", "max_num_seqs"),
        ("E3", "max_num_batched_tokens"),
    ),
)
def test_d5_runtime_profile_before_conditioning(tmp_path, monkeypatch, leaf, mismatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    replay = importlib.import_module("replay_m5")
    config = read_json(ROOT / "configs/m5-performance.json")
    d3_path = ROOT / "configs/m5-d3.json"
    d5_path = ROOT / "configs/m5-d5.json"
    expected = d5_profile(read_json(d5_path), leaf)
    actual = {**expected, "block_size": config["block_size"]}
    if mismatch:
        actual[mismatch] += 1
    clock = dict(clock="CLOCK_MONOTONIC", host="node", boot_id="boot", time_namespace="ns", pid=1)
    monkeypatch.setattr(m5_observer, "clock_identity", lambda: clock)
    arguments, thermal_calls = [], []

    def engine_args(**options):
        arguments.append(options)
        return options

    async def rpc(name, **kwargs):
        if name == "m5_arm":
            return [{"profile": actual, "clock": {**clock, "pid": 2}}]
        if name == "m5_clock":
            return [{"engine_ns": time.monotonic_ns()}]
        assert name == "m5_export"
        return [{"profile": actual, "errors": [], "boundaries": []}]

    llm = SimpleNamespace(collective_rpc=rpc, shutdown=lambda: None)
    fake_modules = {
        "vllm": SimpleNamespace(SamplingParams=dict),
        "vllm.engine.arg_utils": SimpleNamespace(AsyncEngineArgs=engine_args),
        "vllm.sampling_params": SimpleNamespace(RequestOutputKind=SimpleNamespace(DELTA="delta")),
        "vllm.v1.engine.async_llm": SimpleNamespace(
            AsyncLLM=SimpleNamespace(from_engine_args=lambda options: llm)
        ),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    async def condition(args, workload, llm, params, drain, thermal, conditioning, deadline_ns):
        thermal_calls.append((workload, thermal["plan_sha256"], conditioning))

    monkeypatch.setattr(replay, "_d2_condition", condition)
    pilot_target = tmp_path / "target.json"
    target = dict(temperature_c=65, sm_clock_mhz=1875, memory_clock_mhz=9501)
    pilot_target.write_text(
        json.dumps(
            {
                "role": "pilot",
                "status": "target-frozen",
                "schema_version": 2,
                "plan_sha256": digest(d5_path),
                "target": target,
                "plateau": {**target, "gpu_uuid": "gpu"},
            }
        )
    )
    args = SimpleNamespace(
        model=tmp_path,
        mode="guarded",
        tier="endpoint",
        mechanism=None,
        d2_role="pilot" if leaf == "thermal-pilot" else "measure",
        d2_plan=d3_path,
        d2_target=pilot_target,
        d4_plan=None,
        d5_plan=d5_path,
        output=tmp_path,
    )
    workload = {"rate_rps": 8, "sampling": {"max_tokens": 16}, "block_size": config["block_size"]}
    client = {"schema_version": 2, "phases": {}, "deadline_ns": config["gates"]["deadline_ns"]}
    coroutine = replay._run(args, config, workload, client, d5_expected_profile=expected)
    if mismatch:
        with pytest.raises(ValueError, match="D5 active engine differs"):
            asyncio.run(coroutine)
        assert not thermal_calls
    else:
        asyncio.run(coroutine)
        assert thermal_calls == [(workload, digest(d5_path), read_json(d3_path)["conditioning"])]
    assert {
        key: arguments[0][key]
        for key in (
            "max_num_seqs",
            "max_num_active_seqs",
            "max_num_batched_tokens",
            "num_gpu_blocks_override",
        )
    } == {
        "max_num_seqs": 5,
        "max_num_active_seqs": 5 if leaf in {"E2", "E3"} else 4,
        "max_num_batched_tokens": 2048,
        "num_gpu_blocks_override": 5553,
    }
    assert read_json(tmp_path / "engine.json")["profile"] == actual
