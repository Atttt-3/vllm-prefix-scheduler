"""D3 measurement producer checks without a model or GPU."""

import asyncio
import importlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m5_d3
from vllm_prefix_scheduler.m5_campaign import digest


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("replay_m5")


class Stats:
    queued_ts = 1.0
    scheduled_ts = 1.1
    first_token_ts = 1.2
    last_token_ts = 1.3
    num_generation_tokens = 2
    num_preemptions = 0
    arrival_time = 1700000000.0
    first_token_latency = 0.2


def test_native_scalar_copy_does_not_retain_wall_clock_or_mutable_stats():
    stats = Stats()
    first = m5_d3.snapshot_native_metrics(stats, Stats, 10)
    stats.last_token_ts = 1.5
    stats.num_generation_tokens = 4
    stats.first_token_ts = float("nan")
    last = m5_d3.snapshot_native_metrics(stats, Stats, 20)
    assert first == {
        "client_received_ns": 10,
        "queued_ts": 1.0,
        "scheduled_ts": 1.1,
        "first_token_ts": 1.2,
        "last_token_ts": 1.3,
        "native_generation_tokens": 2,
        "native_preemptions": 0,
    }
    assert last["last_token_ts"] == 1.5
    assert last["native_generation_tokens"] == 4
    assert last["first_token_ts"] is None
    assert m5_d3.snapshot_native_metrics(SimpleNamespace(), Stats, 10) is None


def output(stats, ids, *, finished=False):
    return SimpleNamespace(
        metrics=stats,
        outputs=[SimpleNamespace(token_ids=ids)] if ids else [],
        num_cached_tokens=0,
        finished=finished,
    )


def test_only_first_and_last_real_chunks_are_copied_after_receipt(replay, monkeypatch):
    stats = Stats()

    async def generate(*args, **kwargs):
        # The first client delta contains two EngineCore tokens.
        yield output(stats, [11, 12])
        stats.last_token_ts = 1.4
        stats.num_generation_tokens = 3
        yield output(stats, [13])
        stats.last_token_ts = 1.5
        stats.num_generation_tokens = 4
        yield output(stats, [14])
        stats.last_token_ts = 1.6
        yield output(stats, [], finished=True)

    plan = {"request_id": "r", "prompt_token_ids": [1], "cache_salt": "s"}
    row = replay._entry(plan, time.monotonic_ns())
    row["native_metrics"] = {"first": None, "last": None}
    calls = []

    def copy(metrics, stats_type, received_ns):
        assert received_ns == row["last_token_ns"]
        calls.append(received_ns)
        return m5_d3.snapshot_native_metrics(metrics, stats_type, received_ns)

    monkeypatch.setattr(replay, "snapshot_native_metrics", copy)
    asyncio.run(
        replay._consume(
            SimpleNamespace(generate=generate),
            plan,
            row,
            SimpleNamespace(max_tokens=4),
            row["planned_ns"] + 1_000_000_000,
            native_stats_type=Stats,
        )
    )
    assert row["status"] == "completed"
    assert calls == [row["first_token_ns"], row["last_token_ns"]]
    assert row["native_metrics"]["first"]["native_generation_tokens"] == 2
    assert row["native_metrics"]["last"]["native_generation_tokens"] == 4
    assert row["native_metrics"]["last"]["last_token_ts"] == 1.5


def test_partial_request_retains_missing_last_native_timing(replay):
    async def generate(*args, **kwargs):
        yield output(Stats(), [11, 12])

    plan = {"request_id": "partial", "prompt_token_ids": [1], "cache_salt": "s"}
    row = replay._entry(plan, time.monotonic_ns())
    row["native_metrics"] = {"first": None, "last": None}
    asyncio.run(
        replay._consume(
            SimpleNamespace(generate=generate),
            plan,
            row,
            SimpleNamespace(max_tokens=4),
            row["planned_ns"] + 1_000_000_000,
            native_stats_type=Stats,
        )
    )
    assert row["status"] == "failed"
    assert row["token_ids"] == [11, 12]
    assert row["native_metrics"]["first"]["native_generation_tokens"] == 2
    assert row["native_metrics"]["last"] is None


def test_native_sources_bind_loaded_paths_and_bytes(tmp_path, monkeypatch):
    source, pins, imported = {}, {}, {}
    for index, (name, module) in enumerate(m5_d3.SOURCE_MODULES.items()):
        path = tmp_path / f"source-{index}.py"
        path.write_text(name)
        pins[name] = digest(path)
        source[name] = {"sha256": pins[name], "path": str(path)}
        imported[module] = SimpleNamespace(__file__=str(path), RequestStateStats=Stats)
    monkeypatch.setattr(m5_d3, "import_module", imported.__getitem__)
    bound = {"upstream_modules": source, "upstream_sha256": pins.copy()}
    assert m5_d3.bind_loaded_sources(bound, pins) is Stats
    Path(imported["vllm.outputs"].__file__).write_text("changed")
    with pytest.raises(ValueError, match="loaded timing source differs: vllm/outputs.py"):
        m5_d3.bind_loaded_sources(bound, pins)


def test_clock_requires_five_same_domain_brackets_and_drain_order(monkeypatch):
    info = SimpleNamespace(
        implementation="clock_gettime(CLOCK_MONOTONIC)",
        monotonic=True,
        adjustable=False,
        resolution=1e-9,
    )
    monkeypatch.setattr(m5_d3, "time", SimpleNamespace(get_clock_info=lambda name: info))
    assert m5_d3.monotonic_clock_info()["resolution_ns"] == 1
    samples = [
        {"send_ns": i * 10, "engine_ns": i * 10 + 1, "recv_ns": i * 10 + i} for i in range(1, 6)
    ]
    assert m5_d3.verify_clock_brackets(samples, 3, after_ns=10) == samples[0]
    with pytest.raises(ValueError, match="native monotonic domain"):
        m5_d3.verify_clock_brackets(samples, 3, after_ns=11)
    samples[4]["engine_ns"] = 100
    with pytest.raises(ValueError, match="native monotonic domain"):
        m5_d3.verify_clock_brackets(samples, 3)
