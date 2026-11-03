"""D3-only binding and scalar copies of pinned native endpoint timing."""

import math
import time
from importlib import import_module
from pathlib import Path

from .m5_campaign import digest, require

SOURCE_MODULES = {
    "vllm/outputs.py": "vllm.outputs",
    "vllm/v1/core/sched/scheduler.py": "vllm.v1.core.sched.scheduler",
    "vllm/v1/engine/__init__.py": "vllm.v1.engine",
    "vllm/v1/engine/async_llm.py": "vllm.v1.engine.async_llm",
    "vllm/v1/engine/core.py": "vllm.v1.engine.core",
    "vllm/v1/engine/output_processor.py": "vllm.v1.engine.output_processor",
    "vllm/v1/metrics/stats.py": "vllm.v1.metrics.stats",
}
CORE_TIMESTAMPS = ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")


def bind_loaded_sources(sources, pins):
    """Verify the imported files are the same bytes the collector sealed."""
    require(set(pins) == set(SOURCE_MODULES), "D3 timing source inventory differs")
    for name, module in SOURCE_MODULES.items():
        path = Path(import_module(module).__file__).resolve()
        bound = sources["upstream_modules"][name]
        require(
            path == Path(bound["path"]).resolve()
            and digest(path) == bound["sha256"] == pins[name] == sources["upstream_sha256"][name],
            f"D3 loaded timing source differs: {name}",
        )
    return import_module("vllm.v1.metrics.stats").RequestStateStats


def snapshot_native_metrics(stats, stats_type, client_received_ns):
    """Copy only scalar values, after the measured client receipt timestamp."""
    if type(stats) is not stats_type:
        return None

    snapshot = {"client_received_ns": client_received_ns}
    for key in CORE_TIMESTAMPS:
        value = getattr(stats, key, None)
        snapshot[key] = value if type(value) is float and math.isfinite(value) else None
    for name, key in (
        ("native_generation_tokens", "num_generation_tokens"),
        ("native_preemptions", "num_preemptions"),
    ):
        value = getattr(stats, key, None)
        snapshot[name] = value if type(value) is int else None
    return snapshot


def monotonic_clock_info():
    info = time.get_clock_info("monotonic")
    require(
        info.implementation == "clock_gettime(CLOCK_MONOTONIC)"
        and info.monotonic is True
        and info.adjustable is False
        and 0 < info.resolution <= 1e-6,
        "D3 clock implementation cannot bind native timestamps to the client",
    )
    return {
        "implementation": info.implementation,
        "monotonic": info.monotonic,
        "adjustable": info.adjustable,
        "resolution_ns": info.resolution * 1_000_000_000,
        "post_clock_check": None,
        "post_clock_samples": [],
    }


def verify_clock_brackets(samples, max_round_trip_ns, *, after_ns=0):
    previous = after_ns
    require(len(samples) == 5, "D3 native timing requires five clock brackets")
    for row in samples:
        send, engine, recv = (row[key] for key in ("send_ns", "engine_ns", "recv_ns"))
        require(
            all(type(value) is int and value > 0 for value in (send, engine, recv))
            and previous <= send <= engine <= recv,
            "D3 clock bracket differs from the native monotonic domain",
        )
        previous = recv
    shortest = min(samples, key=lambda row: row["recv_ns"] - row["send_ns"])
    require(
        shortest["recv_ns"] - shortest["send_ns"] <= max_round_trip_ns,
        "D3 native monotonic clock bracket exceeds the frozen bound",
    )
    return shortest
