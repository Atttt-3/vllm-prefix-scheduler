"""Independent, fail-closed M2A matrix identity; no native imports.

This does not weaken or reuse the M1 matrix's four-condition contract.
"""

import re

UPSTREAM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
SCENARIOS = {
    "bc_pressure": ("native", "off", "prefix"),
    "bc_ample": ("native", "off", "prefix"),
    "apc_off": ("native", "off"),
    "probe": ("prefix",),
    "window": ("prefix",),
    "chunk": ("native", "off", "prefix"),
    "preemption": ("native", "off", "prefix"),
    "cancellation": ("native", "off", "prefix"),
    "id_reuse": ("native", "off", "prefix"),
    "score_eviction": ("prefix",),
    "global_rejection": ("prefix",),
}

REQUIRED_SOURCES = (
    "vllm/config/scheduler.py",
    "vllm/config/vllm.py",
    "vllm/v1/core/sched/async_scheduler.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/sched/request_queue.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/kv_cache_metrics.py",
    "vllm/v1/request.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/config/kv_events.py",
    "vllm/distributed/kv_events.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/outputs.py",
    "vllm/utils/hashing.py",
    "vllm/config/offload.py",
    "vllm/config/parallel.py",
    "vllm/config/observability.py",
    "vllm/config/device.py",
    "vllm/config/model.py",
    "vllm/config/aux_output.py",
    "vllm/config/lora.py",
    "vllm/config/multimodal.py",
    "vllm/config/speculative.py",
    "vllm/config/kv_transfer.py",
    "vllm/config/ec_transfer.py",
)


def require_assertions():
    """Also protect native assertions and callers from optimized interpreters."""
    if not __debug__:
        raise RuntimeError(
            "M2A validation requires assertions enabled; remove -O/-OO and PYTHONOPTIMIZE"
        )


def require(condition, message):
    """Validation is never removed by Python's optimization flags."""
    if not condition:
        raise ValueError(f"M2A: {message}")


def expected_cases(config):
    """Validate frozen baseline and return the entire independently required matrix."""
    require_assertions()
    require(isinstance(config, dict), "matrix config must be an object")
    repeats = config.get("repeats")
    require(type(repeats) is int and repeats >= 3, "repeats must be an integer >= 3 (not bool)")
    scenarios = config.get("scenarios")
    require(isinstance(scenarios, dict), "scenarios must be the complete named object")
    require(set(scenarios) == set(SCENARIOS), "scenario names do not match required matrix")
    for name, modes in SCENARIOS.items():
        actual = scenarios[name]
        require(
            isinstance(actual, list)
            and len(actual) == len(modes)
            and all(type(mode) is str for mode in actual)
            and set(actual) == set(modes),
            f"scenario {name} must contain each required mode exactly once",
        )
    require(config.get("vllm_version") == "0.32.0", "vLLM version must be 0.32.0")
    require(
        config.get("vllm_commit") == UPSTREAM_COMMIT, "upstream commit is not the fixed baseline"
    )
    require(
        config.get("modes") == ["native", "off", "prefix"],
        "matrix mode catalog must be native/off/prefix",
    )
    require(
        type(config.get("max_wait_ms")) in (int, float) and config["max_wait_ms"] == 500,
        "max_wait_ms contract must be 500",
    )
    lifecycle = {
        "chunk": {
            "num_blocks": 10,
            "max_num_batched_tokens": 16,
            "max_num_active_seqs": 1,
            "prompt_length": 65,
        },
        "preempt": {
            "num_blocks": 5,
            "max_num_batched_tokens": 64,
            "max_num_active_seqs": 2,
            "prompt_length": 32,
            "max_tokens": 3,
            "watermark": 0.0,
            "pending_output_depth": 2,
        },
    }
    require(config.get("lifecycle") == lifecycle, "lifecycle fixtures differ from frozen contract")
    for scenario, parameters in lifecycle.items():
        for field, value in parameters.items():
            actual = config["lifecycle"][scenario][field]
            require(
                type(actual) in ((int, float) if field == "watermark" else (int,))
                and actual == value,
                "lifecycle fixture has nonnumeric or bool field",
            )
    required_values = {
        "seed": 42,
        "block_size": 16,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 2,
        "max_num_active_seqs": 2,
        "max_model_len": 128,
        "max_tokens": 3,
        "pending_output_depth": 2,
        "long_prefill_token_threshold": 0,
        "synthetic_output_token_id": 9001,
        "window": 16,
    }
    for key, value in required_values.items():
        require(type(config.get(key)) is int and config[key] == value, f"{key} must be {value}")
    for key in ("enable_chunked_prefill", "ignore_eos", "scheduler_reserve_full_isl"):
        require(config.get(key) is True, f"{key} must be true")
    require(
        config.get("long_prefill_token_threshold_adaptive") is False,
        "adaptive threshold must be false",
    )
    require(
        type(config.get("watermark")) in (int, float) and config["watermark"] == 0,
        "watermark must be numeric zero",
    )
    capacities = config.get("capacities")
    require(
        isinstance(capacities, dict)
        and capacities == {"pressure": 5, "ample": 10}
        and all(type(value) is int for value in capacities.values()),
        "capacities must be pressure=5 and ample=10 total blocks",
    )
    require(
        type(config.get("max_steps_per_phase")) is int and config["max_steps_per_phase"] > 0,
        "max_steps_per_phase must be a positive integer",
    )
    requests = config.get("requests")
    require(isinstance(requests, list) and len(requests) == 3, "frozen A/B/C requests are required")
    token_ids = {
        "A": [*range(100, 132), 200],
        "B": list(range(300, 349)),
        "C": [*range(100, 132), 201],
    }
    for request, (rid, tokens) in zip(requests, token_ids.items(), strict=True):
        require(isinstance(request, dict), "each request must be an object")
        require(
            request.get("id") == rid and request.get("tokens") == tokens, "A/B/C token input drift"
        )
        require(
            all(type(token) is int for token in request["tokens"]),
            "token IDs must be strict integers",
        )
        arrival = request.get("arrival_time")
        require(
            type(arrival) in (int, float) and arrival == {"A": 0, "B": 1, "C": 2}[rid],
            "A/B/C arrival input drift",
        )
    sources = config.get("source_sha256")
    require(
        isinstance(sources, dict) and set(REQUIRED_SOURCES) <= set(sources),
        "native source manifest is incomplete",
    )
    require(
        all(
            type(value) is str and re.fullmatch("[0-9a-f]{64}", value) for value in sources.values()
        ),
        "native source hashes must be lowercase SHA-256 values",
    )
    require(
        type(config.get("model_config_sha256")) is str
        and re.fullmatch("[0-9a-f]{64}", config["model_config_sha256"]),
        "model metadata fingerprint is missing",
    )
    return {
        f"{repeat}:{scenario}:{mode}": {"repeat": repeat, "scenario": scenario, "mode": mode}
        for repeat in range(repeats)
        for scenario, modes in SCENARIOS.items()
        for mode in modes
    }


def verify_result_cases(results, config):
    """Reject partial/duplicate results before any success summary is written."""
    expected = expected_cases(config)
    require(isinstance(results, list) and results, "matrix results must be nonempty")
    seen = set()
    for result in results:
        require(isinstance(result, dict), "result must be an object")
        case = result.get("case")
        require(isinstance(case, str) and case in expected, "unexpected result case")
        require(case not in seen, "duplicate result case")
        require(type(result.get("repeat")) is int, "result repeat must be a strict integer")
        require(
            all(result.get(key) == value for key, value in expected[case].items()),
            "result identity mismatch",
        )
        require(result.get("status") == "passed", "result is failed, skipped or incomplete")
        seen.add(case)
    require(seen == set(expected), "result cases do not match complete matrix")
    return expected
