"""Frozen M2B leaf matrix and input gate, independent of production policy.

Every leaf is rebuilt independently, including negative configuration cases.
Historical M2A inputs and its guarded-unimplemented contract remain untouched.
"""

import hashlib
import json

UPSTREAM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
BASELINE_COMMIT = "a776cd3467ce1697f9dfd6bf8b858f1952a34de2"
SCENARIOS = {
    "bc_pressure": ("native", "off", "prefix", "guarded"),
    "bc_ample": ("native", "off", "prefix", "guarded"),
    "apc_off": ("native", "off"),
    "probe": ("prefix", "guarded"),
    "window": ("prefix", "guarded"),
    "score_eviction": ("prefix", "guarded"),
    "chunk": ("native", "off", "prefix", "guarded"),
    "preemption": ("native", "off", "prefix", "guarded"),
    "cancellation": ("native", "off", "prefix", "guarded"),
    "id_reuse": ("native", "off", "prefix", "guarded"),
    "global_rejection": ("prefix", "guarded"),
    "clock_thresholds": ("guarded",),
    "clock_absence": ("off", "prefix"),
    "enqueue_identity": ("guarded",),
    "protected_order": ("guarded",),
    "window_age": ("guarded",),
    "allocation_retry": ("guarded",),
    "hot_flow": ("native", "off", "prefix", "guarded"),
    "metadata_terminal": ("guarded",),
    "shutdown": ("guarded",),
    "repeat_cycles": ("guarded",),
    "postcheck_failure": ("prefix", "guarded"),
}

WINDOW_VARIANTS = (
    "stable_ties",
    "all_cold",
    "window_one",
    "window_limit",
    "head_barrier",
    "middle_barrier",
    "preempt_history",
    "partial_progress",
    "placeholder",
    "in_flight",
    "stale",
    "resumable",
    "unsupported",
    "prepend_order",
    "score_exception",
    "pop_exception",
    "prepend_exception",
    "repeated_step",
    "other_queues",
)
PROBE_VARIANTS = ("cold", "partial", "full", "exact16", "exact32", "one", "skip_read")
REJECTION_VARIANTS = (
    "apc_off",
    "async_off",
    "priority",
    "tensor_parallel",
    "pipeline_parallel",
    "data_parallel",
    "decode_parallel",
    "prefill_parallel",
    "aux_output",
    "hash_block",
    "multiple_groups",
    "noncausal",
    "window1_apc_off",
    "cuda_only",
    "lora",
    "kv_offload",
    "cpu_offload",
    "prefetch_offload",
    "pooling",
    "expert_parallel",
    "dbo",
    "sliding_window",
    "attention_chunk",
    "speculative",
    "multimodal",
    "kv_connector",
    "ec_connector",
    "encoder_decoder",
    "diffusion",
    "nonfull_spec",
    "external_lb",
    "hybrid_lb",
)
CLOCK_VARIANTS = (
    "before",
    "exact",
    "after",
    "zero",
    "negative_zero",
    "min_positive_float",
    "float_exact",
    "float_fraction_before",
    "float_fraction_at",
    "float_max",
    "huge_integer",
    "signed_epoch",
    "fault_bool",
    "fault_float",
    "fault_string",
    "fault_backward",
    "fault_provider_add",
    "fault_provider_decision",
)
VARIANTS = {
    "probe": PROBE_VARIANTS,
    "window": WINDOW_VARIANTS,
    "global_rejection": REJECTION_VARIANTS,
    "clock_thresholds": CLOCK_VARIANTS,
    "enqueue_identity": (
        "same_object_duplicate",
        "different_object_duplicate",
        "add_failure_prebind",
        "add_failure_postbind",
        "session_update",
        "initial_session_id",
        "initial_resumable",
        "initial_abort_immediately",
    ),
    "protected_order": ("prior_reorder", "same_tick"),
    "window_age": ("outside_then_enter",),
    "allocation_retry": ("natural_twice",),
    "metadata_terminal": (
        "str",
        "list",
        "generator",
        "none",
        "unknown_duplicate",
        "free_error",
        "generator_raise",
        "batch_free_error",
        "schedule_before_update_error",
        "schedule_after_update_error",
        "fault_weakref_release",
    ),
    "shutdown": ("waiting", "running", "raises"),
    "postcheck_failure": ("queue", "coordinator", "full_attention", "block_size", "v2"),
}


def leaf_catalog():
    """Return a fresh exact catalog; callers cannot shrink a shared mutable list."""
    catalog = []
    for scenario, modes in SCENARIOS.items():
        for mode in modes:
            variants = VARIANTS.get(scenario, ("default",))
            if scenario in {"bc_pressure", "bc_ample"} and mode == "guarded":
                variants = ("unprotected", "protected")
            if scenario == "window" and mode == "guarded":
                variants += ("missing_singleton", "missing_window_one")
            if scenario == "preemption" and mode == "guarded":
                variants += ("preempt_free_error", "preempt_after_transition_error")
            if scenario == "id_reuse" and mode == "guarded":
                variants += ("stale_object_cleanup",)
            for variant in variants:
                catalog.append({"scenario": scenario, "mode": mode, "variant": variant})
    return catalog


def require_assertions():
    if not __debug__:
        raise RuntimeError(
            "M2B validation requires assertions enabled; remove -O/-OO and PYTHONOPTIMIZE"
        )


def require(condition, message):
    if not condition:
        raise ValueError(f"M2B: {message}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# Fixed at creation from the accepted M2A input, not taken from supplied config.
# This covers token inputs, native source fingerprints, model metadata and geometry.
BASELINE_FIELDS = (
    "seed",
    "vllm_version",
    "vllm_commit",
    "block_size",
    "capacities",
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_num_active_seqs",
    "max_model_len",
    "enable_chunked_prefill",
    "watermark",
    "max_tokens",
    "ignore_eos",
    "synthetic_output_token_id",
    "pending_output_depth",
    "max_steps_per_phase",
    "requests",
    "source_sha256",
    "scheduler_reserve_full_isl",
    "long_prefill_token_threshold",
    "long_prefill_token_threshold_adaptive",
    "model_config_sha256",
    "window",
    "max_wait_ms",
    "lifecycle",
)
BASELINE_SHA256 = "31f3daee954da564f4b428792e0c4957fe4706a9513b0437bcd6d0ceb375458e"
CLOCK_PLAN_SHA256 = "ee12306bfdbbf8b42f3be6aff8e8c4b81e9ad653f5bfd4ba31c409eda1de6935"


COMPAT_SCENARIOS = frozenset(
    (
        "bc_pressure",
        "bc_ample",
        "apc_off",
        "probe",
        "window",
        "score_eviction",
        "chunk",
        "preemption",
        "cancellation",
        "id_reuse",
        "global_rejection",
    )
)


def expected_cases(config, suite="full"):
    require_assertions()
    require(type(config) is dict, "matrix config must be an object")
    repeats = config.get("repeats")
    require(type(repeats) is int and repeats >= 3, "repeats must be an integer >= 3")
    require(config.get("task_id") == "M2B-wait-guard", "wrong milestone identity")
    require(config.get("schema_version") == 2, "M2B matrix schema must be 2")
    require(config.get("baseline_commit") == BASELINE_COMMIT, "accepted baseline changed")
    require(config.get("modes") == ["native", "off", "prefix", "guarded"], "mode catalog differs")
    require(
        canonical(config.get("scenarios")) == canonical(SCENARIOS),
        "complete 50-combination scenario catalog differs",
    )
    require(
        config.get("leaves") == leaf_catalog(),
        "named leaf catalog is missing, extra, duplicated, reordered or mislabeled",
    )
    require(set(BASELINE_FIELDS) <= set(config), "frozen native baseline fields are missing")
    require(
        hashlib.sha256(canonical({k: config[k] for k in BASELINE_FIELDS}).encode()).hexdigest()
        == BASELINE_SHA256,
        "frozen native baseline/input/source configuration drift",
    )
    require(
        hashlib.sha256(canonical(config.get("clock_plans")).encode()).hexdigest()
        == CLOCK_PLAN_SHA256,
        "frozen clock plan differs",
    )
    require(suite in {"full", "compat"}, "suite must be full or compat")
    leaves = leaf_catalog()
    if suite == "compat":
        leaves = [
            leaf
            for leaf in leaves
            if leaf["scenario"] in COMPAT_SCENARIOS
            and leaf["variant"]
            not in {
                "protected",
                "missing_singleton",
                "missing_window_one",
                "preempt_free_error",
                "preempt_after_transition_error",
                "stale_object_cleanup",
            }
        ]
    return {
        f"{repeat}:{leaf['scenario']}:{leaf['mode']}:{leaf['variant']}": {"repeat": repeat, **leaf}
        for repeat in range(repeats)
        for leaf in leaves
    }


def verify_result_cases(results, config, suite="full"):
    expected = expected_cases(config, suite=suite)
    require(type(results) is list and bool(results), "results must be a nonempty list")
    seen = set()
    for result in results:
        require(type(result) is dict, "result must be an object")
        case = result.get("case")
        require(type(case) is str and case in expected, "unexpected result case")
        require(case not in seen, "duplicate result case")
        require(type(result.get("repeat")) is int, "repeat must be strict integer")
        require(
            all(result.get(k) == v for k, v in expected[case].items()), "result identity mismatch"
        )
        require(result.get("status") == "passed", "result failed, skipped or incomplete")
        seen.add(case)
    require(seen == set(expected), "result matrix is incomplete")
    return expected
