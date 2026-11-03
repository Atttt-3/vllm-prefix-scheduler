"""Verify D5 effect classification, exact throughput margins and profile scope."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "configs/m5-d5.json").read_text())
TARGET = PLAN["target"]
CRITERIA = PLAN["assessment"]
NS = 1_000_000_000


@pytest.fixture
def d5(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("m5_d5_audit")


def timing(wait_ms, *, spacing_ms=90):
    wait = int(wait_ms * 1_000_000)
    count = TARGET["count_per_engine"]
    return {
        "status": "complete",
        "clock": {"verified": True},
        "planned_count": count,
        "first_decomposed_count": count,
        "terminal_decomposed_count": count,
        "per_id": {
            f"request-{i:06}": {
                "planned_ns": i * spacing_ms * 1_000_000,
                "submit_ns": i * spacing_ms * 1_000_000,
                "first_components_ns": {
                    "submit_to_native_queued": 0,
                    "native_queued_to_first_scheduled": wait,
                    "native_first_scheduled_to_first_token": 100_000_000,
                },
                "end_components_ns": {"native_first_to_last_token": 220_000_000},
            }
            for i in range(count)
        },
    }


def metrics(ttft_ms, window=8 * NS, *, tokens=960):
    return {
        "strata": {"all": {"ttft_ns": {"p95": ttft_ms * 1_000_000}}},
        "duration_ns": {"last_terminal_from_start": window},
        "throughput": {
            "completed_requests": 60,
            "generated_tokens": tokens,
            "last_terminal_requests_per_s": 60 * NS / window,
            "last_terminal_tokens_per_s": tokens * NS / window,
        },
    }


def fixtures():
    runs = {
        f"E{i}": {
            "max_num_active_seqs": cap,
            "status": "admitted",
            "pre_t0_verified": True,
            "prelude": {"verified": True},
            "in_window": 65,
            "endpoint_evidence_complete": True,
            "native_timing": timing(350 if cap == 4 else 40),
            "metrics": metrics(640 if cap == 4 else 400),
        }
        for i, cap in enumerate(TARGET["measure_order_max_num_active_seqs"], 1)
    }
    pairs = [
        {
            "reference": first,
            "observed": second,
            "thermal": {"comparable": True},
            "performance": {"within_tolerance": True},
        }
        for first, second in TARGET["same_cap_pairs"]
    ]
    return runs, pairs


def compare(a, b, limits):
    return {"comparable": abs(a - b) <= 2}


def test_d5_requires_all_four_cross_cap_improvements(d5):
    runs, pairs = fixtures()
    result = d5.assess_d5(runs, pairs, TARGET, CRITERIA, {}, compare, released=True)
    assert result["optimization_signal"] == "supported"
    assert result["cap4_backlog_reproduced"] is True
    assert result["same_cap_gates_passed"] is True
    assert result["cross_cap_throughput_gate_passed"] is True
    assert result["qualification_started"] is False
    assert len(result["all_window_thermal_pairs"]) == 6
    assert [(row["reference"], row["observed"]) for row in result["cross_cap_contrasts"]] == [
        ("E1", "E2"),
        ("E1", "E3"),
        ("E4", "E2"),
        ("E4", "E3"),
    ]
    high = result["runs"]["E1"]["admission"]
    assert high["tail_wait_median_ns"] == 350_000_000
    assert high["peak_waiting"] == 4
    assert high["peak_native_output_envelopes"] == 4


@pytest.mark.parametrize(
    ("case", "signal"),
    [
        ("ttft_tie", "not_supported"),
        ("tail_tie", "not_supported"),
        ("throughput_loss", "not_supported"),
        ("same_cap_breach", "unverified"),
        ("cap4_tail_missing", "unverified"),
        ("cap4_peak_missing", "unverified"),
        ("cross_cap_heat", "unverified"),
        ("bad_clock", "unverified"),
        ("missing_leaf", "unverified"),
        ("unreleased", "unverified"),
    ],
)
def test_d5_fails_the_predeclared_gate_without_hiding_other_results(d5, case, signal):
    runs, pairs = fixtures()
    if case == "ttft_tie":
        runs["E3"]["metrics"] = metrics(640)
    elif case == "tail_tie":
        runs["E3"]["native_timing"] = timing(350)
    elif case == "throughput_loss":
        runs["E3"]["metrics"] = metrics(400, 8_500_000_000)
    elif case == "same_cap_breach":
        pairs[1]["performance"]["within_tolerance"] = False
    elif case == "cap4_tail_missing":
        runs["E4"]["native_timing"] = timing(299)
    elif case == "cap4_peak_missing":
        runs["E4"]["native_timing"] = timing(350, spacing_ms=120)
    elif case == "cross_cap_heat":
        runs["E2"]["in_window"] = runs["E3"]["in_window"] = 62
    elif case == "bad_clock":
        runs["E3"]["native_timing"]["clock"]["verified"] = False
    elif case == "missing_leaf":
        del runs["E4"]
    result = d5.assess_d5(runs, pairs, TARGET, CRITERIA, {}, compare, released=case != "unreleased")
    assert result["optimization_signal"] == signal
    if case == "cross_cap_heat":
        assert result["same_cap_gates_passed"] is True
        assert result["mutually_thermally_comparable"] is False
    if case == "missing_leaf":
        assert result["cap4_backlog_reproduced"] is None
        assert result["runs"]["E4"]["status"] == "unattempted"


def test_d5_exactly_five_percent_loss_passes_but_either_numerator_can_fail(d5):
    reference = metrics(640, 19 * NS)
    assert d5.throughput_guard(reference, metrics(400, 20 * NS), 0.05) == {
        "requests": True,
        "tokens": True,
    }
    assert d5.throughput_guard(reference, metrics(400, 20 * NS, tokens=959), 0.05) == {
        "requests": True,
        "tokens": False,
    }


def test_d5_rejects_cross_cap_or_reordered_same_cap_pairs(d5):
    runs, pairs = fixtures()
    pairs[0]["observed"] = "E2"
    with pytest.raises(ValueError, match="same-cap pair inventory"):
        d5.assess_d5(runs, pairs, TARGET, CRITERIA, {}, compare, released=True)


def test_sealed_d5_profile_requires_exact_cap_and_does_not_leak_into_formal(d5):
    seal = importlib.import_module("m5_seal")
    config = json.loads((ROOT / "configs/m5-performance.json").read_text())
    upstream = json.loads((ROOT / "configs/native-m2b.json").read_text())["source_sha256"]
    upstream.update(config["extra_upstream_sha256"])
    native_names = (
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.input_batch",
        "vllm.v1.worker.gpu.states",
        "vllm.v1.worker.gpu.async_utils",
        "vllm.v1.worker.gpu.model_states.default",
        "vllm.model_executor.models.qwen3",
        "vllm.v1.attention.backends.flash_attn",
    )
    policy = ("scheduler", "config", "probe", "wait_guard")
    git_files = {f"src/vllm_prefix_scheduler/{name}.py": "b" * 64 for name in policy}
    expected = {
        "max_num_seqs": 5,
        "max_num_active_seqs": 4,
        "max_num_batched_tokens": 2048,
        "num_gpu_blocks": 5553,
    }
    profile = {
        **json.loads((ROOT / config["baseline_config"]).read_text()),
        **expected,
        "block_size": 16,
        "attention_backend": ["FLASH_ATTN"],
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "compute_capability": [8, 6],
        "model_type": "vllm.model_executor.models.qwen3.Qwen3ForCausalLM",
        "model_dtype": "torch.bfloat16",
        "model_device": "cuda:0",
        "use_v2_model_runner": True,
        "pipeline_parallel_size": 1,
        "executor_backend": "uni",
        "scheduler_policy": "fcfs",
        "runner_type": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "scheduler_type": "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler",
        "sources": {
            name: {"sha256": upstream[name.replace(".", "/") + ".py"]} for name in native_names
        },
        "production_modules": {name: {"sha256": "b" * 64} for name in policy},
    }
    run = {"pair_id": "D5", "mode": "guarded", "tier": "endpoint", "mechanism": None}
    source = {"upstream_sha256": upstream}
    seal.validate_profile(profile, run, config, source, git_files, d5_expected_profile=expected)
    for name, wrong in {
        "max_num_active_seqs": 5,
        "max_num_seqs": 4,
        "num_gpu_blocks": 5552,
        "max_num_batched_tokens": 4096,
    }.items():
        with pytest.raises(ValueError, match="runtime profile differs"):
            seal.validate_profile(
                {**profile, name: wrong},
                run,
                config,
                source,
                git_files,
                d5_expected_profile=expected,
            )
    with pytest.raises(ValueError, match="profile expectations are required"):
        seal.validate_profile(profile, run, config, source, git_files)
    with pytest.raises(ValueError, match="runtime expectations differ"):
        seal.validate_profile(
            profile,
            {**run, "pair_id": "mixed-01"},
            config,
            source,
            git_files,
            d5_expected_profile=expected,
        )
